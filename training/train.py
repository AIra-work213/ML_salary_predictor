import logging
import os
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data_cleaning import CleaningConfig, build_clean_training_data, save_cleaning_artifacts

MODEL_NAME = "cointegrated/rubert-tiny2"
MODEL_PATH = Path("model_weights.pth")
ARTIFACTS_DIR = Path("artifacts")
TRAIN_SAMPLE_SIZE = 40_000
VAL_SIZE = 8_000
CLEANING_CFG = CleaningConfig(
    min_salary=20_000,
    max_salary=400_000,
    min_description_len=300,
    min_skills_count=2,
    exclude_no_experience=False,
)

# Настройка логгера
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SalaryDataset(Dataset):
    '''Dataset для текстов вакансий и их целевых зарплат.'''
    def __init__(self, texts: list[str], targets: np.ndarray):
        self.texts = texts
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        return self.texts[idx], torch.tensor(self.targets[idx], dtype=torch.float32)


class BatchTokenizerCollator:
    '''DataLoader, который токенизирует батчи текстов с помощью заданного токенизатора.'''
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 224):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*batch)
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.stack(list(labels), dim=0)
        return enc


class BertRegressor(torch.nn.Module):
    '''Модель на основе Bert с головой для регрессии на 2 выхода (минимальная и максимальная зарплата).'''
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        hidden_size = self.bert.config.hidden_size
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 2),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_embedding)



def freeze_bert_except_last_layer(model: BertRegressor) -> None:
    '''Замораживает все слои BERT, кроме последнего encoder layer и pooler, для более эффективного обучения головы.'''
    for param in model.bert.parameters(): param.requires_grad = False
    for p in model.bert.encoder.layer[-1].parameters(): p.requires_grad = True
    for p in model.bert.pooler.parameters(): p.requires_grad = True


def load_compatible_state_dict(model: BertRegressor, model_path: Path) -> None:
    '''Загружает state_dict из файла, автоматически сопоставляя ключи, если структура модели изменилась :/'''
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    remapped = {k.replace("ffnn.0.", "head.0.").replace("ffnn.2.", "head.3."): v for k, v in state.items()}
    model.load_state_dict(remapped, strict=False)



def get_or_build_cleaned_data() -> tuple[Path, int]:
    cleaned_csv = ARTIFACTS_DIR / "train_cleaned.csv"
    report_json = ARTIFACTS_DIR / "cleaning_report.json"
    cfg_dict = {
        "min_salary": CLEANING_CFG.min_salary,
        "max_salary": CLEANING_CFG.max_salary,
        "min_description_len": CLEANING_CFG.min_description_len,
        "min_skills_count": CLEANING_CFG.min_skills_count,
        "exclude_no_experience": CLEANING_CFG.exclude_no_experience,
        "random_state": CLEANING_CFG.random_state,
    }

    if cleaned_csv.exists() and report_json.exists():
        with report_json.open("r", encoding="utf-8") as f:
            report = json.load(f)
        if report.get("config") == cfg_dict:
            return cleaned_csv, int(report.get("final_rows", 0))

    cleaned_df, report = build_clean_training_data("train.csv", CLEANING_CFG)
    save_cleaning_artifacts(cleaned_df, report, ARTIFACTS_DIR)
    return ARTIFACTS_DIR / "train_cleaned.csv", int(report["final_rows"])


def load_rows_by_indices(csv_path: Path, indices: np.ndarray, chunksize: int = 20_000) -> pd.DataFrame:
    selected = np.sort(indices)
    parts, start = [], 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        left, right = np.searchsorted(selected, start), np.searchsorted(selected, start + len(chunk))
        if right > left:
            parts.append(chunk.iloc[selected[left:right] - start])
            if sum(len(p) for p in parts) >= len(selected): break
        start += len(chunk)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["text", "target_min", "target_max"])


def build_dataloader_from_df(tokenizer: AutoTokenizer, df: pd.DataFrame, shuffle: bool) -> DataLoader:
    targets = df[["target_min", "target_max"]].to_numpy()
    dataset = SalaryDataset(df["text"].tolist(), targets)

    workers = max(0, min(2, (os.cpu_count() or 2) // 2))
    loader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        collate_fn=BatchTokenizerCollator(tokenizer),
    )
    return loader



def train() -> None:
    set_seed(42)
    # Ограничиваем доступные ядра для pytorch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cleaned_csv_path, total_rows = get_or_build_cleaned_data()

    rng = np.random.default_rng(42)
    val_size = min(VAL_SIZE, max(1, total_rows // 10), total_rows - 1)
    all_indices = np.arange(total_rows, dtype=np.int64)
    val_indices = rng.choice(all_indices, size=val_size, replace=False)
    train_pool = np.setdiff1d(all_indices, val_indices, assume_unique=False)

    val_df = load_rows_by_indices(cleaned_csv_path, val_indices)
    val_loader = build_dataloader_from_df(tokenizer, val_df, shuffle=False)

    model = BertRegressor(MODEL_NAME)
    freeze_bert_except_last_layer(model)

    if MODEL_PATH.exists():
        load_compatible_state_dict(model, MODEL_PATH)

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    bert_params = [p for p in model.bert.parameters() if p.requires_grad]

    #  Оптимизатор с разными скоростями обучения для головы и последнего слоя BERT
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": 1e-3},
            {"params": bert_params, "lr": 1e-4},
        ],
        weight_decay=0.01,
    )
    #  Лосс MAE
    loss_fn = torch.nn.L1Loss()

    best_loss = float("inf")
    epochs = 4

    for epoch in range(epochs):
        epoch_rng = np.random.default_rng(42 + epoch)
        epoch_train_size = min(TRAIN_SAMPLE_SIZE, len(train_pool))
        epoch_indices = epoch_rng.choice(train_pool, size=epoch_train_size, replace=False)

        epoch_df = load_rows_by_indices(cleaned_csv_path, epoch_indices)
        train_loader = build_dataloader_from_df(tokenizer, epoch_df, shuffle=True)

        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            preds = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(preds, batch["labels"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch["input_ids"], batch["attention_mask"])
                val_loss = loss_fn(preds, batch["labels"])
                val_total_loss += val_loss.item()

        avg_val_loss = val_total_loss / max(1, len(val_loader))
        logger.info(
            "Epoch %s: train loss %.4f | val loss %.4f | train sample %s | val size %s",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            epoch_train_size,
            val_size,
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info("Лучшая модель сохранена %s", MODEL_PATH)

    logger.info("Обучение завершено")


if __name__ == "__main__":
    train()
