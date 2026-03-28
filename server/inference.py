import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel


class BertRegressor(torch.nn.Module):
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


class SalaryPredictor:
    """
    Класс для загрузки обученной модели BERT и выполнения инференса (предсказания зарплаты).
    Обеспечивает загрузку весов в память один раз при инициализации.
    """

    def __init__(self, model_name: str = "cointegrated/rubert-tiny2", model_path: str = "model_weights.pth"):
        """
        Инициализация модели и токенизатора.
        
        :param model_name: Имя трансформера (по умолчанию rubert-tiny2).
        :param model_path: Имя файла с весами обученной модели (относительно корня проекта).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertRegressor(model_name)
        
        # Полный путь к весам вычисляется от корня проекта, чтобы избежать проблем с CWD
        weights_path = Path(model_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Файл весов не найден по пути: {weights_path}")
            
        self.load_compatible_state_dict(self.model, weights_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict[str, float]:
        """
        Выполняет предсказание минимальной и максимальной зарплаты по тексту вакансии.
        
        :param text: Текст с описанием вакансии.
        :return: Словарь с ключами 'salary_min' и 'salary_max'.
        """
        # Токенизация с теми же параметрами, что и при обучении
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=224,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Инференс без градиентов для скорости и экономии памяти
        with torch.no_grad():
            preds = self.model(input_ids, attention_mask)
            
        # Возвращаем предсказанные значения
        preds = preds.squeeze(0).cpu().tolist()
        return {
            "salary_min": float(preds[0]),
            "salary_max": float(preds[1])
        }
    
    def load_compatible_state_dict(self, model: BertRegressor, model_path: Path) -> None:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        remapped = {k.replace("ffnn.0.", "head.0.").replace("ffnn.2.", "head.3."): v for k, v in state.items()}
        model.load_state_dict(remapped, strict=False)
