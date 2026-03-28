import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class CleaningConfig:
    min_salary: int = 20_000
    max_salary: int = 400_000
    min_description_len: int = 250
    min_skills_count: int = 1
    exclude_no_experience: bool = False
    random_state: int = 42


def _salary_to_range(raw_salary: Any) -> tuple[float, float] | None:
    if pd.isna(raw_salary):
        return None

    s = str(raw_salary)
    if s.strip() == "" or "з/п не указана" in s.lower():
        return None

    digits_only = "".join(ch if ch.isdigit() else " " for ch in s)
    numbers = [int(n) for n in digits_only.split() if n.isdigit()]

    if not numbers:
        return None
    if len(numbers) == 1:
        value = float(numbers[0])
        return value, value

    low, high = float(numbers[0]), float(numbers[1])
    if low > high:
        low, high = high, low
    return low, high


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _skills_count(skills: str) -> int:
    tokens = pd.Series([skills]).str.split(r"[,;\n]", regex=True).iloc[0]
    return len([t for t in tokens if str(t).strip()])


def build_clean_training_data(
    source_csv: str | Path,
    config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config or CleaningConfig()

    keep_cols = ["title", "salary", "experience", "description", "key_skills"]
    df = pd.read_csv(
        source_csv,
        sep=";",
        on_bad_lines="skip",
        usecols=lambda c: c in keep_cols,
    )

    report: dict[str, Any] = {
        "initial_rows": int(len(df)),
        "config": {
            "min_salary": cfg.min_salary,
            "max_salary": cfg.max_salary,
            "min_description_len": cfg.min_description_len,
            "min_skills_count": cfg.min_skills_count,
            "exclude_no_experience": cfg.exclude_no_experience,
            "random_state": cfg.random_state,
        },
    }

    for col in ["title", "salary", "experience", "description", "key_skills"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["title"] = df["title"].map(_normalize_text)
    df["experience"] = df["experience"].map(_normalize_text)
    df["description"] = df["description"].map(_normalize_text)
    df["key_skills"] = df["key_skills"].map(_normalize_text)

    df = df[df["title"] != ""]
    report["after_title_filter"] = int(len(df))

    salary_ranges = df["salary"].map(_salary_to_range)
    df = df.assign(_salary_range=salary_ranges)
    df = df[df["_salary_range"].notna()]
    report["after_salary_parse"] = int(len(df))

    df["target_min"] = df["_salary_range"].map(lambda x: x[0])
    df["target_max"] = df["_salary_range"].map(lambda x: x[1])
    df.drop(columns=["_salary_range"], inplace=True)

    salary_mask = (df["target_min"] >= cfg.min_salary) & (df["target_max"] <= cfg.max_salary)
    df = df[salary_mask]
    report["after_salary_bounds"] = int(len(df))

    if cfg.exclude_no_experience:
        df = df[df["experience"].str.lower() != "нет опыта"]
    df = df[df["experience"] != ""]
    report["after_experience_filter"] = int(len(df))

    df["description_len"] = df["description"].str.len()
    df = df[df["description_len"] >= cfg.min_description_len]
    report["after_description_filter"] = int(len(df))

    df["skills_count"] = df["key_skills"].map(_skills_count)
    df = df[df["skills_count"] >= cfg.min_skills_count]
    report["after_skills_filter"] = int(len(df))

    df = df.drop_duplicates(subset=["title", "experience", "description", "salary"])
    report["after_dedup"] = int(len(df))

    df["target_min"] = df["target_min"] / 1000.0
    df["target_max"] = df["target_max"] / 1000.0

    df["text"] = (
        "ДОЛЖНОСТЬ: "
        + df["title"]
        + " | ОПЫТ: "
        + df["experience"]
        + " | НАВЫКИ: "
        + df["key_skills"]
        + " | ОПИСАНИЕ: "
        + df["description"]
    )

    report["final_rows"] = int(len(df))
    report["retention_rate"] = round(
        (report["final_rows"] / report["initial_rows"]) if report["initial_rows"] else 0.0,
        4,
    )

    final_cols = [
        "text",
        "target_min",
        "target_max",
        "title",
        "experience",
        "description_len",
        "skills_count",
    ]
    return df[final_cols].reset_index(drop=True), report


def save_cleaning_artifacts(
    cleaned_df: pd.DataFrame,
    report: dict[str, Any],
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(output_path / "train_cleaned.csv", index=False)
    cleaned_df.head(500).to_csv(output_path / "train_cleaned_preview.csv", index=False)

    with (output_path / "cleaning_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
