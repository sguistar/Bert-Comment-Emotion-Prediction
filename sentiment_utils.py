# -*- coding: utf-8 -*-
"""Shared helpers for movie review sentiment modeling and reputation analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
DATA_DIR = ASSETS_DIR / "data"
PICTURES_DIR = ASSETS_DIR / "pictures"
DATA_PATH = DATA_DIR / "cleaned_data.csv"
OUTPUT_DIR = ROOT / "outputs" / "sentiment"
BEST_MODEL_SPLIT_DIR = OUTPUT_DIR / "bert-v2" / "splits"
SPLIT_DIR = OUTPUT_DIR / "splits"
ML_DIR = OUTPUT_DIR / "ml"
BERT_DIR = OUTPUT_DIR / "bert"
REPUTATION_DIR = OUTPUT_DIR / "reputation"
VALIDATION_DIR = OUTPUT_DIR / "validation"

LABEL_ORDER = ["负向", "中性", "正向"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

TEXT_COL = "清洗评论"
TOKEN_COL = "分词结果"
TARGET_COL = "情感标签"

LEAKAGE_COLUMNS = {"评分", "评分_5分制"}
RANDOM_STATE = 42


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, SPLIT_DIR, ML_DIR, BERT_DIR, REPUTATION_DIR, VALIDATION_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_cleaned_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"电影名称", TEXT_COL, TOKEN_COL, TARGET_COL, "评分_5分制"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"cleaned_data.csv 缺少必要字段: {missing}")
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df[TOKEN_COL] = df[TOKEN_COL].fillna("").astype(str)
    df[TARGET_COL] = df[TARGET_COL].astype(str)
    invalid_labels = sorted(set(df[TARGET_COL]) - set(LABEL_ORDER))
    if invalid_labels:
        raise ValueError(f"发现未知情感标签: {invalid_labels}")
    if df[TARGET_COL].isna().any() or df[TARGET_COL].eq("").any():
        raise ValueError("情感标签存在缺失或空值")
    return df


def load_best_model_dashboard_data(
    split_dir: Path = BEST_MODEL_SPLIT_DIR,
) -> pd.DataFrame:
    """Rebuild chart features for the best model's train+valid rows only.

    The frozen test split is intentionally never opened here.
    """

    import prepare_sentiment_v2 as preparation

    model_frames: list[pd.DataFrame] = []
    for split_name in ("train", "valid"):
        split_path = split_dir / f"{split_name}.csv"
        frame = pd.read_csv(split_path, encoding="utf-8-sig")
        missing = {"text", "label", "movie", "row_id", "source"} - set(
            frame.columns
        )
        if missing:
            raise ValueError(
                f"{split_path.name} 缺少最佳模型数据列: {sorted(missing)}"
            )
        frame = frame.copy()
        frame["model_split"] = split_name
        model_frames.append(frame)
    model_rows = pd.concat(model_frames, ignore_index=True)
    if model_rows["row_id"].duplicated().any():
        raise ValueError("最佳模型 train/valid 存在重复 row_id")

    raw = preparation.load_raw_data(preparation.CANONICAL_INPUT)
    structured, _cleaning_summary = preparation.clean_structured_data(raw)
    featured, _feature_summary = preparation.engineer_features(structured)
    featured = featured.copy()
    featured["row_id"] = [
        preparation.stable_row_id(
            preparation.CANONICAL_SOURCE,
            row[preparation.COL_MOVIE],
            row[preparation.COL_USER],
            row[preparation.COL_SCORE],
            row.get("评分_5分制", ""),
            row[preparation.COL_TIME],
            row[preparation.COL_TEXT],
        )
        for _, row in featured.iterrows()
    ]
    if featured["row_id"].duplicated().any():
        raise ValueError("重建图表特征时产生重复 row_id")

    model_lookup = model_rows.set_index("row_id")
    dashboard = featured.loc[featured["row_id"].isin(model_lookup.index)].copy()
    missing_ids = sorted(set(model_lookup.index) - set(dashboard["row_id"]))
    if missing_ids:
        raise ValueError(f"有 {len(missing_ids)} 条最佳模型数据无法重建图表特征")
    dashboard["model_split"] = dashboard["row_id"].map(
        model_lookup["model_split"]
    )

    aligned = dashboard.set_index("row_id").loc[model_lookup.index]
    rebuilt_text = aligned[TEXT_COL].map(preparation.clean_comment_text).astype(str)
    text_mismatches = int(
        rebuilt_text.ne(model_lookup["text"].astype(str)).sum()
    )
    label_mismatches = int(
        aligned[TARGET_COL].astype(str).ne(model_lookup["label"].astype(str)).sum()
    )
    movie_mismatches = int(
        aligned["电影名称"].astype(str).ne(model_lookup["movie"].astype(str)).sum()
    )
    if text_mismatches or label_mismatches or movie_mismatches:
        raise ValueError(
            "最佳模型图表数据校验失败: "
            f"text={text_mismatches}, label={label_mismatches}, movie={movie_mismatches}"
        )

    dashboard = dashboard.sort_values(
        ["model_split", "row_id"], kind="stable"
    ).reset_index(drop=True)
    dashboard[TEXT_COL] = dashboard[TEXT_COL].fillna("").astype(str)
    dashboard[TOKEN_COL] = dashboard[TOKEN_COL].fillna("").astype(str)
    dashboard[TARGET_COL] = dashboard[TARGET_COL].astype(str)
    if len(dashboard) != len(model_rows):
        raise ValueError(
            f"最佳模型图表数据行数错误: {len(dashboard)} != {len(model_rows)}"
        )
    return dashboard


def build_tfidf_text(df: pd.DataFrame) -> pd.Series:
    token_text = df[TOKEN_COL].fillna("").astype(str).str.strip()
    clean_text = df[TEXT_COL].fillna("").astype(str).str.strip()
    model_text = token_text.where(token_text.ne(""), clean_text)
    return model_text.where(model_text.ne(""), clean_text)


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COL],
    )
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLIT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(SPLIT_DIR / "valid.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False, encoding="utf-8-sig")


def distribution_table(*named_frames: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in named_frames:
        counts = frame[TARGET_COL].value_counts().reindex(LABEL_ORDER, fill_value=0)
        total = int(counts.sum())
        row = {"split": name, "total": total}
        for label in LABEL_ORDER:
            row[f"{label}_count"] = int(counts[label])
            row[f"{label}_rate"] = round(float(counts[label] / total), 4) if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_predictions(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def save_classification_report(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    output_path: Path,
) -> pd.DataFrame:
    report = classification_report(
        list(y_true),
        list(y_pred),
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    df.to_csv(output_path, encoding="utf-8-sig")
    return df


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    output_path: Path,
    title: str,
) -> None:
    configure_chinese_font()
    matrix = confusion_matrix(list(y_true), list(y_pred), labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_xticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value > matrix.max() * 0.55 else "#172033"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_comment_for_prediction(text: str) -> str:
    from feature_engineering import clean_comment_text, tokenize_text

    clean_text = clean_comment_text(text)
    tokens = tokenize_text(clean_text)
    return " ".join(tokens) if tokens else clean_text


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0 or math.isnan(denominator):
        return 0.0
    return float(numerator / denominator)
