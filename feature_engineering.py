# -*- coding: utf-8 -*-
"""
Douban movie comment data cleaning, feature engineering, text processing,
and visualization.

Run:
    python feature_engineering.py

Outputs:
    assets/data/cleaned_data.csv
    assets/pictures/*.png
    assets/pictures/processing_summary.json
"""

from __future__ import annotations

import csv
import html
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
DATA_DIR = ASSETS_DIR / "data"
PICTURES_DIR = ASSETS_DIR / "pictures"
RAW_PATH = DATA_DIR / "douban_comments.csv"
CLEANED_PATH = DATA_DIR / "cleaned_data.csv"
OUTPUT_DIR = PICTURES_DIR

# The file name and task scope are 2025-2026, so 2024 rows are treated as
# out-of-scope collection noise rather than valid analysis records.
MIN_YEAR = 2025
MAX_YEAR = 2026

# Raw Douban short-comment scores are stored as values like "50星".
# Dividing by 10 converts them to the familiar 1-5 star scale.
SCORE_DIVISOR = 10

# A 5-star scale is mapped to coarse sentiment labels. 4-5 stars are positive,
# 3 stars are neutral, and 1-2 stars are negative; this mirrors common rating
# analysis practice and avoids over-interpreting small score differences.
POSITIVE_SCORE_THRESHOLD = 4.0
NEGATIVE_SCORE_THRESHOLD = 2.0

# Long comments are defined by the empirical upper quartile so the flag adapts
# to this dataset instead of relying on an arbitrary fixed character count.
LONG_COMMENT_QUANTILE = 0.75

# The word cloud is a summary visual, not a vocabulary table. Keeping the top
# 80 terms balances coverage and readability in the exported PNG.
TOP_N_WORDS = 80

RANDOM_SEED = 20260630

COL_MOVIE = "电影名称"
COL_USER = "用户名"
COL_SCORE = "评分"
COL_TIME = "评论时间"
COL_IP = "IP地址"
COL_TEXT = "评论内容"

STOPWORDS = {
    "一个",
    "一些",
    "不是",
    "不能",
    "不过",
    "为了",
    "什么",
    "他们",
    "你们",
    "我们",
    "但是",
    "就是",
    "还是",
    "没有",
    "真的",
    "这个",
    "这种",
    "那个",
    "那么",
    "因为",
    "所以",
    "如果",
    "已经",
    "可以",
    "看到",
    "感觉",
    "电影",
    "影片",
    "导演",
    "故事",
    "角色",
    "自己",
    "时候",
    "还有",
    "以及",
    "或者",
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "movie",
    "film",
}

DOMAIN_TERMS = (
    "剧情拖沓",
    "演员出戏",
    "节奏紧凑",
    "情感真挚",
    "叙事混乱",
    "画面精美",
    "配乐出色",
    "演技在线",
    "剧情稀碎",
    "节奏拖沓",
)


def get_cjk_font(size: int) -> ImageFont.FreeTypeFont:
    """Return a Chinese-capable font for PNG charts."""
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV with a robust Chinese encoding strategy.

    GB18030/GBK can decode the source file, but a few bytes are malformed.
    encoding_errors="replace" keeps the row structure intact and surfaces
    suspicious characters for text cleaning instead of stopping the pipeline.
    """
    # 原始文件为中文 CSV，使用 GB18030 可覆盖 GBK 和更多中文字符。
    # encoding_errors="replace" 用于容忍少量异常字节，防止因单个坏字符中断整个预处理流程。
    return pd.read_csv(path, encoding="gb18030", encoding_errors="replace")


def parse_score(raw_score: object) -> float | None:
    """Convert raw score text such as '50星' to a 1-5 numeric score."""
    # 先判断空值，例如缺失评分或“暂无”都不能直接转为数值。
    if pd.isna(raw_score):
        return None
    text = str(raw_score).strip()
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    # 豆瓣原始评分如 50星、40星，除以 10 后统一为 1-5 分制。
    score = int(match.group(1)) / SCORE_DIVISOR
    if 1 <= score <= 5:
        return float(score)
    return None


def normalize_text_value(value: object) -> str:
    """Normalize whitespace and replacement characters in a scalar text value."""
    # 通用字符串规范化，后续用于用户名、电影名称和评论文本等字段。
    if pd.isna(value):
        return ""
    # 解码 HTML 实体，并删除编码替换符，减少无效噪声。
    text = html.unescape(str(value))
    text = text.replace("\uFFFD", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_comment_text(text: object) -> str:
    """Clean review text while preserving Chinese sentiment-bearing content."""
    # 文本先做基础规范化，英文统一转小写便于词频统计。
    value = normalize_text_value(text).lower()
    # 链接、@用户和话题符号对评论语义分析帮助较小，因此移除。
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"@[\w\u4e00-\u9fff_-]+", " ", value)
    value = re.sub(r"#[^#\s]+#", " ", value)
    # Keep Chinese, English, numbers, and common punctuation used for tone.
    # 保留中文、英文、数字和常见情绪标点，避免过度清洗损失情感信息。
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9，。！？!?、,.；;：:\s]", " ", value)
    value = re.sub(r"([!！?？])\1+", r"\1", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize_text(clean_text: str) -> list[str]:
    """Tokenize text with domain terms and a deterministic fallback."""
    tokens: list[str] = []
    clean_text = clean_text or ""
    text_for_cut = clean_text

    tokens.extend(re.findall(r"[A-Za-z]{2,}", clean_text.lower()))
    for term in DOMAIN_TERMS:
        if term in clean_text:
            tokens.append(term)
            text_for_cut = text_for_cut.replace(term, " ")

    try:
        import jieba

        for term in DOMAIN_TERMS:
            jieba.add_word(term)
        candidates = [token.strip().lower() for token in jieba.lcut(text_for_cut) if token.strip()]
    except Exception:
        candidates = []
        for chunk in re.findall(r"[一-鿿]+", text_for_cut):
            if len(chunk) == 2:
                candidates.append(chunk)
            elif len(chunk) > 2:
                candidates.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))

    for token in candidates:
        if len(token) >= 2 and token not in STOPWORDS:
            tokens.append(token)
    return [token for token in tokens if len(token) >= 2 and token not in STOPWORDS]

def clean_structured_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean rows, cast key fields, and remove invalid records."""
    # summary 记录每一步清洗前后的行数变化，便于报告或审核追溯。
    summary: dict[str, object] = {
        "raw_rows": int(len(raw_df)),
        "raw_columns": list(raw_df.columns),
    }

    # 复制原始数据，避免在函数内直接修改输入 DataFrame。
    df = raw_df.copy()
    # 先对核心文本字段统一去空格和异常字符，再做缺失与重复判断。
    for column in [COL_MOVIE, COL_USER, COL_SCORE, COL_TIME, COL_TEXT]:
        df[column] = df[column].map(normalize_text_value)

    before_missing = df[[COL_MOVIE, COL_USER, COL_SCORE, COL_TIME, COL_TEXT]].isna().sum()
    before_missing = before_missing.to_dict()
    before_missing["评分不可解析"] = int(df[COL_SCORE].map(parse_score).isna().sum())
    summary["before_missing"] = before_missing

    # 构造数值评分和 datetime 字段，这是后续分布图和时间特征的基础。
    df["评分_5分制"] = df[COL_SCORE].map(parse_score)
    df["评论时间_dt"] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df["原始评论长度"] = df[COL_TEXT].astype(str).str.len()

    key_subset = [COL_MOVIE, COL_USER, COL_TIME, COL_TEXT]
    # 以“电影+用户+时间+内容”作为评论粒度键，删除重复采集记录。
    duplicate_mask = df.duplicated(subset=key_subset, keep="first")
    summary["duplicate_rows_removed"] = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()

    # 无法解析的时间不能用于时间分析，因此记录并删除。
    invalid_time_mask = df["评论时间_dt"].isna()
    summary["invalid_time_rows_removed"] = int(invalid_time_mask.sum())
    df = df.loc[~invalid_time_mask].copy()

    # 按数据文件主题仅保留 2025-2026 年，剔除越界样本。
    out_of_scope_mask = ~df["评论时间_dt"].dt.year.between(MIN_YEAR, MAX_YEAR)
    summary["out_of_scope_year_rows_removed"] = int(out_of_scope_mask.sum())
    df = df.loc[~out_of_scope_mask].copy()

    # “暂无”等不可解析评分无法参与评分分布和情感标签计算，所以删除。
    unrated_mask = df["评分_5分制"].isna()
    summary["unrated_rows_removed"] = int(unrated_mask.sum())
    df = df.loc[~unrated_mask].copy()

    empty_required_mask = (
        df[COL_MOVIE].eq("") | df[COL_TIME].eq("") | df[COL_TEXT].eq("")
    )
    summary["empty_required_rows_removed"] = int(empty_required_mask.sum())
    df = df.loc[~empty_required_mask].copy()

    # 用户名不是分析必需度量，少量缺失用“未知用户”填充以保留评论内容。
    missing_user_mask = df[COL_USER].eq("")
    summary["missing_user_filled"] = int(missing_user_mask.sum())
    df.loc[missing_user_mask, COL_USER] = "未知用户"

    summary["ip_missing_rate"] = float(raw_df[COL_IP].isna().mean()) if COL_IP in raw_df else None
    # IP 地址列全部缺失，不能提供有效分析信息，故从清洗数据中移除。
    if COL_IP in df.columns:
        df = df.drop(columns=[COL_IP])

    summary["cleaned_rows"] = int(len(df))
    summary["rows_removed_total"] = int(summary["raw_rows"] - summary["cleaned_rows"])
    return df.reset_index(drop=True), summary


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create numerical, time, label, and text-derived features."""
    output = df.copy()
    # 拆分时间特征，便于按年、月、日期、小时等维度做统计。
    output["年份"] = output["评论时间_dt"].dt.year
    output["月份"] = output["评论时间_dt"].dt.month
    output["年月"] = output["评论时间_dt"].dt.to_period("M").astype(str)
    output["日期"] = output["评论时间_dt"].dt.date.astype(str)
    output["小时"] = output["评论时间_dt"].dt.hour
    output["星期"] = output["评论时间_dt"].dt.day_name()

    # 基于评分构造粗粒度情感标签，便于后续分组分析或建模。
    output["情感标签"] = output["评分_5分制"].apply(label_sentiment)
    # 生成清洗文本和分词结果，作为词云和文本特征的输入。
    output["清洗评论"] = output[COL_TEXT].map(clean_comment_text)
    output["分词列表"] = output["清洗评论"].map(tokenize_text)
    output["分词结果"] = output["分词列表"].map(lambda tokens: " ".join(tokens))

    # 计算文本长度、字符类型和标点数，衡量评论表达方式。
    output["评论长度"] = output[COL_TEXT].astype(str).str.len()
    output["清洗后长度"] = output["清洗评论"].astype(str).str.len()
    output["中文字符数"] = output["清洗评论"].map(
        lambda text: len(re.findall(r"[\u4e00-\u9fff]", text))
    )
    output["英文词数"] = output["清洗评论"].map(
        lambda text: len(re.findall(r"[A-Za-z]{2,}", text))
    )
    output["数字个数"] = output["清洗评论"].map(lambda text: len(re.findall(r"\d", text)))
    output["感叹号数"] = output["清洗评论"].map(lambda text: len(re.findall(r"[!！]", text)))
    output["问号数"] = output["清洗评论"].map(lambda text: len(re.findall(r"[?？]", text)))
    output["分词数量"] = output["分词列表"].map(len)

    # 长评论阈值采用数据自身的上四分位数，比固定阈值更适合当前样本分布。
    long_threshold = float(output["评论长度"].quantile(LONG_COMMENT_QUANTILE))
    output["是否长评论"] = output["评论长度"] >= long_threshold
    output["电影评论数"] = output.groupby(COL_MOVIE)[COL_MOVIE].transform("count")

    summary = {
        "feature_rows": int(len(output)),
        "long_comment_threshold": long_threshold,
        "positive_threshold": POSITIVE_SCORE_THRESHOLD,
        "negative_threshold": NEGATIVE_SCORE_THRESHOLD,
        "top_word_count": TOP_N_WORDS,
        "sentiment_counts": output["情感标签"].value_counts().to_dict(),
    }
    return output, summary


def label_sentiment(score: float) -> str:
    # 4-5 分视为正向，3 分为中性，1-2 分为负向。
    if score >= POSITIVE_SCORE_THRESHOLD:
        return "正向"
    if score <= NEGATIVE_SCORE_THRESHOLD:
        return "负向"
    return "中性"


def export_cleaned_data(df: pd.DataFrame, path: Path) -> None:
    # 只导出清洗后分析需要的字段，包括原始字段和新增特征。
    columns = [
        COL_MOVIE,
        COL_USER,
        COL_SCORE,
        "评分_5分制",
        "情感标签",
        COL_TIME,
        "年份",
        "月份",
        "年月",
        "日期",
        "小时",
        "星期",
        COL_TEXT,
        "清洗评论",
        "分词结果",
        "评论长度",
        "清洗后长度",
        "中文字符数",
        "英文词数",
        "数字个数",
        "感叹号数",
        "问号数",
        "分词数量",
        "是否长评论",
        "电影评论数",
    ]
    export_df = df[columns].copy()
    # utf-8-sig 可减少 Excel 打开中文 CSV 时的乱码风险。
    export_df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill="#172033"):
    draw.text(xy, text, font=font, fill=fill)


def wrap_label(label: str, max_chars: int = 8) -> str:
    if len(label) <= max_chars:
        return label
    return label[:max_chars] + "…"


def draw_grouped_bar_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    output_path: Path,
    y_label: str = "数量",
) -> None:
    width, height = 1400, 840
    margin_left, margin_right = 110, 70
    margin_top, margin_bottom = 150, 135
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "#F7F9FC")
    draw = ImageDraw.Draw(image)
    font_title = get_cjk_font(34)
    font_subtitle = get_cjk_font(18)
    font_label = get_cjk_font(16)
    font_small = get_cjk_font(14)

    draw_text(draw, (46, 36), title, font_title)
    draw_text(draw, (48, 84), subtitle, font_subtitle, "#53606F")

    max_value = max([max(values) if values else 0 for _, values, _ in series] + [1])
    y_max = max(1, math.ceil(max_value * 1.15))
    grid_count = 5
    for i in range(grid_count + 1):
        value = y_max * i / grid_count
        y = margin_top + chart_h - int(chart_h * value / y_max)
        draw.line((margin_left, y, width - margin_right, y), fill="#DDE5EE", width=1)
        draw_text(draw, (18, y - 10), f"{value:.0f}", font_small, "#53606F")

    draw.line(
        (margin_left, margin_top, margin_left, margin_top + chart_h), fill="#62758C", width=2
    )
    draw.line(
        (margin_left, margin_top + chart_h, width - margin_right, margin_top + chart_h),
        fill="#62758C",
        width=2,
    )
    draw_text(draw, (18, margin_top - 28), y_label, font_small, "#53606F")

    group_count = len(labels)
    group_w = chart_w / max(group_count, 1)
    bar_gap = 6
    series_count = len(series)
    bar_w = max(16, (group_w * 0.62 - bar_gap * (series_count - 1)) / max(series_count, 1))

    for idx, label in enumerate(labels):
        group_x = margin_left + idx * group_w
        total_bar_w = bar_w * series_count + bar_gap * (series_count - 1)
        start_x = group_x + (group_w - total_bar_w) / 2
        for s_idx, (_, values, color) in enumerate(series):
            value = values[idx] if idx < len(values) else 0
            bar_h = int(chart_h * value / y_max)
            x1 = int(start_x + s_idx * (bar_w + bar_gap))
            x2 = int(x1 + bar_w)
            y2 = margin_top + chart_h
            y1 = y2 - bar_h
            draw.rounded_rectangle((x1, y1, x2, y2), radius=4, fill=color)
            if value > 0:
                draw_text(draw, (x1 - 2, y1 - 22), f"{value:.0f}", font_small, "#334155")
        lx = int(group_x + group_w / 2 - 36)
        draw_text(draw, (lx, margin_top + chart_h + 16), wrap_label(label), font_label, "#172033")

    legend_x = margin_left
    legend_y = height - 62
    for name, _, color in series:
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 24, legend_y + 14), radius=3, fill=color)
        draw_text(draw, (legend_x + 32, legend_y - 4), name, font_label, "#334155")
        legend_x += 190

    image.save(output_path)


def quantiles(values: Iterable[float]) -> dict[str, float]:
    series = pd.Series(list(values), dtype="float64").dropna()
    return {
        "min": float(series.min()),
        "q1": float(series.quantile(0.25)),
        "median": float(series.quantile(0.5)),
        "q3": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def draw_boxplot(
    title: str,
    subtitle: str,
    datasets: list[tuple[str, list[float], str]],
    output_path: Path,
) -> None:
    width, height = 1200, 760
    margin_left, margin_right = 110, 80
    margin_top, margin_bottom = 150, 120
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "#F7F9FC")
    draw = ImageDraw.Draw(image)
    font_title = get_cjk_font(34)
    font_subtitle = get_cjk_font(18)
    font_label = get_cjk_font(16)
    font_small = get_cjk_font(14)
    draw_text(draw, (46, 36), title, font_title)
    draw_text(draw, (48, 84), subtitle, font_subtitle, "#53606F")

    stats = [(label, quantiles(values), color) for label, values, color in datasets]
    max_value = max(item[1]["max"] for item in stats)
    y_max = max(10, math.ceil(max_value * 1.08 / 50) * 50)

    for i in range(6):
        value = y_max * i / 5
        y = margin_top + chart_h - int(chart_h * value / y_max)
        draw.line((margin_left, y, width - margin_right, y), fill="#DDE5EE")
        draw_text(draw, (25, y - 10), f"{value:.0f}", font_small, "#53606F")

    draw.line((margin_left, margin_top, margin_left, margin_top + chart_h), fill="#62758C", width=2)
    draw.line((margin_left, margin_top + chart_h, width - margin_right, margin_top + chart_h), fill="#62758C", width=2)

    def y_for(value: float) -> int:
        return margin_top + chart_h - int(chart_h * value / y_max)

    gap = chart_w / (len(stats) + 1)
    box_w = 150
    for idx, (label, stat, color) in enumerate(stats, start=1):
        x = int(margin_left + gap * idx)
        y_min, y_q1, y_med, y_q3, y_max_line = (
            y_for(stat["min"]),
            y_for(stat["q1"]),
            y_for(stat["median"]),
            y_for(stat["q3"]),
            y_for(stat["max"]),
        )
        draw.line((x, y_max_line, x, y_q3), fill="#334155", width=3)
        draw.line((x, y_q1, x, y_min), fill="#334155", width=3)
        draw.line((x - 45, y_max_line, x + 45, y_max_line), fill="#334155", width=3)
        draw.line((x - 45, y_min, x + 45, y_min), fill="#334155", width=3)
        draw.rounded_rectangle(
            (x - box_w // 2, y_q3, x + box_w // 2, y_q1),
            radius=8,
            fill=color,
            outline="#334155",
            width=2,
        )
        draw.line((x - box_w // 2, y_med, x + box_w // 2, y_med), fill="#0F172A", width=4)
        draw_text(draw, (x - 70, margin_top + chart_h + 20), label, font_label)
        draw_text(draw, (x - 70, y_med - 28), f"中位数 {stat['median']:.0f}", font_small, "#334155")

    image.save(output_path)


def draw_word_cloud(word_counts: Counter, output_path: Path) -> None:
    random.seed(RANDOM_SEED)
    width, height = 1400, 880
    image = Image.new("RGB", (width, height), "#F7F9FC")
    draw = ImageDraw.Draw(image)
    title_font = get_cjk_font(34)
    subtitle_font = get_cjk_font(18)
    draw_text(draw, (46, 36), "清洗后评论词云", title_font)
    draw_text(draw, (48, 84), f"Top {TOP_N_WORDS} 高频词，已去除停用词和单字词", subtitle_font, "#53606F")

    words = word_counts.most_common(TOP_N_WORDS)
    if not words:
        image.save(output_path)
        return

    max_count = words[0][1]
    min_count = words[-1][1]
    palette = ["#1F4D78", "#2E74B5", "#6B8E23", "#B45309", "#9B1C1C", "#475569"]
    boxes: list[tuple[int, int, int, int]] = []
    area = (70, 140, width - 70, height - 70)

    def overlaps(box: tuple[int, int, int, int]) -> bool:
        padding = 8
        x1, y1, x2, y2 = box
        for bx1, by1, bx2, by2 in boxes:
            if not (x2 + padding < bx1 or bx2 + padding < x1 or y2 + padding < by1 or by2 + padding < y1):
                return True
        return False

    for word, count in words:
        if max_count == min_count:
            font_size = 28
        else:
            font_size = int(18 + (count - min_count) / (max_count - min_count) * 54)
        font = get_cjk_font(font_size)
        bbox = draw.textbbox((0, 0), word, font=font)
        word_w, word_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        placed = False
        for _ in range(500):
            x = random.randint(area[0], max(area[0], area[2] - word_w))
            y = random.randint(area[1], max(area[1], area[3] - word_h))
            box = (x, y, x + word_w, y + word_h)
            if not overlaps(box):
                draw.text((x, y), word, fill=random.choice(palette), font=font)
                boxes.append(box)
                placed = True
                break
        if not placed:
            continue

    image.save(output_path)


def create_visualizations(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 可视化前先复用评分解析逻辑，确保图表与清洗口径一致。
    raw_score_numeric = raw_df[COL_SCORE].map(parse_score)
    before_missing = [
        int(raw_df[COL_USER].isna().sum() + raw_df[COL_USER].astype(str).str.strip().eq("").sum()),
        int(raw_score_numeric.isna().sum()),
        int(pd.to_datetime(raw_df[COL_TIME], errors="coerce").isna().sum()),
        int(raw_df[COL_TEXT].isna().sum() + raw_df[COL_TEXT].astype(str).str.strip().eq("").sum()),
    ]
    after_missing = [
        int(cleaned_df[COL_USER].isna().sum()),
        int(cleaned_df["评分_5分制"].isna().sum()),
        int(cleaned_df["评论时间_dt"].isna().sum()),
        int(cleaned_df["清洗评论"].isna().sum() + cleaned_df["清洗评论"].eq("").sum()),
    ]
    missing_path = OUTPUT_DIR / "missing_before_after.png"
    # 绘制缺失/无效值处理前后对比图，直观展示数据质量改善。
    draw_grouped_bar_chart(
        "关键字段缺失/无效值对比",
        "原始数据中的“暂无评分”和空用户名在清洗后被删除或填充",
        ["用户名", "评分", "评论时间", "评论内容"],
        [("处理前", before_missing, "#94A3B8"), ("处理后", after_missing, "#2E74B5")],
        missing_path,
        "行数",
    )

    score_labels = ["1", "2", "3", "4", "5"]
    before_score_counts = [
        int((raw_score_numeric == float(score)).sum()) for score in range(1, 6)
    ]
    after_score_counts = [
        int((cleaned_df["评分_5分制"] == float(score)).sum()) for score in range(1, 6)
    ]
    score_path = OUTPUT_DIR / "rating_distribution_before_after.png"
    # 绘制缺失/无效值处理前后对比图，直观展示数据质量改善。
    draw_grouped_bar_chart(
        "评分分布对比",
        "评分从“50星”格式转换为 1-5 分制，清洗后移除“暂无”评分和重复/越界记录",
        score_labels,
        [("处理前", before_score_counts, "#94A3B8"), ("处理后", after_score_counts, "#2E74B5")],
        score_path,
        "评论数",
    )

    boxplot_path = OUTPUT_DIR / "comment_length_boxplot_before_after.png"
    # 箱线图用于比较清洗前后文本长度的中位数和离散程度。
    draw_boxplot(
        "评论长度箱线图对比",
        "处理前为原始评论长度，处理后为清洗文本长度；异常长文本被保留为有效信息",
        [
            ("处理前", raw_df[COL_TEXT].astype(str).str.len().tolist(), "#CBD5E1"),
            ("处理后", cleaned_df["清洗后长度"].tolist(), "#93C5FD"),
        ],
        boxplot_path,
    )

    month_counts = cleaned_df["年月"].value_counts().sort_index()
    month_path = OUTPUT_DIR / "monthly_comment_distribution.png"
    # 绘制缺失/无效值处理前后对比图，直观展示数据质量改善。
    draw_grouped_bar_chart(
        "清洗后评论月份分布",
        f"仅保留 {MIN_YEAR}-{MAX_YEAR} 年范围内记录，用于观察样本时间覆盖",
        list(month_counts.index),
        [("评论数", [int(v) for v in month_counts.values], "#2E74B5")],
        month_path,
        "评论数",
    )

    all_tokens: list[str] = []
    for tokens in cleaned_df["分词列表"]:
        all_tokens.extend(tokens)
    word_counts = Counter(all_tokens)
    wordcloud_path = OUTPUT_DIR / "wordcloud_cleaned_comments.png"
    # 基于清洗后分词结果生成词云，用于展示高频主题。
    draw_word_cloud(word_counts, wordcloud_path)

    return {
        "missing": missing_path,
        "score": score_path,
        "length_boxplot": boxplot_path,
        "monthly": month_path,
        "wordcloud": wordcloud_path,
    }



def build_summary(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, summary: dict, feature_summary: dict) -> dict:
    # 可视化前先复用评分解析逻辑，确保图表与清洗口径一致。
    raw_score_numeric = raw_df[COL_SCORE].map(parse_score)
    return {
        **summary,
        **feature_summary,
        "raw_score_counts": raw_df[COL_SCORE].value_counts(dropna=False).to_dict(),
        "cleaned_score_counts": cleaned_df["评分_5分制"].value_counts().sort_index().to_dict(),
        "raw_text_length": raw_df[COL_TEXT].astype(str).str.len().describe().to_dict(),
        "cleaned_text_length": cleaned_df["清洗后长度"].describe().to_dict(),
        "raw_numeric_score_missing": int(raw_score_numeric.isna().sum()),
        "cleaned_path": str(CLEANED_PATH),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw_df = load_raw_data(RAW_PATH)
    cleaned_structured, cleaning_summary = clean_structured_data(raw_df)
    featured_df, feature_summary = engineer_features(cleaned_structured)
    export_cleaned_data(featured_df, CLEANED_PATH)
    figures = create_visualizations(raw_df, featured_df)
    summary = build_summary(raw_df, featured_df, cleaning_summary, feature_summary)
    summary["figures"] = {name: str(path) for name, path in figures.items()}
    (OUTPUT_DIR / "processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"cleaned rows: {len(featured_df)}")
    print(f"wrote: {CLEANED_PATH}")
    for name, path in figures.items():
        print(f"figure {name}: {path}")


if __name__ == "__main__":
    main()
