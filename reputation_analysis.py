# -*- coding: utf-8 -*-
"""Build movie reputation metrics, trend tables, keywords, and static charts."""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sentiment_utils import (
    LABEL_ORDER,
    REPUTATION_DIR,
    TARGET_COL,
    TOKEN_COL,
    configure_chinese_font,
    ensure_output_dirs,
    load_cleaned_data,
    safe_rate,
    write_json,
)


def token_counter(series: pd.Series) -> Counter:
    counter: Counter[str] = Counter()
    for value in series.fillna("").astype(str):
        counter.update(token for token in value.split() if token.strip())
    return counter


def top_words_frame(counter: Counter, limit: int = 80) -> pd.DataFrame:
    return pd.DataFrame(counter.most_common(limit), columns=["词语", "频次"])


def calculate_reputation_index(avg_score: float, positive_rate: float, rating_weight: float = 0.6, positive_weight: float = 0.4) -> float:
    rating_norm = (avg_score - 1.0) / 4.0
    return (rating_weight * rating_norm + positive_weight * positive_rate) * 100


def calculate_movie_reputation(
    df: pd.DataFrame,
    rating_weight: float = 0.6,
    positive_weight: float = 0.4,
) -> pd.DataFrame:
    rows = []
    for movie, group in df.groupby("电影名称"):
        total = len(group)
        counts = group[TARGET_COL].value_counts()
        avg_score = float(group["评分_5分制"].mean())
        positive_rate = safe_rate(float(counts.get("正向", 0)), total)
        neutral_rate = safe_rate(float(counts.get("中性", 0)), total)
        negative_rate = safe_rate(float(counts.get("负向", 0)), total)
        reputation_index = calculate_reputation_index(avg_score, positive_rate, rating_weight, positive_weight)
        rows.append(
            {
                "电影名称": movie,
                "评论数": total,
                "平均评分": round(avg_score, 3),
                "正向数": int(counts.get("正向", 0)),
                "中性数": int(counts.get("中性", 0)),
                "负向数": int(counts.get("负向", 0)),
                "正向率": round(positive_rate, 4),
                "中性率": round(neutral_rate, 4),
                "负向率": round(negative_rate, 4),
                "平均评论长度": round(float(group["评论长度"].mean()), 2),
                "长评论比例": round(float(group["是否长评论"].mean()), 4),
                "口碑指数": round(reputation_index, 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["口碑指数", "评论数"], ascending=[False, False])


def calculate_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, group in df.groupby("年月"):
        total = len(group)
        counts = group[TARGET_COL].value_counts()
        rows.append(
            {
                "年月": month,
                "评论数": total,
                "平均评分": round(float(group["评分_5分制"].mean()), 3),
                "正向率": round(safe_rate(float(counts.get("正向", 0)), total), 4),
                "中性率": round(safe_rate(float(counts.get("中性", 0)), total), 4),
                "负向率": round(safe_rate(float(counts.get("负向", 0)), total), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("年月")


def calculate_movie_keywords(df: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    rows = []
    for movie, group in df.groupby("电影名称"):
        words = token_counter(group[TOKEN_COL]).most_common(limit)
        rows.append(
            {
                "电影名称": movie,
                "关键词": "、".join(word for word, _ in words),
                "关键词频次": "、".join(f"{word}:{count}" for word, count in words),
            }
        )
    return pd.DataFrame(rows).sort_values("电影名称")


def collect_examples(df: pd.DataFrame) -> pd.DataFrame:
    examples = []
    for label in LABEL_ORDER:
        group = df[df[TARGET_COL] == label].copy()
        group = group.sort_values(["评论长度"], ascending=False).head(8)
        for _, row in group.iterrows():
            examples.append(
                {
                    "情感标签": label,
                    "电影名称": row["电影名称"],
                    "评分_5分制": row["评分_5分制"],
                    "评论长度": row["评论长度"],
                    "评论内容": row["评论内容"],
                }
            )
    return pd.DataFrame(examples)


def save_reputation_charts(movie_metrics: pd.DataFrame, trend: pd.DataFrame) -> None:
    configure_chinese_font()

    top_movies = movie_metrics.head(12).sort_values("口碑指数")
    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=150)
    ax.barh(top_movies["电影名称"], top_movies["口碑指数"], color="#2E74B5")
    ax.set_title("电影口碑指数 Top 12", fontsize=14, pad=12)
    ax.set_xlabel("口碑指数")
    ax.grid(axis="x", color="#DDE5EE", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(REPUTATION_DIR / "top_reputation_movies.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 5.8), dpi=150)
    ax1.plot(trend["年月"], trend["评论数"], marker="o", color="#2E74B5", label="评论数")
    ax1.set_ylabel("评论数")
    ax1.tick_params(axis="x", rotation=35)
    ax1.grid(axis="y", color="#DDE5EE", linewidth=0.8)
    ax2 = ax1.twinx()
    ax2.plot(trend["年月"], trend["正向率"], marker="s", color="#2F855A", label="正向率")
    ax2.plot(trend["年月"], trend["负向率"], marker="^", color="#C53030", label="负向率")
    ax2.set_ylabel("情感比例")
    ax1.set_title("评论热度与情感比例趋势", fontsize=14, pad=12)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPUTATION_DIR / "monthly_reputation_trend.png", bbox_inches="tight")
    plt.close(fig)


def build_reputation_outputs() -> None:
    ensure_output_dirs()
    df = load_cleaned_data()
    movie_metrics = calculate_movie_reputation(df)
    trend = calculate_monthly_trend(df)
    movie_keywords = calculate_movie_keywords(df)
    examples = collect_examples(df)
    global_words = top_words_frame(token_counter(df[TOKEN_COL]))

    movie_metrics.to_csv(REPUTATION_DIR / "movie_reputation_metrics.csv", index=False, encoding="utf-8-sig")
    trend.to_csv(REPUTATION_DIR / "monthly_trend.csv", index=False, encoding="utf-8-sig")
    movie_keywords.to_csv(REPUTATION_DIR / "movie_keywords.csv", index=False, encoding="utf-8-sig")
    examples.to_csv(REPUTATION_DIR / "sentiment_examples.csv", index=False, encoding="utf-8-sig")
    global_words.to_csv(REPUTATION_DIR / "global_top_words.csv", index=False, encoding="utf-8-sig")

    for label in LABEL_ORDER:
        label_words = top_words_frame(token_counter(df.loc[df[TARGET_COL] == label, TOKEN_COL]), 60)
        label_words.to_csv(REPUTATION_DIR / f"top_words_{label}.csv", index=False, encoding="utf-8-sig")

    save_reputation_charts(movie_metrics, trend)
    if len(movie_metrics):
        top_movie = movie_metrics.iloc[0]["电影名称"]
        top_reputation_index = float(movie_metrics.iloc[0]["口碑指数"])
    else:
        top_movie = "N/A"
        top_reputation_index = 0.0

    write_json(
        REPUTATION_DIR / "summary.json",
        {
            "movie_count": int(df["电影名称"].nunique()),
            "comment_count": int(len(df)),
            "top_movie": top_movie,
            "top_reputation_index": top_reputation_index,
            "outputs": [
                "movie_reputation_metrics.csv",
                "monthly_trend.csv",
                "movie_keywords.csv",
                "sentiment_examples.csv",
                "global_top_words.csv",
            ],
        },
    )
    print("reputation analysis completed")
    print(movie_metrics.head(10).to_string(index=False))


if __name__ == "__main__":
    build_reputation_outputs()
