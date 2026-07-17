# -*- coding: utf-8 -*-
"""Streamlit dashboard for movie review sentiment and reputation analysis."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from best_sentiment_ensemble_v2 import (
    BEST_ENSEMBLE_ID,
    BestSentimentEnsemble,
    ensemble_artifact_fingerprint,
    load_best_validation_metrics,
)
from sentiment_utils import (
    BERT_DIR,
    LABEL_ORDER,
    ML_DIR,
    TARGET_COL,
    TEXT_COL,
    TOKEN_COL,
    VALIDATION_DIR,
    build_tfidf_text,
    load_best_model_dashboard_data,
    normalize_comment_for_prediction,
)


ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="电影评论情感分类与口碑分析",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp { background: #ffffff; color: #172033; }
    section[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1440px; }
    h1, h2, h3 { letter-spacing: 0; color: #172033; }
    h1 { font-size: 1.7rem; line-height: 1.25; margin-bottom: .2rem; }
    h2 { font-size: 1.18rem; margin-top: 1rem; }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetric"] label { color: #64748b; }
    .status-positive { color: #2f855a; font-weight: 650; }
    .status-neutral { color: #b7791f; font-weight: 650; }
    .status-negative { color: #c53030; font-weight: 650; }
    .small-note { color: #64748b; font-size: .9rem; }
    div.stButton > button[kind="primary"] {
        background: #0f6bca;
        border-color: #0f6bca;
        color: white;
        border-radius: 8px;
        font-weight: 650;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #0b5dae;
        border-color: #0b5dae;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_best_model_dashboard_data()


@st.cache_data
def get_reputation_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from reputation_analysis import (
        calculate_monthly_trend,
        calculate_movie_reputation,
        collect_examples,
        token_counter,
        top_words_frame,
    )

    df = get_data()
    return (
        calculate_movie_reputation(df),
        calculate_monthly_trend(df),
        top_words_frame(token_counter(df[TOKEN_COL])),
        collect_examples(df),
    )


@st.cache_data
def get_model_comparison() -> pd.DataFrame:
    path = ML_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame()


@st.cache_data
def get_bert_metrics() -> dict:
    path = BERT_DIR / "metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def get_best_ensemble_metrics() -> dict:
    try:
        return load_best_validation_metrics()
    except Exception:
        return {}


@st.cache_data
def get_error_summary() -> pd.DataFrame:
    path = BERT_DIR / "error_summary.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame()


@st.cache_data
def get_error_cases() -> pd.DataFrame:
    path = BERT_DIR / "error_analysis.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame()


@st.cache_data
def get_acceptance_report() -> dict:
    path = VALIDATION_DIR / "acceptance_report.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def model_summary_table(comparison: pd.DataFrame, bert_metrics: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if len(comparison):
        for _, row in comparison.iterrows():
            rows.append(
                {
                    "模型": row["model"],
                    "验证集 Macro F1": round(float(row["valid_macro_f1"]), 4),
                    "测试集 Accuracy": round(float(row["test_accuracy"]), 4),
                    "测试集 Macro F1": round(float(row["test_macro_f1"]), 4),
                    "来源": "传统机器学习",
                }
            )
    if bert_metrics:
        test_metrics = bert_metrics.get("test_metrics", {})
        rows.append(
            {
                "模型": "BERT / Chinese RoBERTa",
                "验证集 Macro F1": round(float(bert_metrics.get("best_valid_macro_f1", 0)), 4),
                "测试集 Accuracy": round(float(test_metrics.get("accuracy", 0)), 4),
                "测试集 Macro F1": round(float(test_metrics.get("macro_f1", 0)), 4),
                "来源": bert_metrics.get("base_model", "BERT"),
            }
        )
    return pd.DataFrame(rows).sort_values("测试集 Macro F1", ascending=False) if rows else pd.DataFrame()


@st.cache_resource
def load_traditional_model():
    path = ML_DIR / "best_traditional_model.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def resolve_bert_model_dir() -> Path | None:
    status_path = BERT_DIR / "status.json"
    candidates: list[Path] = []
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("model_dir"):
                candidates.append(Path(status["model_dir"]))
        except Exception:
            pass
    candidates.extend(sorted(BERT_DIR.glob("best_model_*"), reverse=True))
    candidates.append(BERT_DIR / "best_model")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@st.cache_resource
def load_bert_model():
    model_dir = resolve_bert_model_dir()
    if model_dir is None:
        return None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception:
        return None


@st.cache_resource
def load_best_sentiment_ensemble(
    artifact_fingerprint: tuple[tuple[str, int, int], ...],
):
    _ = artifact_fingerprint
    try:
        return BestSentimentEnsemble(), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    split_options = ["全部", "训练集", "验证集"]
    movies = ["全部"] + sorted(df["电影名称"].unique().tolist())
    years = ["全部"] + sorted(df["年份"].astype(str).unique().tolist())
    months = ["全部"] + sorted(df["月份"].astype(str).unique().tolist(), key=lambda x: int(x))
    sentiments = ["全部"] + LABEL_ORDER

    st.sidebar.title("筛选")
    model_split = st.sidebar.selectbox("模型数据分区", split_options)
    movie = st.sidebar.selectbox("电影", movies)
    year = st.sidebar.selectbox("年份", years)
    month = st.sidebar.selectbox("月份", months)
    sentiment = st.sidebar.selectbox("情感", sentiments)

    filtered = df.copy()
    split_value = {"训练集": "train", "验证集": "valid"}.get(model_split)
    if split_value is not None:
        filtered = filtered[filtered["model_split"] == split_value]
    if movie != "全部":
        filtered = filtered[filtered["电影名称"] == movie]
    if year != "全部":
        filtered = filtered[filtered["年份"].astype(str) == year]
    if month != "全部":
        filtered = filtered[filtered["月份"].astype(str) == month]
    if sentiment != "全部":
        filtered = filtered[filtered[TARGET_COL] == sentiment]
    return filtered


def sentiment_badge(label: str) -> str:
    css = {
        "正向": "status-positive",
        "中性": "status-neutral",
        "负向": "status-negative",
    }.get(label, "")
    return f'<span class="{css}">{label}</span>'


def horizontal_bar_chart(data: pd.DataFrame, x_field: str, y_field: str, height: int = 280) -> None:
    chart = (
        alt.Chart(data)
        .mark_bar(color="#0f6bca")
        .encode(
            x=alt.X(
                f"{x_field}:N",
                title=None,
                axis=alt.Axis(labelAngle=0, labelOverlap=False),
                sort=None,
            ),
            y=alt.Y(f"{y_field}:Q", title=None),
            tooltip=[alt.Tooltip(f"{x_field}:N"), alt.Tooltip(f"{y_field}:Q")],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def trend_line_chart(data: pd.DataFrame, x_field: str, y_field: str, title: str, height: int = 300) -> None:
    plot_data = data[[x_field, y_field]].copy()
    plot_data["指标"] = y_field
    chart = (
        alt.Chart(plot_data)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X(
                f"{x_field}:N",
                title=None,
                axis=alt.Axis(labelAngle=-35, labelOverlap=False),
                sort=None,
            ),
            y=alt.Y(f"{y_field}:Q", title=None),
            color=alt.Color(
                "指标:N",
                title=None,
                scale=alt.Scale(domain=[y_field], range=["#0f6bca"]),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[alt.Tooltip(f"{x_field}:N"), alt.Tooltip(f"{y_field}:Q")],
        )
        .properties(title=title, height=height)
    )
    st.altair_chart(chart, width="stretch")


def multi_metric_line_chart(data: pd.DataFrame, x_field: str, metrics: list[str], title: str, height: int = 300) -> None:
    rating_metric = "平均评分"
    rate_metrics = [metric for metric in metrics if metric != rating_metric]
    metric_domain = [rating_metric, "正向率", "负向率"]
    metric_colors = ["#0f6bca", "#2f855a", "#c53030"]
    x_axis = alt.X(
        f"{x_field}:N",
        title=None,
        axis=alt.Axis(labelAngle=-35, labelOverlap=False),
        sort=None,
    )

    rating_data = data[[x_field, rating_metric]].copy()
    rating_data["指标"] = rating_metric
    rating_chart = (
        alt.Chart(rating_data)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=x_axis,
            y=alt.Y(
                f"{rating_metric}:Q",
                title="平均评分",
                scale=alt.Scale(domain=[1, 5]),
                axis=alt.Axis(titleColor="#0f6bca"),
            ),
            color=alt.Color(
                "指标:N",
                title=None,
                scale=alt.Scale(domain=metric_domain, range=metric_colors),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip(f"{x_field}:N"),
                alt.Tooltip("指标:N"),
                alt.Tooltip(f"{rating_metric}:Q", format=".3f"),
            ],
        )
    )

    rate_data = data[[x_field, *rate_metrics]].melt(x_field, var_name="指标", value_name="比例")
    rate_chart = (
        alt.Chart(rate_data)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=x_axis,
            y=alt.Y(
                "比例:Q",
                title="情感比例",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", titleColor="#2d3748", orient="right"),
            ),
            color=alt.Color(
                "指标:N",
                title=None,
                scale=alt.Scale(domain=metric_domain, range=metric_colors),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[alt.Tooltip(f"{x_field}:N"), alt.Tooltip("指标:N"), alt.Tooltip("比例:Q", format=".1%")],
        )
    )

    chart = alt.layer(rating_chart, rate_chart).resolve_scale(y="independent").properties(title=title, height=height)
    st.altair_chart(chart, width="stretch")


def predict_comment(
    text: str,
) -> tuple[str, str, float | None, dict[str, float] | None, str]:
    try:
        fingerprint = ensemble_artifact_fingerprint()
        ensemble, ensemble_error = load_best_sentiment_ensemble(fingerprint)
    except Exception as exc:
        ensemble, ensemble_error = None, f"{type(exc).__name__}: {exc}"
    if ensemble is not None:
        result = ensemble.predict(text)
        return (
            str(result["label"]),
            "最佳五模型加权集成",
            float(result["confidence"]),
            dict(result["probabilities"]),
            "",
        )

    bert = load_bert_model()
    if bert is not None:
        tokenizer, model, device = bert
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = F.softmax(logits, dim=-1).detach().cpu().squeeze(0)
            label_id = int(probabilities.argmax().item())
            confidence = float(probabilities[label_id].item())
        label = model.config.id2label.get(label_id, LABEL_ORDER[label_id])
        probability_map = {
            LABEL_ORDER[index]: float(probabilities[index].item())
            for index in range(min(len(LABEL_ORDER), len(probabilities)))
        }
        return (
            label,
            "BERT（集成加载失败后的回退）",
            confidence,
            probability_map,
            ensemble_error,
        )

    traditional = load_traditional_model()
    if traditional is not None:
        model_text = normalize_comment_for_prediction(text)
        label = str(traditional.predict([model_text])[0])
        confidence = None
        if hasattr(traditional, "predict_proba"):
            probabilities = traditional.predict_proba([model_text])[0]
            confidence = float(max(probabilities))
        return label, "传统机器学习模型（回退）", confidence, None, ensemble_error

    return "模型未训练", "无可用模型", None, None, ensemble_error


def render_overview(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    sentiment_counts = filtered[TARGET_COL].value_counts().reindex(LABEL_ORDER, fill_value=0)
    avg_score = filtered["评分_5分制"].mean() if len(filtered) else 0
    positive_rate = sentiment_counts["正向"] / max(int(sentiment_counts.sum()), 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("评论数", f"{len(filtered):,}")
    col2.metric("电影数", f"{filtered['电影名称'].nunique():,}")
    col3.metric("平均评分", f"{avg_score:.2f}")
    col4.metric("正向率", f"{positive_rate:.1%}")

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("情感分布")
        sentiment_chart = sentiment_counts.rename_axis("情感").reset_index(name="评论数")
        horizontal_bar_chart(sentiment_chart, "情感", "评论数", height=280)
    with right:
        st.subheader("评分分布")
        score_counts = filtered["评分_5分制"].value_counts().sort_index()
        score_chart = score_counts.rename_axis("评分").reset_index(name="评论数")
        score_chart["评分"] = score_chart["评分"].map(lambda value: f"{value:g}")
        horizontal_bar_chart(score_chart, "评分", "评论数", height=280)


def render_reputation(filtered: pd.DataFrame, movie_metrics: pd.DataFrame, trend: pd.DataFrame) -> None:
    st.subheader("电影口碑榜")
    visible_movies = filtered["电影名称"].unique().tolist()
    table = movie_metrics[movie_metrics["电影名称"].isin(visible_movies)].copy()
    if table.empty:
        st.info("当前筛选条件下没有可展示的电影口碑指标。")
        return

    weight_col, note_col = st.columns([0.42, 0.58])
    with weight_col:
        rating_weight = st.slider("评分权重", 0.0, 1.0, 0.6, 0.05)
    positive_weight = round(1.0 - rating_weight, 2)
    with note_col:
        st.caption(f"口碑指数 = {rating_weight:.0%} * 评分归一化 + {positive_weight:.0%} * 正向率；默认权重为 60% / 40%。")

    table["口碑指数(可调)"] = (
        ((table["平均评分"].astype(float) - 1.0) / 4.0 * rating_weight + table["正向率"].astype(float) * positive_weight)
        * 100
    ).round(2)
    table = table.sort_values(["口碑指数(可调)", "评论数"], ascending=[False, False])
    st.dataframe(
        table[
            [
                "电影名称",
                "评论数",
                "平均评分",
                "正向率",
                "中性率",
                "负向率",
                "长评论比例",
                "口碑指数",
                "口碑指数(可调)",
            ]
        ],
        hide_index=True,
        width="stretch",
    )

    st.subheader("评论热度与口碑趋势")
    visible_months = sorted(filtered["年月"].dropna().astype(str).unique().tolist())
    visible_trend = trend[trend["年月"].astype(str).isin(visible_months)].copy()
    if len(visible_trend):
        heat_col, reputation_col = st.columns(2)
        with heat_col:
            trend_line_chart(visible_trend, "年月", "评论数", "评论热度趋势", height=300)
        with reputation_col:
            multi_metric_line_chart(
                visible_trend,
                "年月",
                ["平均评分", "正向率", "负向率"],
                "评分与情感比例趋势",
                height=300,
            )
    else:
        st.info("当前筛选条件下没有可展示的月度趋势。")


def render_keywords(words: pd.DataFrame, examples: pd.DataFrame, filtered: pd.DataFrame) -> None:
    st.subheader("关键词")
    col1, col2 = st.columns([0.95, 1.05])
    with col1:
        st.dataframe(words.head(30), hide_index=True, width="stretch")
    with col2:
        tokens = build_tfidf_text(filtered).str.split().explode().dropna()
        filtered_words = tokens.value_counts().head(20).rename_axis("词语").reset_index(name="频次")
        st.bar_chart(filtered_words.set_index("词语")["频次"], height=360)

    st.subheader("典型评论")
    examples_display = examples.head(18).rename(columns={"评分_5分制": "评分5分制"})
    st.dataframe(examples_display, hide_index=True, width="stretch")


def render_model_area() -> None:
    st.subheader("模型评估")
    best_metrics = get_best_ensemble_metrics()
    if best_metrics:
        metric_columns = st.columns(4)
        metric_columns[0].metric("最佳验证 Macro F1", f"{best_metrics.get('macro_f1', 0):.4f}")
        metric_columns[1].metric("验证 Accuracy", f"{best_metrics.get('accuracy', 0):.4f}")
        metric_columns[2].metric(
            "验证 Macro Precision", f"{best_metrics.get('macro_precision', 0):.4f}"
        )
        metric_columns[3].metric(
            "验证 Macro Recall", f"{best_metrics.get('macro_recall', 0):.4f}"
        )
        st.caption(
            f"当前在线推理模型：{BEST_ENSEMBLE_ID}。该结果来自固定验证集；封存测试集尚未解封。"
        )

    comparison = get_model_comparison()
    bert_metrics = get_bert_metrics()
    summary = model_summary_table(comparison, bert_metrics)
    if len(summary):
        st.dataframe(summary, hide_index=True, width="stretch")
    else:
        st.caption("传统模型对比结果未加载；当前页面以最佳集成验证指标为准。")

    error_summary = get_error_summary()
    error_cases = get_error_cases()
    if len(error_summary):
        left, right = st.columns([0.85, 1.15])
        with left:
            st.markdown("**BERT 误判类型**")
            st.dataframe(error_summary, hide_index=True, width="stretch")
        with right:
            st.markdown("**误判样例**")
            sample_cols = [col for col in [TARGET_COL, "预测标签", "电影名称", TEXT_COL] if col in error_cases.columns]
            st.dataframe(error_cases[sample_cols].head(8), hide_index=True, width="stretch")
    else:
        st.caption("当前精简部署未保留旧版 BERT 误判分析结果。")

    acceptance = get_acceptance_report()
    if acceptance:
        status_text = "通过" if acceptance.get("passed") else "需检查"
        st.caption(f"项目验收状态：{status_text}；报告文件：outputs/sentiment/validation/acceptance_report.json")

    with st.expander("查看原始评估详情"):
        if len(comparison):
            st.dataframe(comparison, hide_index=True, width="stretch")
        if bert_metrics:
            st.json(bert_metrics)

    st.subheader("单条评论预测")
    sample = st.text_area("评论文本", value="剧情节奏很好，人物情感也很打动我。", height=110)
    if st.button("预测情感", type="primary"):
        if not sample.strip():
            st.warning("请输入评论文本后再预测。")
            return
        label, source, confidence, probabilities, ensemble_error = predict_comment(sample)
        if ensemble_error:
            st.warning(f"最佳集成加载失败，已降级使用备用模型：{ensemble_error}")
        if confidence is None:
            confidence_text = ""
        elif source == "最佳五模型加权集成":
            confidence_text = f"；集成置信得分（参考）：{confidence:.1%}"
        else:
            confidence_text = f"；置信度：{confidence:.1%}"
        st.markdown(
            f"预测结果：{sentiment_badge(label)}　<span class='small-note'>模型：{source}{confidence_text}</span>",
            unsafe_allow_html=True,
        )
        if probabilities:
            probability_frame = pd.DataFrame(
                {
                    "情感": LABEL_ORDER,
                    "概率": [probabilities.get(label_name, 0.0) for label_name in LABEL_ORDER],
                }
            )
            probability_chart = (
                alt.Chart(probability_frame)
                .mark_bar()
                .encode(
                    x=alt.X("情感:N", title=None, sort=LABEL_ORDER),
                    y=alt.Y("概率:Q", title="预测概率", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color(
                        "情感:N",
                        title=None,
                        scale=alt.Scale(
                            domain=LABEL_ORDER,
                            range=["#c53030", "#b7791f", "#2f855a"],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("情感:N"),
                        alt.Tooltip("概率:Q", format=".1%"),
                    ],
                )
                .properties(height=220)
            )
            st.altair_chart(probability_chart, width="stretch")


def main() -> None:
    df = get_data()
    movie_metrics, trend, words, examples = get_reputation_tables()
    filtered = filter_data(df)

    st.title("电影评论情感分类与口碑可视化分析")
    st.caption(
        "数据来源：当前最佳集成模型的固定训练集与验证集；"
        "封存测试集未读取。目标：正向 / 中性 / 负向三分类与电影口碑分析。"
    )

    render_overview(df, filtered)
    st.divider()
    render_reputation(filtered, movie_metrics, trend)
    st.divider()
    render_keywords(words, examples, filtered)
    st.divider()
    render_model_area()


if __name__ == "__main__":
    main()
