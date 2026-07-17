# BERT 电影评论情感预测

中文 | [English](README.en.md)

本项目使用经过校准的五模型 BERT 集成，对中文电影评论进行负向、中性、正向三分类，并提供电影口碑可视化分析界面。

## 当前模型

在线部署模型为 `ensemble_diverse_weighted_calibrated_v2`，它是项目保留的可比三分类模型中验证集 Macro F1 最高的模型，可预测：

- 负向
- 中性
- 正向

固定验证集指标：

| 指标 | 分数 |
|---|---:|
| Macro F1 | 0.7669 |
| Accuracy | 0.8037 |
| Macro Precision | 0.7818 |
| Macro Recall | 0.7565 |

仓库包含为最终评估保留的封存测试集，但可视化界面不会读取该测试集。

## 可视化界面

Streamlit 界面提供：

- 情感分布和评分分布；
- 电影口碑排行和月度趋势；
- 关键词及典型评论浏览；
- 当前最佳集成模型的固定验证指标；
- 单条评论交互式预测和三分类概率展示。

所有图表使用与当前模型对应的固定训练集和验证集，共包含 1,816 条评论、37 部电影。星级评分字段不会作为模型输入，以避免目标泄漏。

## 仓库结构

```text
assets/data/douban_comments.csv              原始标准评论数据
outputs/sentiment/bert-v2/splits/            固定训练集、验证集和封存测试集
outputs/sentiment/bert-v2/experiments/       保留的 8 个部署模型制品目录
best_sentiment_ensemble_v2.py                校准集成推理
feature_engineering.py                       数据清洗和特征重建
prepare_sentiment_v2.py                      确定性拆分与数据辅助函数
reputation_analysis.py                       口碑指标与关键词计算
sentiment_utils.py                           数据和评估公共函数
streamlit_app.py                             Streamlit 界面入口
```

## 安装

模型权重使用 Git LFS 存储。克隆或拉取仓库前，请先安装并启用 Git LFS。

```powershell
git lfs install
git clone git@github.com:sguistar/Bert-Comment-Emotion-Prediction.git
cd Bert-Comment-Emotion-Prediction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果需要支持 CUDA 的 PyTorch，请先根据本机 CUDA 环境安装对应的 PyTorch 版本，再安装其余依赖。

## 启动

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501
```

浏览器访问 <http://localhost:8501/>。当可用 GPU 显存不少于 3 GiB 时使用 CUDA 推理，否则自动回退到 CPU。

## 数据与评估说明

- 标准数据拆分是确定性的，并使用稳定行 ID 标识。
- 可视化数据重建只读取 `train.csv` 和 `valid.csv`。
- `test.csv` 保持封存，仅用于最终一次性评估。
- 模型和数据血缘记录在 `outputs/sentiment/bert-v2/manifest.json` 及各保留实验的 `metrics.json` 中。
- 验证集结果不能表述为封存测试集性能。
