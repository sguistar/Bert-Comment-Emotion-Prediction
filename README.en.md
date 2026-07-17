# BERT Comment Emotion Prediction

[中文](README.md) | English

Chinese movie-comment sentiment classification and reputation visualization using a calibrated five-model BERT ensemble.

## Current model

The deployed model is `ensemble_diverse_weighted_calibrated_v2`, the best comparable three-class model retained by this project. It predicts:

- 负向 (negative)
- 中性 (neutral)
- 正向 (positive)

Validation metrics on the fixed validation split:

| Metric | Score |
|---|---:|
| Macro F1 | 0.7669 |
| Accuracy | 0.8037 |
| Macro Precision | 0.7818 |
| Macro Recall | 0.7565 |

The sealed test split is included for future final evaluation but is not read by the dashboard.

## Dashboard

The Streamlit interface provides:

- sentiment and rating distributions;
- movie reputation ranking and monthly trends;
- keyword and representative-comment exploration;
- fixed validation metrics for the deployed ensemble;
- interactive single-comment prediction with three-class probabilities.

All dashboard charts use the same fixed train + validation rows associated with the deployed model: 1,816 comments across 37 movies. Star-rating columns are not used as model inputs.

## Repository layout

```text
assets/data/douban_comments.csv              Raw canonical comment data
outputs/sentiment/bert-v2/splits/            Fixed train/valid/sealed-test splits
outputs/sentiment/bert-v2/experiments/       Eight retained deployment artifacts
best_sentiment_ensemble_v2.py                Calibrated ensemble inference
feature_engineering.py                       Cleaning and feature reconstruction
prepare_sentiment_v2.py                      Deterministic split/data helpers
reputation_analysis.py                       Reputation and keyword calculations
sentiment_utils.py                           Shared data and evaluation utilities
streamlit_app.py                             Dashboard entry point
```

## Setup

Model weights are stored with Git LFS. Install Git LFS before cloning or pulling the repository.

```powershell
git lfs install
git clone git@github.com:sguistar/Bert-Comment-Emotion-Prediction.git
cd Bert-Comment-Emotion-Prediction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a CUDA-enabled PyTorch build, follow the PyTorch installation instructions appropriate for the local CUDA runtime before installing the remaining requirements.

## Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501
```

Open <http://localhost:8501/>. Inference uses CUDA when at least 3 GiB of GPU memory is available; otherwise it falls back to CPU.

## Data and evaluation notes

- Canonical splits are deterministic and identified by stable row IDs.
- Dashboard reconstruction reads only `train.csv` and `valid.csv`.
- `test.csv` remains sealed and should only be opened for the final one-time evaluation.
- Model and data lineage are recorded in `outputs/sentiment/bert-v2/manifest.json` and the retained experiment metrics.
- Validation results should not be presented as sealed-test performance.
