![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Sports Analytics](https://img.shields.io/badge/Sports-Analytics-blue) ![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

# MLB Pitch Sequence Prediction — Multi-class Sequence Model

Predicts next pitch type from Statcast pitch sequence data using sequence-aware ML on Tarik Skubal career data.

---

## Task

**Sequence Classification**

---

## Architecture

```
Statcast Pitch Log → Sequence Feature Engineering → XGBoost Multi-class → Pitch Tendency Analysis
```

---

## Key Features

- Pitch type prediction (Fastball, Slider, Changeup, Curveball, Sinker)
- Sequence-aware feature engineering (prior pitch type, count, batter handedness)
- Count-state and game-situation features
- Pitcher tendencies by batter/count/inning encoded
- Confusion matrix analysis of most-mispredicted pitch transitions

---

## Dataset

[MLB Statcast](https://baseballsavant.mlb.com/) — Tarik Skubal career pitch-by-pitch data

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/baseball-pitch-sequence-predictor.git
cd baseball-pitch-sequence-predictor
pip install -r requirements.txt

# See data/README.md for dataset download
jupyter notebook notebooks/03_modeling.ipynb  # baseline XGBoost
# or run helpers:
python src/_extract_metrics.py
```

---

## Tech Stack

`scikit-learn · XGBoost · pandas · Statcast API`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Germany.

---

## License

MIT
