# 🚚 Amazon Supply Chain Intelligence
## Delivery Delay Risk Predictor — ML Pipeline

---

## 📌 Project Overview

This project is built for **Amazon Supply Chain Intelligence**, an AI-driven
logistics division aiming to predict delivery delay risk for every e-commerce
order before it ships.

Using historical logistics data — including GPS coordinates, fuel consumption,
ETA variation, traffic congestion, weather severity, driver behaviour scores,
delay probability, and 18 other numeric signals — the system trains a
**multi-class classifier** to categorise each delivery into one of three
risk tiers:

| Class | Meaning |
|-------|---------|
| ✅ **On-Time** | Delivery expected within the promised window |
| ⚠️ **At Risk** | Delivery may be marginally delayed (moderate risk) |
| ❌ **Delayed** | High probability of significant customer-impacting delay |

---

## 🤖 Models Trained

| # | Model | Role |
|---|-------|------|
| 1 | **Logistic Regression** | Linear baseline — fast, interpretable, used for coefficient analysis |
| 2 | **Random Forest** | Primary model — 500-tree bagging ensemble, tuned via RandomizedSearchCV |

Both models are evaluated head-to-head on the same held-out test set.
The best performing model is automatically saved as `best_model.pkl`.

---

## 🏗️ Pipeline Architecture

```
Raw CSV Dataset
      │
      ▼
┌─────────────────────────────────────────┐
│  Step 1 — EDA & Preprocessing           │
│  • Type detection & missing imputation  │
│  • 3×IQR outlier clipping               │
│  • Standard Scaling (24 numeric cols)   │
│  • Label Encoding on target column      │
│  • Saves: preprocessed_data.csv         │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Step 2 — Feature Engineering           │
│  • Interaction features (risk signals)  │
│  • Polynomial features (x², √x)         │
│  • Composite Risk Score                 │
│  • Log transforms on skewed columns     │
│  • Saves: engineered_data.csv           │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Step 3 — Model Training                │
│  • SMOTE balances class imbalance       │
│  • Logistic Regression (CV baseline)   │
│  • Random Forest (RandomizedSearchCV)  │
│  • Best model auto-selected & saved     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Step 4 — Evaluation & Report           │
│  • Accuracy, F1, ROC-AUC both models   │
│  • Side-by-side confusion matrices     │
│  • ROC & Precision-Recall curves       │
│  • RF importances + LR coefficients    │
│  • Full text report saved               │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
amazon_supply_chain_ml/
│
├── data/
│   └── dynamic_supply_chain_logistics_dataset.csv   ← YOUR DATA HERE
│
├── outputs/                                          ← auto-created on run
│   ├── models/          ← .pkl artefacts
│   ├── plots/           ← all 10 charts (PNG)
│   └── reports/         ← text report + model comparison CSV
│
├── config.py                    ← central config — edit column names here
├── step1_eda_preprocessing.py
├── step2_feature_engineering.py
├── step3_train_models.py        ← Logistic Regression + Random Forest
├── step4_evaluate_report.py     ← evaluation of both models
├── main_pipeline.py             ← single entry point
├── requirements.txt
├── README.md
└── STEP_BY_STEP_GUIDE.md
```

---

## ⚙️ Key Techniques

| Technique | Purpose |
|-----------|---------|
| **SMOTE** | Balances the 23k vs 3k vs 5k class imbalance before training |
| **RandomizedSearchCV** | Tunes Random Forest hyperparameters (20 combos × 5 folds) |
| **StratifiedKFold** | Preserves class ratios in every CV fold |
| **Feature Engineering** | 15+ derived features (interactions, polynomials, risk scores) |
| **3×IQR Clipping** | Robust outlier handling for sensor/GPS data |
| **StandardScaler** | Normalises all 24 numeric features |

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `plots/01_target_distribution.png` | Class balance bar + pie |
| `plots/02_numeric_distributions.png` | Histograms & boxplots |
| `plots/03_correlation_heatmap.png` | Feature correlation matrix |
| `plots/04_feature_correlation_ranking.png` | Top features by target correlation |
| `plots/05_model_comparison.png` | LR vs RF — all metrics side-by-side |
| `plots/06_confusion_matrix.png` | Confusion matrices for both models |
| `plots/07_roc_curves.png` | ROC curves — both models on same chart |
| `plots/08_precision_recall_curves.png` | PR curves — both models |
| `plots/09_feature_importances.png` | RF importances + LR coefficients |
| `plots/10_confidence_distribution.png` | Prediction confidence histograms |
| `reports/final_evaluation_report.txt` | Complete metrics text report |
| `reports/model_comparison.csv` | Head-to-head accuracy table |
| `models/best_model.pkl` | Best model ready for inference |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies (Python 3.9+)
pip install -r requirements.txt

# 2. Place your dataset
cp /path/to/dataset.csv data/dynamic_supply_chain_logistics_dataset.csv

# 3. Run the full pipeline
python main_pipeline.py
```

---

## ⚙️ Configuration

Open `config.py` to adjust:

```python
TARGET_COL   = "risk_classification"   # your target column name
NUMERIC_COLS = [...]                   # your 24 feature column names
TEST_SIZE    = 0.20                    # train/test split
CV_FOLDS     = 5                       # cross-validation folds
```

---

## 👤 Project Context

**Division**: Amazon Supply Chain Intelligence  
**Goal**: Proactive delivery risk detection to improve customer satisfaction  
**Built with**: Python · Scikit-learn · imbalanced-learn · Matplotlib · Seaborn
