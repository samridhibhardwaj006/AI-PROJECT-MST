# 🗺️ Step-by-Step Guide
## Amazon Supply Chain Intelligence — ML Pipeline
### Models: Logistic Regression | Random Forest

---

## ✅ Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9 or higher |
| pip | latest |
| RAM | ≥ 8 GB recommended |

---

## 📦 Step 0 — Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**

| Library | Purpose |
|---------|---------|
| `pandas` / `numpy` | Data loading and manipulation |
| `scikit-learn` | Logistic Regression, Random Forest, metrics, CV |
| `imbalanced-learn` | SMOTE over-sampling |
| `matplotlib` / `seaborn` | All 10 visualisation charts |
| `joblib` | Saving and loading model `.pkl` files |
| `scipy` | Statistical utilities |

> Note: `xgboost` and `lightgbm` are **NOT** required —
> this project uses only Logistic Regression and Random Forest.

---

## 📂 Step 0b — Place Your Dataset

```bash
mkdir -p data
cp /path/to/dynamic_supply_chain_logistics_dataset.csv data/
```

> If your column names differ, open `config.py` and update
> `TARGET_COL` and `NUMERIC_COLS` before running anything.

---

## 🔍 Step 1 — EDA & Preprocessing

```bash
python step1_eda_preprocessing.py
```

### What happens:

**1. Load & inspect**
Reads the CSV, prints shape, dtypes, missing value counts, duplicate rows.
Drops `timestamp` (detected as ID-like — unique per row, no predictive value).

**2. Auto-detect column types**
Your dataset has 24 numeric feature columns and 1 string target
(`risk_classification`). No categorical features to encode.

**3. Target distribution chart**
Bar + pie chart showing the split between On-Time / At Risk / Delayed.
Reveals the severe imbalance (23k vs 3k vs 5k) that SMOTE will fix.
Saved → `plots/01_target_distribution.png`

**4. Numeric distributions**
Histograms and boxplots for every feature — spot skewed columns
(like `shipping_costs`, `historical_demand`) and outliers in GPS data.
Saved → `plots/02_numeric_distributions.png`

**5. Correlation heatmap**
Shows which of the 24 features correlate with each other.
High correlation pairs (e.g. `delay_probability` & `disruption_likelihood_score`)
may be redundant but are kept since tree models handle this fine.
Saved → `plots/03_correlation_heatmap.png`

**6. Impute missing values**
Numeric → filled with **median** (robust to outlier influence).
No categorical columns to impute in your dataset.

**7. Clip outliers (3×IQR)**
GPS latitude/longitude and sensor readings can have extreme values.
3×IQR fence clips only genuine statistical extremes, not legitimate
large values like long shipment routes.

**8. Encode target label**
`risk_classification` string values → integers (0, 1, 2) via LabelEncoder.
Mapping saved to `label_encoder.pkl` for decoding predictions later.

**9. Standard scale all numeric features**
Each of the 24 columns → zero mean, unit variance.
Critical for Logistic Regression convergence.
Harmless for Random Forest but keeps the pipeline consistent.

**Output files created:**
```
outputs/preprocessed_data.csv
outputs/models/scaler.pkl
outputs/models/label_encoder.pkl
outputs/models/num_imputer.pkl
outputs/models/feature_cols.pkl
```

---

## 🔧 Step 2 — Feature Engineering

```bash
python step2_feature_engineering.py
```

### What happens:

**1. Interaction features**
Multiplies or divides pairs of columns that together carry more signal
than either alone. Examples using your real column names:

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `feat_delay_x_eta` | delay_probability × eta_variation_hours | Combined delay + timing risk |
| `feat_disruption_x_route` | disruption_likelihood × route_risk_level | Compound route hazard |
| `feat_traffic_x_eta` | traffic_congestion × eta_variation | Traffic-driven delay pressure |
| `feat_driver_fatigue` | driver_behavior × fatigue_score | Human reliability signal |
| `feat_port_x_customs` | port_congestion × customs_clearance | Border delay risk |
| `feat_inventory_ratio` | warehouse_inventory ÷ historical_demand | Stock adequacy ratio |

**2. Polynomial features**
Squares (x²) and square roots (√x) of the top 6 risk-driver columns.
Allows Logistic Regression to model non-linear patterns it otherwise can't.

**3. Composite Risk Score**
A single engineered column averaging all risk signals:
- Positive contributors: delay_probability, disruption_likelihood,
  route_risk_level, traffic_congestion, weather_severity, port_congestion
- Negative contributors (inverted): driver_behavior_score,
  supplier_reliability_score, handling_equipment_availability

This is typically one of the top 3 most important features for Random Forest.

**4. Log transforms**
Applied to `shipping_costs`, `historical_demand`, `lead_time_days`
to compress right-skewed distributions.

**5. Feature-target correlation chart**
All features ranked by absolute Pearson correlation with the target label.
Red = engineered, blue = original.
Saved → `plots/04_feature_correlation_ranking.png`

**Output files:**
```
outputs/engineered_data.csv
outputs/models/all_feature_cols.pkl
```

---

## 🤖 Step 3 — Model Training

```bash
python step3_train_models.py
```

> ⏱️ Expected time: **3–10 minutes** depending on hardware.

### What happens:

**1. Load data + train/test split**
Stratified 80/20 split. Stratified = each class is proportionally
represented in both halves. Test set is locked away immediately.

**2. SMOTE — balance the training set**

Your raw class counts:
```
On-Time  : 23,944  (very dominant)
Delayed  :  5,011
At Risk  :  3,110  (minority — model would ignore this without SMOTE)
```

After SMOTE:
```
On-Time  : 19,155
Delayed  : 19,155
At Risk  : 19,155  ← synthetic samples created by interpolating
                      between real At-Risk neighbours in feature space
```

**3. Logistic Regression training**

```
Algorithm : Linear decision boundary in 50-dimensional feature space
Solver    : lbfgs (efficient for multi-class)
max_iter  : 3000 (ensures convergence on 57k SMOTE samples)
C         : 1.0  (regularisation — prevents overfitting)
Strategy  : 5-fold stratified cross-validation → refit on full SMOTE set
Output    : cv_accuracy ± std, test_accuracy
```

Why use it: Fast baseline that tells you the minimum bar Random Forest
must beat. Its coefficients also reveal which features drive each class
linearly — useful for explainability.

**4. Random Forest training + tuning**

```
Algorithm  : 500 independent decision trees, majority vote
Tuning     : RandomizedSearchCV — 20 random hyperparameter combinations
             each evaluated with 5-fold stratified CV
Search space:
  n_estimators      : [300, 500, 700]      ← number of trees
  max_depth         : [None, 15, 25, 35]   ← tree depth limit
  min_samples_split : [2, 4, 6]            ← split threshold
  min_samples_leaf  : [1, 2, 4]            ← leaf size limit
  max_features      : ["sqrt", "log2"]     ← features per split

class_weight = "balanced"  ← additional imbalance handling on top of SMOTE
```

Why use it: Each tree sees a random subset of data and features, making
the ensemble much more robust than any single tree. The majority vote
across 500 trees smooths out individual errors.

**5. Best model selection**
Both models' test accuracies are compared. The winner is saved as
`best_model.pkl` — this is what Step 4 uses for detailed analysis.

**Comparison table printed to terminal:**
```
┌──────────────────────────┬──────────────────┬──────────────────┐
│ Metric                   │ Logistic Reg.    │ Random Forest    │
├──────────────────────────┼──────────────────┼──────────────────┤
│ CV Accuracy              │ 0.XXXX           │ 0.XXXX           │
│ Test Accuracy            │ 0.XXXX           │ 0.XXXX           │
│ Train Time               │  XX.Xs           │  XX.Xs           │
│ Best Model?              │   NO             │ ✅ YES            │
└──────────────────────────┴──────────────────┴──────────────────┘
```

**Output files:**
```
outputs/models/Logistic_Regression.pkl
outputs/models/Random_Forest.pkl
outputs/models/best_model.pkl
outputs/models/best_model_name.pkl
outputs/models/test_split.pkl
outputs/models/final_feature_cols.pkl
outputs/reports/model_comparison.csv
outputs/plots/05_model_comparison.png
```

---

## 📊 Step 4 — Evaluation & Report

```bash
python step4_evaluate_report.py
```

### What happens:

**1. Load both models + test set**
Both Logistic Regression and Random Forest are loaded and evaluated
on the identical held-out test set for a fair comparison.

**2. Metrics computed for each model**

| Metric | What it measures |
|--------|-----------------|
| Accuracy | % of all predictions correct |
| Macro F1 | F1 averaged equally across all 3 classes — penalises ignoring minority classes |
| Weighted F1 | F1 weighted by class frequency |
| ROC-AUC (OvR) | Discrimination ability at all thresholds — closer to 1.0 = better |

**3. Side-by-side metrics chart**
4 metrics plotted as grouped bars — LR vs RF for each.
Saved → `plots/05_model_comparison.png`

**4. Dual confusion matrices**
Both models shown in a 2×2 grid — raw counts and normalised.
You can directly compare where each model makes mistakes.
Saved → `plots/06_confusion_matrix.png`

**5. ROC Curves (both models, one chart)**
Logistic Regression = dashed lines, Random Forest = solid lines.
3 curves per model (one per class), each with its AUC value.
Saved → `plots/07_roc_curves.png`

**6. Precision-Recall Curves (both models)**
More informative than ROC for the imbalanced classes.
Saved → `plots/08_precision_recall_curves.png`

**7. Feature Importance comparison**
- **Random Forest**: Gini importance scores (how much each feature
  reduces impurity across all 500 trees)
- **Logistic Regression**: Mean absolute coefficients across 3 classes
  (how strongly each feature pushes predictions toward each class)
Both shown side-by-side with engineered features highlighted in red.
Saved → `plots/09_feature_importances.png`

**8. Prediction confidence histograms**
Shows how confident each model is in its predictions.
Random Forest tends to be better calibrated; Logistic Regression
can be overconfident on linearly-separable regions.
Saved → `plots/10_confidence_distribution.png`

**9. Full text report**
Everything above written to `reports/final_evaluation_report.txt`
including the head-to-head comparison table with a winner indicator.

---

## 🚀 All-in-One (Recommended)

```bash
python main_pipeline.py
```

Runs Steps 1 → 4 automatically. Estimated total time: 5–15 minutes.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `KeyError: 'risk_classification'` | Check `TARGET_COL` in `config.py` matches exactly |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| SMOTE error: too few samples | Change `k_neighbors=5` to `k_neighbors=3` in step3 |
| Logistic Regression not converging | Increase `max_iter=3000` to `max_iter=5000` |
| Memory error | Reduce `n_iter=20` to `n_iter=10` in RandomizedSearchCV |
| Step 4 KeyError on model file | Make sure Step 3 ran fully (check `outputs/models/` folder) |

---

## 📈 Expected Terminal Output (Step 4 finale)

```
════════════════════════════════════════════════════════════
  LOGISTIC REGRESSION
  🎯 Accuracy : XX.XX%
  🎯 Macro F1 : 0.XXXX
  🎯 ROC-AUC  : 0.XXXX

  RANDOM FOREST
  🎯 Accuracy : XX.XX%
  🎯 Macro F1 : 0.XXXX
  🎯 ROC-AUC  : 0.XXXX

  🏆 Best Model : Random_Forest
════════════════════════════════════════════════════════════

✔ STEP 4 COMPLETE — PROJECT FINISHED ✅
```

---

## 🧠 Why These Two Models?

| Property | Logistic Regression | Random Forest |
|----------|--------------------|--------------| 
| Type | Linear | Non-linear (ensemble) |
| Interpretability | High — coefficients show direction | Medium — importance scores |
| Speed | Very fast (~seconds) | Moderate (~minutes) |
| Handles non-linearity | ❌ No | ✅ Yes |
| Handles feature interactions | ❌ No | ✅ Yes (automatically) |
| Expected accuracy on this data | Lower (~75–85%) | Higher (~85–94%) |
| Best for | Baseline + explainability | Production predictions |

Random Forest almost always wins on this type of high-dimensional
numeric data with non-linear risk relationships. Logistic Regression
provides the baseline and interpretability layer.
