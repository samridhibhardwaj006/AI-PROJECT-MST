"""
step3_train_models.py
======================
STEP 3 — Model Training

Models trained:
  1. Logistic Regression  — regularised linear baseline
  2. Random Forest        — depth-limited bagging ensemble

Leakage prevention:
  4 target-leaking columns removed in config.py (delay_probability,
  disruption_likelihood_score, delivery_time_deviation, route_risk_level).
  Models now learn from genuine operational features only.

Overfitting prevention:
  • Logistic Regression : C=0.1  (stronger L2 regularisation)
  • Random Forest       : max_depth capped in search space (≤20)
                          min_samples_leaf ≥ 4
  • Both               : SMOTE balances classes without leaking test info
"""

import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)

import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

print("\n" + "=" * 60)
print("  STEP 3 — MODEL TRAINING")
print("  Models: Logistic Regression  |  Random Forest")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df           = pd.read_csv(os.path.join(config.OUTPUT_DIR, "engineered_data.csv"))
all_features = joblib.load(os.path.join(config.MODEL_DIR, "all_feature_cols.pkl"))

# Keep only features that exist AND are not leaky
# (leaky cols were already excluded from config.NUMERIC_COLS so they
#  were never scaled/encoded — they simply won't appear in all_features)
all_features = [f for f in all_features if f in df.columns]

X = df[all_features].fillna(0).values
y = df["label"].values

print(f"\n✔ Feature matrix : {X.shape}")
unique, counts = np.unique(y, return_counts=True)
print(f"  Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT  (stratified)
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = config.TEST_SIZE,
    random_state = config.RANDOM_STATE,
    stratify     = y,
)
print(f"\n  Train : {X_train.shape[0]:,} samples")
print(f"  Test  : {X_test.shape[0]:,} samples")

# Save test split for Step 4
joblib.dump((X_test, y_test),
            os.path.join(config.MODEL_DIR, "test_split.pkl"))
joblib.dump(all_features,
            os.path.join(config.MODEL_DIR, "final_feature_cols.pkl"))

# ─────────────────────────────────────────────────────────────────────────────
# SMOTE — BALANCE TRAINING SET ONLY
# (applied AFTER split — no test data contamination)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Applying SMOTE to balance training classes...")
smote = SMOTE(
    sampling_strategy = config.SMOTE_STRATEGY,
    random_state      = config.RANDOM_STATE,
    k_neighbors       = 5,
)
X_res, y_res = smote.fit_resample(X_train, y_train)
unique_r, counts_r = np.unique(y_res, return_counts=True)
print(f"  After SMOTE : {X_res.shape[0]:,} training samples")
print(f"  Balanced    : {dict(zip(unique_r.tolist(), counts_r.tolist()))}")

cv = StratifiedKFold(
    n_splits     = config.CV_FOLDS,
    shuffle      = True,
    random_state = config.RANDOM_STATE,
)

results    = {}
best_acc   = 0.0
best_name  = None
best_model = None

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — LOGISTIC REGRESSION
# C=0.1 → stronger L2 regularisation prevents over-reliance on any one feature
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  ── Training: Logistic Regression ──────────────────────────")
print("     Regularisation : C=0.1  (L2, stronger penalty)")
print("     Strategy       : 5-fold CV → refit on full SMOTE set")
t0 = time.time()

lr_clf = LogisticRegression(
    max_iter     = 3000,
    C            = 0.1,        # stronger regularisation vs default C=1.0
    solver       = "lbfgs",
    random_state = config.RANDOM_STATE,
)

lr_cv_scores = cross_val_score(
    lr_clf, X_res, y_res,
    cv=cv, scoring="accuracy", n_jobs=-1,
)
lr_cv_score = lr_cv_scores.mean()
lr_cv_std   = lr_cv_scores.std()

lr_clf.fit(X_res, y_res)
lr_test_acc = accuracy_score(y_test, lr_clf.predict(X_test))
lr_elapsed  = time.time() - t0

print(f"\n    CV Accuracy  : {lr_cv_score:.4f}  (±{lr_cv_std:.4f})")
print(f"    Test Accuracy: {lr_test_acc:.4f}  ({lr_test_acc*100:.2f}%)")
print(f"    Train Time   : {lr_elapsed:.1f}s")

joblib.dump(lr_clf, os.path.join(config.MODEL_DIR, "Logistic_Regression.pkl"))
results["Logistic_Regression"] = {
    "cv_accuracy":   lr_cv_score,
    "cv_std":        lr_cv_std,
    "test_accuracy": lr_test_acc,
    "train_time_s":  lr_elapsed,
}

if lr_test_acc > best_acc:
    best_acc   = lr_test_acc
    best_name  = "Logistic_Regression"
    best_model = lr_clf

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — RANDOM FOREST
# max_depth capped at 20, min_samples_leaf ≥ 4 → prevents memorisation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  ── Training: Random Forest ────────────────────────────────")
print("     Regularisation : max_depth ≤ 20, min_samples_leaf ≥ 4")
print("     Strategy       : RandomizedSearchCV (20 combos × 5-fold CV)")
t0 = time.time()

rf_base = RandomForestClassifier(
    class_weight = "balanced",
    random_state = config.RANDOM_STATE,
    n_jobs       = -1,
)

rf_param_grid = {
    "n_estimators":      [200, 300, 400],
    "max_depth":         [8, 12, 16, 20],     # capped — no None/unlimited
    "min_samples_split": [10, 20, 30],         # needs more samples to split
    "min_samples_leaf":  [4, 8, 12],           # larger leaves = less overfitting
    "max_features":      ["sqrt", "log2"],
}

rf_search = RandomizedSearchCV(
    rf_base,
    param_distributions = rf_param_grid,
    n_iter              = 20,
    cv                  = cv,
    scoring             = "accuracy",
    random_state        = config.RANDOM_STATE,
    n_jobs              = -1,
    verbose             = 1,
    refit               = True,
)
rf_search.fit(X_res, y_res)

rf_best     = rf_search.best_estimator_
rf_cv_score = rf_search.best_score_
rf_test_acc = accuracy_score(y_test, rf_best.predict(X_test))
rf_elapsed  = time.time() - t0

print(f"\n    Best Params  : {rf_search.best_params_}")
print(f"    CV Accuracy  : {rf_cv_score:.4f}")
print(f"    Test Accuracy: {rf_test_acc:.4f}  ({rf_test_acc*100:.2f}%)")
print(f"    Train Time   : {rf_elapsed:.1f}s")

joblib.dump(rf_best, os.path.join(config.MODEL_DIR, "Random_Forest.pkl"))
results["Random_Forest"] = {
    "cv_accuracy":   rf_cv_score,
    "cv_std":        None,
    "test_accuracy": rf_test_acc,
    "train_time_s":  rf_elapsed,
    "best_params":   rf_search.best_params_,
}

if rf_test_acc > best_acc:
    best_acc   = rf_test_acc
    best_name  = "Random_Forest"
    best_model = rf_best

# ─────────────────────────────────────────────────────────────────────────────
# SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(config.MODEL_DIR, "best_model.pkl"))
joblib.dump(best_name,  os.path.join(config.MODEL_DIR, "best_model_name.pkl"))

print(f"\n{'─'*55}")
print(f"  🏆 Best model   : {best_name}")
print(f"  🎯 Test Accuracy: {best_acc:.4f}  ({best_acc*100:.2f}%)")
print(f"{'─'*55}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE COMPARISON TABLE & CHART
# ─────────────────────────────────────────────────────────────────────────────
res_df = (
    pd.DataFrame(results)
    .T.reset_index()
    .rename(columns={"index": "Model"})
)
res_df.to_csv(os.path.join(config.REPORT_DIR, "model_comparison.csv"),
              index=False)

model_names = list(results.keys())
test_accs   = [results[m]["test_accuracy"] for m in model_names]
cv_accs     = [results[m]["cv_accuracy"]   for m in model_names]
colors      = ["#27ae60" if m == best_name else "#3498db" for m in model_names]
bar_labels  = ["Logistic\nRegression", "Random\nForest"]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

bars = axes[0].bar(bar_labels, test_accs, color=colors,
                   edgecolor="black", width=0.45)
axes[0].set_ylim(max(0, min(test_accs) - 0.10), 1.03)
axes[0].set_ylabel("Accuracy", fontsize=12)
axes[0].set_title("Test Set Accuracy\n(Green = Best Model)",
                  fontsize=13, fontweight="bold")
for bar, acc in zip(bars, test_accs):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{acc*100:.2f}%",
        ha="center", va="bottom", fontsize=13, fontweight="bold",
    )

x     = np.arange(len(model_names))
width = 0.32
b1 = axes[1].bar(x - width/2, cv_accs,   width, label="CV Accuracy",
                 color="#3498db", edgecolor="black", alpha=0.85)
b2 = axes[1].bar(x + width/2, test_accs, width, label="Test Accuracy",
                 color="#27ae60", edgecolor="black", alpha=0.85)
axes[1].set_ylim(max(0, min(cv_accs + test_accs) - 0.10), 1.03)
axes[1].set_xticks(x)
axes[1].set_xticklabels(bar_labels)
axes[1].set_ylabel("Accuracy", fontsize=12)
axes[1].set_title("CV vs Test Accuracy\n(Overfitting Check)",
                  fontsize=13, fontweight="bold")
axes[1].legend(fontsize=11)
for bar in list(b1) + list(b2):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.004,
        f"{bar.get_height()*100:.1f}%",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

plt.suptitle("Amazon Supply Chain — Model Comparison",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, "05_model_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  Plot saved → 05_model_comparison.png")

print(f"""
  ┌──────────────────────────┬──────────────────┬──────────────────┐
  │ Metric                   │ Logistic Reg.    │ Random Forest    │
  ├──────────────────────────┼──────────────────┼──────────────────┤
  │ CV Accuracy              │ {lr_cv_score:.4f}           │ {rf_cv_score:.4f}           │
  │ Test Accuracy            │ {lr_test_acc:.4f}           │ {rf_test_acc:.4f}           │
  │ Train Time               │ {lr_elapsed:5.1f}s           │ {rf_elapsed:5.1f}s           │
  │ Best Model?              │ {"✅ YES" if best_name=="Logistic_Regression" else "   NO"}             │ {"✅ YES" if best_name=="Random_Forest" else "   NO"}             │
  └──────────────────────────┴──────────────────┴──────────────────┘
""")
print("\n✔ STEP 3 COMPLETE\n")
