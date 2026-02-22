"""
step4_evaluate_report.py
=========================
STEP 4 — Evaluation & Final Report

Models evaluated:
  1. Logistic Regression  — linear baseline
  2. Random Forest        — best model (bagging ensemble)

What this script does:
  • Loads best model + held-out test set from Step 3
  • Computes Accuracy, Macro F1, Weighted F1, ROC-AUC (OvR)
  • Prints full per-class classification report
  • Saves confusion matrices (count + normalised)
  • Saves multi-class ROC curves with per-class AUC
  • Saves Precision-Recall curves
  • Saves Feature Importance chart — top 30 (Random Forest only)
  • Saves side-by-side comparison of both models
  • Writes complete text evaluation report to outputs/reports/
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import label_binarize

import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

print("\n" + "=" * 60)
print("  STEP 4 — EVALUATION & REPORT")
print("  Models: Logistic Regression  |  Random Forest")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
best_model  = joblib.load(os.path.join(config.MODEL_DIR, "best_model.pkl"))
best_name   = joblib.load(os.path.join(config.MODEL_DIR, "best_model_name.pkl"))
le_target   = joblib.load(os.path.join(config.MODEL_DIR, "label_encoder.pkl"))
lr_model    = joblib.load(os.path.join(config.MODEL_DIR, "Logistic_Regression.pkl"))
rf_model    = joblib.load(os.path.join(config.MODEL_DIR, "Random_Forest.pkl"))
X_test, y_test = joblib.load(os.path.join(config.MODEL_DIR, "test_split.pkl"))
all_features   = joblib.load(os.path.join(config.MODEL_DIR, "final_feature_cols.pkl"))

class_names = le_target.classes_.tolist()
n_classes   = len(class_names)

print(f"\n  Best model : {best_name}")
print(f"  Classes    : {class_names}")
print(f"  Test size  : {len(y_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIONS — BOTH MODELS
# ─────────────────────────────────────────────────────────────────────────────
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

def evaluate_model(model, name):
    """Return dict of all metrics for one model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return {
        "name":        name,
        "y_pred":      y_pred,
        "y_prob":      y_prob,
        "accuracy":    accuracy_score(y_test, y_pred),
        "macro_f1":    f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "micro_f1":    f1_score(y_test, y_pred, average="micro"),
        "roc_auc":     roc_auc_score(y_test_bin, y_prob,
                                      multi_class="ovr", average="macro"),
        "report":      classification_report(y_test, y_pred,
                                              target_names=class_names),
        "cm":          confusion_matrix(y_test, y_pred),
    }

lr_eval = evaluate_model(lr_model, "Logistic_Regression")
rf_eval = evaluate_model(rf_model, "Random_Forest")

# Best model eval for detailed plots
best_eval = rf_eval if best_name == "Random_Forest" else lr_eval

# ─────────────────────────────────────────────────────────────────────────────
# PRINT RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_metrics(ev):
    print(f"\n  ── {ev['name']} ──────────────────────────────────────")
    print(f"    Test Accuracy       : {ev['accuracy']:.4f}  ({ev['accuracy']*100:.2f}%)")
    print(f"    Macro F1-Score      : {ev['macro_f1']:.4f}")
    print(f"    Weighted F1-Score   : {ev['weighted_f1']:.4f}")
    print(f"    ROC-AUC (OvR Macro) : {ev['roc_auc']:.4f}")
    print(f"\n  Classification Report:\n{ev['report']}")

print_metrics(lr_eval)
print_metrics(rf_eval)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — SIDE-BY-SIDE MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
metrics_names = ["Accuracy", "Macro F1", "Weighted F1", "ROC-AUC"]
lr_scores  = [lr_eval["accuracy"], lr_eval["macro_f1"],
              lr_eval["weighted_f1"], lr_eval["roc_auc"]]
rf_scores  = [rf_eval["accuracy"], rf_eval["macro_f1"],
              rf_eval["weighted_f1"], rf_eval["roc_auc"]]

x     = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - width/2, lr_scores, width, label="Logistic Regression",
            color="#3498db", edgecolor="black", alpha=0.88)
b2 = ax.bar(x + width/2, rf_scores, width, label="Random Forest",
            color="#27ae60", edgecolor="black", alpha=0.88)

ax.set_ylim(max(0, min(lr_scores + rf_scores) - 0.12), 1.05)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Logistic Regression vs Random Forest — All Metrics",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, "05_model_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved → 05_model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — CONFUSION MATRICES (both models, 2×2 grid)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for row, ev in enumerate([lr_eval, rf_eval]):
    cm      = ev["cm"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[row, 0],
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="gray",
                annot_kws={"size": 12, "weight": "bold"})
    axes[row, 0].set_title(f"{ev['name']}\nConfusion Matrix (Counts)",
                            fontsize=12, fontweight="bold")
    axes[row, 0].set_xlabel("Predicted"); axes[row, 0].set_ylabel("Actual")

    # Normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[row, 1],
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="gray", vmin=0, vmax=1,
                annot_kws={"size": 12, "weight": "bold"})
    axes[row, 1].set_title(f"{ev['name']}\nConfusion Matrix (Normalised)",
                            fontsize=12, fontweight="bold")
    axes[row, 1].set_xlabel("Predicted"); axes[row, 1].set_ylabel("Actual")

plt.suptitle("Confusion Matrices — Logistic Regression vs Random Forest",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, "06_confusion_matrix.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved → 06_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — ROC CURVES (both models on same chart)
# ─────────────────────────────────────────────────────────────────────────────
colors_class = ["#e74c3c", "#f39c12", "#2ecc71"]
styles_model = {"Logistic_Regression": "--", "Random_Forest": "-"}
lw_model     = {"Logistic_Regression": 1.8, "Random_Forest": 2.5}

fig, ax = plt.subplots(figsize=(10, 7))

for ev in [lr_eval, rf_eval]:
    short = "LR" if "Logistic" in ev["name"] else "RF"
    ls    = styles_model[ev["name"]]
    lw    = lw_model[ev["name"]]
    for i, (cls, col) in enumerate(zip(class_names, colors_class)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], ev["y_prob"][:, i])
        auc          = roc_auc_score(y_test_bin[:, i], ev["y_prob"][:, i])
        ax.plot(fpr, tpr, color=col, lw=lw, linestyle=ls,
                label=f"[{short}] {cls}  AUC={auc:.3f}")

ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — LR (dashed) vs RF (solid)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, "07_roc_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved → 07_roc_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — PRECISION-RECALL CURVES (both models)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

for ev in [lr_eval, rf_eval]:
    short = "LR" if "Logistic" in ev["name"] else "RF"
    ls    = styles_model[ev["name"]]
    lw    = lw_model[ev["name"]]
    for i, (cls, col) in enumerate(zip(class_names, colors_class)):
        prec, rec, _ = precision_recall_curve(y_test_bin[:, i],
                                               ev["y_prob"][:, i])
        ap            = average_precision_score(y_test_bin[:, i],
                                                ev["y_prob"][:, i])
        ax.plot(rec, prec, color=col, lw=lw, linestyle=ls,
                label=f"[{short}] {cls}  AP={ap:.3f}")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves — LR (dashed) vs RF (solid)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower left")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, "08_precision_recall_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved → 08_precision_recall_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — FEATURE IMPORTANCES (Random Forest only — LR has coefficients)
# ─────────────────────────────────────────────────────────────────────────────

# ── Random Forest feature importances ────────────────────────────────────────
rf_imp = pd.Series(
    rf_model.feature_importances_,
    index=all_features[:len(rf_model.feature_importances_)]
).sort_values(ascending=False)

top30_rf = rf_imp.head(30)
colors_rf = ["#c0392b" if "feat_" in f else "#8e44ad" for f in top30_rf.index]

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

axes[0].barh(range(len(top30_rf)), top30_rf.values[::-1],
             color=colors_rf[::-1], edgecolor="black")
axes[0].set_yticks(range(len(top30_rf)))
axes[0].set_yticklabels(top30_rf.index[::-1], fontsize=8)
axes[0].set_xlabel("Importance Score", fontsize=11)
axes[0].set_title("Random Forest — Top 30 Feature Importances\n"
                  "(🔴 Engineered   🟣 Original)",
                  fontsize=12, fontweight="bold")

# ── Logistic Regression coefficients (mean absolute across classes) ───────────
lr_coef = np.abs(lr_model.coef_).mean(axis=0)
lr_imp  = pd.Series(
    lr_coef,
    index=all_features[:len(lr_coef)]
).sort_values(ascending=False)

top30_lr  = lr_imp.head(30)
colors_lr = ["#c0392b" if "feat_" in f else "#2980b9" for f in top30_lr.index]

axes[1].barh(range(len(top30_lr)), top30_lr.values[::-1],
             color=colors_lr[::-1], edgecolor="black")
axes[1].set_yticks(range(len(top30_lr)))
axes[1].set_yticklabels(top30_lr.index[::-1], fontsize=8)
axes[1].set_xlabel("Mean |Coefficient|", fontsize=11)
axes[1].set_title("Logistic Regression — Top 30 Feature Coefficients\n"
                  "(🔴 Engineered   🔵 Original)",
                  fontsize=12, fontweight="bold")

plt.suptitle("Feature Importance Comparison — RF vs LR",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, "09_feature_importances.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved → 09_feature_importances.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — PREDICTION CONFIDENCE DISTRIBUTION (both models)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, ev in zip(axes, [lr_eval, rf_eval]):
    max_proba = ev["y_prob"].max(axis=1)
    ax.hist(max_proba, bins=50, color="#2980b9", edgecolor="white", alpha=0.85)
    ax.axvline(max_proba.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {max_proba.mean():.3f}")
    ax.set_xlabel("Max Predicted Probability (Confidence)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{ev['name']}\nPrediction Confidence Distribution",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, "10_confidence_distribution.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved → 10_confidence_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# FULL TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────
report_path = os.path.join(config.REPORT_DIR, "final_evaluation_report.txt")
sep         = "=" * 70
sep2        = "─" * 70

with open(report_path, "w", encoding="utf-8") as f:
    f.write(sep + "\n")
    f.write("  AMAZON SUPPLY CHAIN INTELLIGENCE\n")
    f.write("  DELIVERY DELAY RISK PREDICTOR — FINAL EVALUATION REPORT\n")
    f.write("  Models: Logistic Regression  |  Random Forest\n")
    f.write(sep + "\n\n")

    f.write(f"  Best Model        : {best_name}\n\n")

    for ev in [lr_eval, rf_eval]:
        tag = "  ← BEST MODEL" if ev["name"] == best_name else ""
        f.write(sep2 + "\n")
        f.write(f"  {ev['name']}{tag}\n")
        f.write(sep2 + "\n")
        f.write(f"  Test Accuracy     : {ev['accuracy']:.4f}  ({ev['accuracy']*100:.2f}%)\n")
        f.write(f"  Macro F1-Score    : {ev['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1-Score : {ev['weighted_f1']:.4f}\n")
        f.write(f"  Micro F1-Score    : {ev['micro_f1']:.4f}\n")
        f.write(f"  ROC-AUC (OvR Mac) : {ev['roc_auc']:.4f}\n\n")

        per_class_auc = {
            cls: roc_auc_score(y_test_bin[:, i], ev["y_prob"][:, i])
            for i, cls in enumerate(class_names)
        }
        f.write("  Per-Class AUC:\n")
        for cls, auc in per_class_auc.items():
            f.write(f"    {cls:<20s}: {auc:.4f}\n")
        f.write("\n")

        f.write("  Classification Report:\n")
        f.write(ev["report"] + "\n")

        cm_df = pd.DataFrame(ev["cm"], index=class_names, columns=class_names)
        f.write("  Confusion Matrix (Raw Counts):\n")
        f.write(cm_df.to_string() + "\n\n")

        cm_norm = ev["cm"].astype(float) / ev["cm"].sum(axis=1, keepdims=True)
        cm_norm_df = pd.DataFrame(cm_norm.round(3),
                                   index=class_names, columns=class_names)
        f.write("  Confusion Matrix (Normalised):\n")
        f.write(cm_norm_df.to_string() + "\n\n")

    # Side-by-side comparison table
    f.write(sep2 + "\n")
    f.write("  HEAD-TO-HEAD COMPARISON\n")
    f.write(sep2 + "\n")
    f.write(f"  {'Metric':<25} {'Logistic Regression':>22} {'Random Forest':>18}\n")
    f.write("  " + "─" * 65 + "\n")
    metrics_map = [
        ("Test Accuracy",    "accuracy"),
        ("Macro F1",         "macro_f1"),
        ("Weighted F1",      "weighted_f1"),
        ("ROC-AUC (OvR)",   "roc_auc"),
    ]
    for label, key in metrics_map:
        lv = lr_eval[key]; rv = rf_eval[key]
        winner = "←" if rv > lv else ("→" if lv > rv else "=")
        f.write(f"  {label:<25} {lv:>18.4f}   {rv:>14.4f}  {winner}\n")

    f.write(f"\n  Winner: {best_name}\n")
    f.write(sep + "\n")
    f.write("END OF REPORT\n")
    f.write(sep + "\n")

print(f"\n✔ Full report saved  →  {report_path}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE MODEL COMPARISON CSV
# ─────────────────────────────────────────────────────────────────────────────
comp_df = pd.DataFrame({
    "Model":        ["Logistic_Regression", "Random_Forest"],
    "Accuracy":     [lr_eval["accuracy"],    rf_eval["accuracy"]],
    "Macro_F1":     [lr_eval["macro_f1"],    rf_eval["macro_f1"]],
    "Weighted_F1":  [lr_eval["weighted_f1"], rf_eval["weighted_f1"]],
    "ROC_AUC":      [lr_eval["roc_auc"],     rf_eval["roc_auc"]],
})
comp_df.to_csv(os.path.join(config.REPORT_DIR, "model_comparison.csv"),
               index=False)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL TERMINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
{'='*60}
  LOGISTIC REGRESSION
  🎯 Accuracy : {lr_eval['accuracy']*100:.2f}%
  🎯 Macro F1 : {lr_eval['macro_f1']:.4f}
  🎯 ROC-AUC  : {lr_eval['roc_auc']:.4f}

  RANDOM FOREST
  🎯 Accuracy : {rf_eval['accuracy']*100:.2f}%
  🎯 Macro F1 : {rf_eval['macro_f1']:.4f}
  🎯 ROC-AUC  : {rf_eval['roc_auc']:.4f}

  🏆 Best Model : {best_name}
{'='*60}
""")
print("\n✔ STEP 4 COMPLETE — PROJECT FINISHED ✅\n")
