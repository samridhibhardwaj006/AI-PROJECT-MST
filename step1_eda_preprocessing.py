"""
step1_eda_preprocessing.py
===========================
STEP 1 — Exploratory Data Analysis & Preprocessing

What this script does:
  • Loads raw dataset and prints shape / dtypes
  • Auto-detects categorical vs numeric columns
  • Visualises target class distribution (bar + pie)
  • Plots histograms & boxplots for every numeric feature
  • Draws a feature correlation heatmap
  • Imputes missing values (median for numerics, mode for categoricals)
  • Clips outliers using 3×IQR fence
  • One-Hot encodes all categorical columns
  • Standard-scales all numeric columns
  • Saves preprocessed_data.csv + fitted scaler/imputer artefacts
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="husl")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 1 — EDA & PREPROCESSING")
print("=" * 60)

df = pd.read_csv(config.DATA_PATH)
print(f"\n✔ Dataset loaded  →  {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
print(f"\nColumns:\n  {df.columns.tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BASIC INFO
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Data Types ──────────────────────────────────────────────")
print(df.dtypes)

print("\n── Missing Values ──────────────────────────────────────────")
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols):
    print(missing_cols)
else:
    print("  None found ✔")

print("\n── Duplicate Rows ──────────────────────────────────────────")
n_dup = df.duplicated().sum()
print(f"  {n_dup} duplicates found", "→ dropping" if n_dup else "✔")
if n_dup:
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. AUTO-DETECT COLUMN TYPES
# ─────────────────────────────────────────────────────────────────────────────
# Drop ID-like columns (unique string per row)
drop_cols = [
    c for c in df.columns
    if df[c].dtype == object
    and df[c].nunique() == df.shape[0]
    and c != config.TARGET_COL
]
if drop_cols:
    print(f"\n  Dropping ID-like columns: {drop_cols}")
    df.drop(columns=drop_cols, inplace=True)

cat_cols = [
    c for c in df.columns
    if df[c].dtype == object and c != config.TARGET_COL
]
num_cols = [
    c for c in df.columns
    if df[c].dtype != object and c != config.TARGET_COL
]

print(f"\n  Categorical cols ({len(cat_cols)}): {cat_cols}")
print(f"  Numeric cols     ({len(num_cols)}): {num_cols}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TARGET DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Target Distribution ─────────────────────────────────────")
vc = df[config.TARGET_COL].value_counts()
print(vc)
print(f"\n  Class imbalance ratio (max/min): {vc.max()/vc.min():.2f}x")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

vc.plot(
    kind="bar", ax=axes[0],
    color=["#2ecc71", "#f39c12", "#e74c3c"],
    edgecolor="black", width=0.55,
)
axes[0].set_title("Target Class Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Delivery Status")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)
for bar in axes[0].patches:
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"{int(bar.get_height()):,}",
        ha="center", va="bottom", fontsize=10,
    )

vc.plot(
    kind="pie", ax=axes[1],
    autopct="%1.1f%%",
    colors=["#2ecc71", "#f39c12", "#e74c3c"],
    startangle=90, pctdistance=0.8,
)
axes[1].set_title("Target Class Share", fontsize=14, fontweight="bold")
axes[1].set_ylabel("")

plt.suptitle("Delivery Status — Class Distribution", fontsize=15,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, "01_target_distribution.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved → 01_target_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. NUMERIC DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
if num_cols:
    n = len(num_cols)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, col in enumerate(num_cols):
        data = df[col].dropna()

        # Histogram
        axes[0, i].hist(data, bins=40, color="#3498db",
                        edgecolor="white", alpha=0.85)
        axes[0, i].set_title(f"{col}\n(Histogram)", fontsize=10)
        axes[0, i].set_ylabel("Frequency")

        # Boxplot
        axes[1, i].boxplot(
            data, vert=True, patch_artist=True,
            boxprops=dict(facecolor="#3498db", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
        )
        axes[1, i].set_title(f"{col}\n(Boxplot)", fontsize=10)

    plt.suptitle("Numeric Feature Distributions", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, "02_numeric_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved → 02_numeric_distributions.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
if len(num_cols) > 1:
    plt.figure(figsize=(max(8, len(num_cols)), max(6, len(num_cols) - 1)))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5,
    )
    plt.title("Numeric Feature Correlation Matrix", fontsize=14,
              fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, "03_correlation_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved → 03_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. IMPUTE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
if num_cols:
    num_imputer = SimpleImputer(strategy="median")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    joblib.dump(num_imputer, os.path.join(config.MODEL_DIR, "num_imputer.pkl"))

if cat_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    joblib.dump(cat_imputer, os.path.join(config.MODEL_DIR, "cat_imputer.pkl"))

print("\n  ✔ Missing values imputed.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. OUTLIER CLIPPING (IQR × 3 — robust fence)
# ─────────────────────────────────────────────────────────────────────────────
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 3 * IQR, Q3 + 3 * IQR
    n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lo, hi)
    if n_clipped:
        print(f"  Clipped {n_clipped} outliers in '{col}'")

print("  ✔ Outlier clipping complete.")

# ─────────────────────────────────────────────────────────────────────────────
# 9. ENCODE TARGET LABEL
# ─────────────────────────────────────────────────────────────────────────────
le_target = LabelEncoder()
df["label"] = le_target.fit_transform(df[config.TARGET_COL])
joblib.dump(le_target, os.path.join(config.MODEL_DIR, "label_encoder.pkl"))
class_map = dict(zip(le_target.classes_,
                     le_target.transform(le_target.classes_)))
print(f"\n  Target encoding: {class_map}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. ONE-HOT ENCODE CATEGORICALS
# ─────────────────────────────────────────────────────────────────────────────
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
print(f"\n  Shape after OHE: {df_encoded.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. STANDARD SCALE NUMERICS
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
joblib.dump(scaler, os.path.join(config.MODEL_DIR, "scaler.pkl"))
print("  ✔ Standard scaling applied.")

# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE PREPROCESSED DATA & METADATA
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(config.OUTPUT_DIR, "preprocessed_data.csv")
df_encoded.to_csv(out_path, index=False)

feature_cols = [
    c for c in df_encoded.columns
    if c not in [config.TARGET_COL, "label"]
]
joblib.dump(feature_cols, os.path.join(config.MODEL_DIR, "feature_cols.pkl"))
joblib.dump(num_cols,     os.path.join(config.MODEL_DIR, "num_cols.pkl"))
joblib.dump(cat_cols,     os.path.join(config.MODEL_DIR, "cat_cols.pkl"))

print(f"\n✔ Preprocessed data saved  →  {out_path}")
print(f"  Features after encoding: {len(feature_cols)}")
print("\n✔ STEP 1 COMPLETE\n")
