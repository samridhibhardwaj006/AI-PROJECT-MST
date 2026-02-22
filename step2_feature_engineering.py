"""
step2_feature_engineering.py
==============================
STEP 2 — Feature Engineering

What this script does:
  • Loads the preprocessed dataset from Step 1
  • Creates interaction features from real column names
  • Adds polynomial features (square, square-root) on key numerics
  • Builds a Composite Risk Score from multiple risk signals
  • Saves enriched dataset as engineered_data.csv
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

print("\n" + "=" * 60)
print("  STEP 2 — FEATURE ENGINEERING")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────────────────────────────────────
df           = pd.read_csv(os.path.join(config.OUTPUT_DIR, "preprocessed_data.csv"))
num_cols     = joblib.load(os.path.join(config.MODEL_DIR, "num_cols.pkl"))
feature_cols = joblib.load(os.path.join(config.MODEL_DIR, "feature_cols.pkl"))

print(f"\n✔ Loaded preprocessed data  →  {df.shape}")
print(f"  Existing feature count: {len(feature_cols)}")

new_feats = []

def add_feat(name, values):
    df[name] = values
    new_feats.append(name)

# ─────────────────────────────────────────────────────────────────────────────
# 1. INTERACTION FEATURES  (using your actual column names)
# ─────────────────────────────────────────────────────────────────────────────
def cols_exist(*names):
    return all(c in df.columns for c in names)

# Risk signal interactions
if cols_exist("delay_probability", "eta_variation_hours"):
    add_feat("feat_delay_x_eta",
             df["delay_probability"] * df["eta_variation_hours"])

if cols_exist("disruption_likelihood_score", "route_risk_level"):
    add_feat("feat_disruption_x_route",
             df["disruption_likelihood_score"] * df["route_risk_level"])

if cols_exist("traffic_congestion_level", "eta_variation_hours"):
    add_feat("feat_traffic_x_eta",
             df["traffic_congestion_level"] * df["eta_variation_hours"])

if cols_exist("weather_condition_severity", "route_risk_level"):
    add_feat("feat_weather_x_route",
             df["weather_condition_severity"] * df["route_risk_level"])

if cols_exist("driver_behavior_score", "fatigue_monitoring_score"):
    add_feat("feat_driver_fatigue",
             df["driver_behavior_score"] * df["fatigue_monitoring_score"])

if cols_exist("supplier_reliability_score", "lead_time_days"):
    add_feat("feat_supplier_x_leadtime",
             df["supplier_reliability_score"] / (df["lead_time_days"].abs() + 1e-5))

if cols_exist("port_congestion_level", "customs_clearance_time"):
    add_feat("feat_port_x_customs",
             df["port_congestion_level"] * df["customs_clearance_time"])

if cols_exist("fuel_consumption_rate", "shipping_costs"):
    add_feat("feat_fuel_x_cost",
             df["fuel_consumption_rate"] * df["shipping_costs"])

if cols_exist("loading_unloading_time", "handling_equipment_availability"):
    add_feat("feat_load_x_equip",
             df["loading_unloading_time"] / (df["handling_equipment_availability"].abs() + 1e-5))

if cols_exist("warehouse_inventory_level", "historical_demand"):
    add_feat("feat_inventory_ratio",
             df["warehouse_inventory_level"] / (df["historical_demand"].abs() + 1e-5))

# ─────────────────────────────────────────────────────────────────────────────
# 2. POLYNOMIAL FEATURES on top risk drivers
# ─────────────────────────────────────────────────────────────────────────────
poly_candidates = [
    "delay_probability",
    "disruption_likelihood_score",
    "eta_variation_hours",
    "route_risk_level",
    "traffic_congestion_level",
    "delivery_time_deviation",
]
for col in poly_candidates:
    if col in df.columns:
        add_feat(f"feat_{col}_sq",   df[col] ** 2)
        add_feat(f"feat_{col}_sqrt", np.abs(df[col]) ** 0.5)

# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPOSITE RISK SCORE
# ─────────────────────────────────────────────────────────────────────────────
risk_pos = []   # higher value = more risk
risk_neg = []   # lower value = more risk (inverted)

for col in ["delay_probability", "disruption_likelihood_score",
            "route_risk_level", "traffic_congestion_level",
            "weather_condition_severity", "eta_variation_hours",
            "port_congestion_level", "fatigue_monitoring_score",
            "delivery_time_deviation"]:
    if col in df.columns:
        risk_pos.append(df[col])

for col in ["driver_behavior_score", "supplier_reliability_score",
            "handling_equipment_availability", "warehouse_inventory_level"]:
    if col in df.columns:
        risk_neg.append(-df[col])   # invert: low score = high risk

all_risk = risk_pos + risk_neg
if all_risk:
    add_feat("feat_composite_risk_score",
             sum(all_risk) / len(all_risk))

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOG TRANSFORMS on skewed columns
# ─────────────────────────────────────────────────────────────────────────────
log_candidates = ["shipping_costs", "historical_demand", "lead_time_days"]
for col in log_candidates:
    if col in df.columns:
        shifted = df[col] - df[col].min() + 1e-5
        add_feat(f"feat_{col}_log", np.log1p(shifted))

# ─────────────────────────────────────────────────────────────────────────────
# 5. REPORT NEW FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Engineered features added ({len(new_feats)}):")
for f in new_feats:
    print(f"    + {f}")

all_features = feature_cols + new_feats
joblib.dump(all_features, os.path.join(config.MODEL_DIR, "all_feature_cols.pkl"))

# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE-TARGET CORRELATION RANKING
# ─────────────────────────────────────────────────────────────────────────────
existing_all = [f for f in all_features if f in df.columns]
feat_corr = (
    df[existing_all]
    .corrwith(df["label"])
    .abs()
    .sort_values(ascending=False)
)
top20 = feat_corr.head(20)

plt.figure(figsize=(11, 7))
colors = ["#c0392b" if f in new_feats else "#2980b9" for f in top20.index]
plt.barh(range(len(top20)), top20.values, color=colors, edgecolor="black")
plt.yticks(range(len(top20)), top20.index, fontsize=9)
plt.gca().invert_yaxis()
plt.xlabel("| Pearson Correlation with Target |", fontsize=11)
plt.title("Top 20 Features — Absolute Correlation with Target\n"
          "(🔴 Engineered   🔵 Original)",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, "04_feature_correlation_ranking.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  Plot saved → 04_feature_correlation_ranking.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE ENRICHED DATA
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(config.OUTPUT_DIR, "engineered_data.csv")
df.to_csv(out_path, index=False)
print(f"\n✔ Engineered data saved  →  {out_path}")
print(f"  Total features for modelling: {len(existing_all)}")
print("\n✔ STEP 2 COMPLETE\n")
