"""
config.py
=========
Central configuration for Amazon Supply Chain Intelligence ML Pipeline.

LEAKAGE FIX:
  Removed 4 target-leaking columns that directly encode risk_classification:
    ✗ delay_probability         — pre-computed delay score (IS the target)
    ✗ disruption_likelihood_score — pre-computed risk label
    ✗ delivery_time_deviation   — measures actual delivery outcome
    ✗ route_risk_level          — direct risk classification signal

  Remaining 20 columns are genuine operational/sensor features
  that a real system would have BEFORE classification is done.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dynamic_supply_chain_logistics_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Target ──────────────────────────────────────────────────────────────────
TARGET_COL   = "risk_classification"
CLASS_LABELS = []    # auto-read from LabelEncoder at runtime

# ─── Feature groups ──────────────────────────────────────────────────────────
# NO string categorical columns — all features are numeric.
# 'timestamp' auto-dropped as ID-like in Step 1.
#
# REMOVED (leaky — directly derived from / equivalent to the target):
#   delay_probability, disruption_likelihood_score,
#   delivery_time_deviation, route_risk_level
#
# KEPT (genuine operational inputs available before classification):
CATEGORICAL_COLS = []

NUMERIC_COLS = [
    "vehicle_gps_latitude",        # real-time GPS position
    "vehicle_gps_longitude",       # real-time GPS position
    "fuel_consumption_rate",       # operational efficiency signal
    "eta_variation_hours",         # deviation from planned ETA
    "traffic_congestion_level",    # live traffic index
    "warehouse_inventory_level",   # stock availability
    "loading_unloading_time",      # handling speed at depot
    "handling_equipment_availability", # dock/crane availability
    "order_fulfillment_status",    # order processing state
    "weather_condition_severity",  # weather index at destination
    "port_congestion_level",       # port/hub congestion
    "shipping_costs",              # cost proxy for route complexity
    "supplier_reliability_score",  # upstream supplier performance
    "lead_time_days",              # planned lead time
    "historical_demand",           # demand volume context
    "iot_temperature",             # cargo temperature sensor
    "cargo_condition_status",      # cargo integrity check
    "customs_clearance_time",      # border crossing delay
    "driver_behavior_score",       # driver safety/performance
    "fatigue_monitoring_score",    # driver fatigue index
]

# ─── Train / Test split ──────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ─── SMOTE ───────────────────────────────────────────────────────────────────
SMOTE_STRATEGY = "auto"

# ─── Cross-validation folds ──────────────────────────────────────────────────
CV_FOLDS = 5
