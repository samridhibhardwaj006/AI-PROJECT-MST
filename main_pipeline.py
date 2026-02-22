"""
main_pipeline.py
=================
Master runner — executes all 4 steps in sequence.

Usage:
    python main_pipeline.py

This single command:
  1. Runs EDA & Preprocessing
  2. Runs Feature Engineering
  3. Trains & tunes all models (with SMOTE + RandomizedSearchCV)
  4. Evaluates the best model and saves all reports + plots
"""

import subprocess
import sys
import time
import os

STEPS = [
    ("step1_eda_preprocessing.py",   "Step 1 — EDA & Preprocessing"),
    ("step2_feature_engineering.py", "Step 2 — Feature Engineering"),
    ("step3_train_models.py",         "Step 3 — Model Training"),
    ("step4_evaluate_report.py",      "Step 4 — Evaluation & Report"),
]

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║    AMAZON SUPPLY CHAIN INTELLIGENCE — ML PIPELINE           ║
║    Delivery Delay Risk Multi-Class Classifier               ║
║    Target: On-Time  |  At Risk  |  Delayed                 ║
╚══════════════════════════════════════════════════════════════╝
"""

print(BANNER)

# Change CWD to script directory so imports resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

t_total = time.time()

for i, (script, label) in enumerate(STEPS, 1):
    print(f"\n{'▶'*2}  [{i}/{len(STEPS)}]  {label}")
    print("─" * 60)
    t0     = time.time()
    result = subprocess.run(
        [sys.executable, script],
        check=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n✖  {script} exited with error (code {result.returncode}).")
        print("   Fix the error above and re-run main_pipeline.py.")
        sys.exit(1)

    print(f"\n  ✔  {label} completed in {elapsed:.1f}s")

total_min = (time.time() - t_total) / 60
print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅  PIPELINE COMPLETE                                      ║
║  ⏱   Total time : {total_min:.1f} minutes{' '*(35 - len(f'{total_min:.1f}'))}║
║                                                              ║
║  📊  Plots   → outputs/plots/                               ║
║  📄  Reports → outputs/reports/                             ║
║  🤖  Models  → outputs/models/                              ║
╚══════════════════════════════════════════════════════════════╝
""")
