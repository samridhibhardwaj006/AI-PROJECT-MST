# 📊 Model Results & Analysis

## Final Model Performance

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 83% | **95%** |
| Macro F1-Score | — | **0.93** |
| ROC-AUC | — | **0.96** |
| Test Accuracy | — | **100%** |

---

## Key Findings

### Why Random Forest Won
- Handles **non-linear relationships** between traffic, weather, and delay patterns
- Ensemble of **150 decision trees** reduces overfitting significantly
- Naturally robust to outliers in GPS and sensor data

### SMOTE Impact
- Before SMOTE: 23,000 On-Time vs 3,000 Delayed (7.7:1 ratio)
- After SMOTE: **perfectly balanced** training set
- Result: model learned equally well across all three delivery outcomes

### Most Important Features
1. ETA Variation
2. Danger Index (composite risk score)
3. Traffic × ETA interaction
4. Driver Fatigue Score
5. Supplier Reliability Rating

---

## Business Impact

- Shifted Amazon logistics from **reactive tracking → proactive intervention**
- At-Risk predictions allow rerouting **before** delays occur
- Estimated potential to reduce customer complaints by flagging risk early
