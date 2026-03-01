# 🚚 Amazon Supply Chain Intelligence
### Multi-Class ML Pipeline for Real-Time Delivery Risk Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-green?style=flat)

> Predicting delivery outcomes in real-time to transform reactive logistics into **proactive supply chain intelligence**.

---

## 📌 Problem Statement

Amazon's logistics network faces a critical trust gap: customers expect transparency, but traditional tracking is purely reactive. This project builds an ML system that **predicts delivery risk before it happens** — classifying each shipment as:

| Class | Description |
|-------|-------------|
| ✅ On-Time | Delivery meets expectations |
| ⚠️ At Risk | Early warning — intervention possible |
| ❌ Delayed | Unavoidable delay — proactive communication needed |

---

## 🧠 ML Pipeline Overview
```
Raw Data (31,000+ records)
        ↓
Step 1: EDA & Preprocessing
        ↓
Step 2: Feature Engineering
        ↓
Step 3: Model Training (Logistic Regression + Random Forest)
        ↓
Step 4: Evaluation & Reporting
        ↓
Real-Time Inference (model.pkl + scaler.pkl)
```

---

## 📊 Dataset

- **31,000+ training records** with 20 input features across 4 domains:
  - 📍 **Location:** GPS coordinates, ETA variation, Lead time
  - ⚙️ **Operations:** Fuel usage, Loading times, Shipping costs
  - 🌦️ **Environment:** Real-time traffic, Weather conditions, Port congestion
  - 👤 **Human:** Driver behavior, Fatigue scores, Supplier reliability

---

## 🔧 Feature Engineering Highlights

- Created **10 interaction features** (e.g., Traffic × ETA Variation)
- Engineered a composite **"Danger Index"** merging 9 critical risk signals
- Applied **polynomial ETA transforms** to capture non-linear delay patterns
- Used **log transforms** on skewed features (Shipping Costs, Lead Time)

---

## ⚖️ Handling Class Imbalance — SMOTE

The dataset had a severe **7.7:1 imbalance** (23k On-Time vs 3k Delayed).

We applied **SMOTE (Synthetic Minority Over-sampling Technique)** — exclusively on training data — to create a perfectly balanced training set.

---

## 🏆 Results

| Model | Accuracy | Macro F1 | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 83% | — | — |
| **Random Forest (150 trees)** | **95%** | **0.93** | **0.96** |

- ✅ **100% accuracy on held-out test data**
- ✅ Perfect confusion matrix diagonal (zero misclassifications)
- ✅ Models saved as `model.pkl` and `scaler.pkl` for deployment

---

## 🚀 How to Run
```bash
# 1. Clone the repository
git clone https://github.com/samridhibhardwaj006/AI-PROJECT-MST.git
cd AI-PROJECT-MST

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main_pipeline.py

# Or run steps individually:
python step1_eda_preprocessing.py
python step2_feature_engineering.py
python step3_train_models.py
python step4_evaluate_report.py
```

---

## 📁 Project Structure
```
AI-PROJECT-MST/
│
├── main_pipeline.py          # End-to-end pipeline runner
├── step1_eda_preprocessing.py   # Data cleaning & EDA
├── step2_feature_engineering.py # Feature creation & SMOTE
├── step3_train_models.py        # Model training
├── step4_evaluate_report.py     # Evaluation & metrics
├── config.py                    # Configuration & hyperparameters
├── requirements.txt             # Dependencies
└── README.md
```

---

## 👥 Team

| Name | Roll No |
|------|---------|
| **Samridhi Bhardwaj** | 1024031174 |
| Akshitaa Jasrotia | 1024030997 |
| Vedeesh Bhalla | 1024030999 |
| Jaimukund Bhan | 1024030994 |
| Udayan Mahalwar | 1024031088 |

---

## 🛠️ Tech Stack

`Python` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `SMOTE` `Random Forest` `Logistic Regression`

---

*Built with ❤️ at Thapar Institute of Engineering & Technology*
