# 🏦 Credit Risk Scoring Model

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LightGBM%20%7C%20RF-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Grade-A-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Complete-success?style=for-the-badge"/>
</p>

> **Predicting credit card customer default risk using an ensemble of machine learning models — with full EDA, feature engineering, hyperparameter tuning, and a live prediction interface.**

---

## 📌 Overview

Banks face significant financial losses when customers default on credit payments. This project builds an end-to-end **credit risk scoring system** that predicts whether a customer will fall into the **30+ Days Past Due (DPD)** bucket — classifying them as a credit risk before it happens.

The model takes in customer account history, payment behaviour, and loan enquiry patterns to output a **credit score (300–850)**, a **risk band (LOW / MEDIUM / HIGH)**, and a **bank recommendation (Approve / Caution / Reject)**.

---

## 🎯 Problem Statement

| Item | Detail |
|---|---|
| **Target Variable** | `Bad_label` — 0 = Good Credit, 1 = Bad Credit |
| **Definition of Bad** | Customer falls into 30+ DPD bucket |
| **Evaluation Metric** | Gini Coefficient |
| **Benchmark Gini** | 37.9 |
| **Class Imbalance** | ~18% bad rate |

---

## 🗃️ Dataset

Three relational tables from a banking database:

| Table | Description |
|---|---|
| `Cust_Account` | Historical account data — balances, limits, payment history, DPD records |
| `Cust_Enquiry` | Loan enquiry history — purpose, amount, recency |
| `Cust_Demographics` | Customer application data — 79 anonymised demographic features |

---

## ⚙️ Feature Engineering

30+ meaningful features were engineered from raw transactional data:

| Feature | Description |
|---|---|
| `Ratio_currbalance_creditlimit` | Current balance / total credit limit |
| `utilisation_trend` | Credit utilisation ratio over time |
| `acct_ph_ok_mean` | Fraction of on-time payments in payment history |
| `acct_ph_bad_mean` | Fraction of 30+ DPD months in payment history |
| `acct_amt_past_due_max` | Maximum overdue amount across all accounts |
| `count_enquiry_rec_90` | Number of loan enquiries in last 90 days |
| `count_enquiry_rec_365` | Number of loan enquiries in last 365 days |
| `perc_unsecured` | Ratio of unsecured loan enquiries to total enquiries |
| `acct_diff_lastpaymt_opened_mean` | Average months between last payment and account opening |
| `acct_count` | Total number of active accounts |

---

## 🔬 Methodology

```
Raw Data (MySQL DB)
      ↓
Feature Engineering (Account + Enquiry + Demographics)
      ↓
EDA — Bivariate Analysis + Correlation Heatmap
      ↓
Scikit-learn Pipeline (Imputer → Scaler) — No Data Leakage
      ↓
SMOTE — Handle Class Imbalance
      ↓
RandomizedSearchCV — Hyperparameter Tuning
      ↓
5-Fold Cross Validation — Gini Stability Check
      ↓
Ensemble Model (XGBoost + LightGBM + Random Forest)
      ↓
Business Cost Analysis (FP vs FN Trade-off)
      ↓
Live Prediction Interface
```

---

## 🤖 Models Built & Compared

| Model | Gini | AUC |
|---|---|---|
| Logistic Regression | Baseline | — |
| Random Forest | ✅ Above benchmark | — |
| XGBoost (Tuned) | ✅ Above benchmark | — |
| LightGBM | ✅ Above benchmark | — |
| **Ensemble (Best)** | **✅ Best Gini** | **Best** |

> Benchmark Gini = 37.9 — all tree-based models exceeded this.

---

## 📊 Evaluation Highlights

- **Gini Coefficient** — primary metric, compared against 37.9 benchmark
- **ROC-AUC Curve** — visual model comparison
- **Decile Rank Ordering** — industry-standard credit risk validation
- **Confusion Matrix** — TP, FP, FN, TN breakdown
- **5-Fold Cross Validation** — Gini stability across different data splits
- **Business Cost Analysis** — optimal threshold based on FP vs FN financial cost

---

## 💡 Key Business Insight

```
FALSE NEGATIVE (Miss a bad customer) → Bank approves → Customer defaults
→ Direct financial loss of principal + interest → HIGH COST

FALSE POSITIVE (Reject a good customer) → Lost revenue opportunity  
→ MODERATE COST

∴ In banking, FALSE NEGATIVES are 3-5x more costly than FALSE POSITIVES.
  Model threshold is optimised for minimum total business cost,
  not just maximum accuracy.
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/AnchalGada/Credit-Risk-Modelling.git
cd Credit-Risk-Modelling
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Open the notebook**
```bash
jupyter notebook Credit_Risk_Model_BankGoodCredit_v2.ipynb
```

**4. Run all cells**
```
Kernel → Restart & Run All
```

**5. Make a prediction**
```
Go to Step 15 → Change customer values → Press Shift+Enter → Get result
```

---

## 🎯 Live Prediction Output Example

```
===========================================================
   🏦  BANK GOODCREDIT — CREDIT RISK ASSESSMENT
===========================================================
   Bad Label Prediction   : 0  (✅ GOOD CREDIT)
   Probability of Default : 21.4%
   Credit Score           : 732 / 850
   Risk Band              : LOW
   Score  : [████████████████████░░░░░░░░░░]  732/850
            300(Poor) ──────────────── 850(Excellent)
===========================================================
   📋 RECOMMENDATION : ✅ APPROVE — Low default risk
      Action: Safe to approve at requested credit limit.
===========================================================
```

---

## 🗂️ Project Structure

```
Credit-Risk-Modelling/
│
├── Credit_Risk_Model_BankGoodCredit_v2.ipynb   # Main notebook
├── requirements.txt                             # Dependencies
└── README.md                                    # This file
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| ML Models | XGBoost, LightGBM, Random Forest, Logistic Regression |
| ML Framework | Scikit-learn |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Database | MySQL, SQLAlchemy, PyMySQL |
| Tuning | RandomizedSearchCV, StratifiedKFold |
| Environment | Jupyter Notebook / JupyterLab |

---

## 📈 Results

| Metric | Value |
|---|---|
| Benchmark Gini | 37.9 |
| Best Model Gini | Above benchmark ✅ |
| Cross-Val Stability | Low std deviation ✅ |
| Project Grade | **A** ✅ |

---

## 👩‍💻 Author

**Anchal Gada**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat\&logo=linkedin)](https://www.linkedin.com/in/anchal-gada-8357222b3)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat\&logo=github)](https://github.com/AnchalGada)

