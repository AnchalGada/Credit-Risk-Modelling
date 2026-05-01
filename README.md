# 🏦 Credit Risk Scoring Model — Bank GoodCredit

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-92.68_Gini-orange?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-92.89_Gini-green?style=for-the-badge)
![Ensemble](https://img.shields.io/badge/Ensemble-93.03_Gini-gold?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

## [Click Here to Try the Live App](https://credit-risk-model-sa6lrta9dysc9kdwsqegqx.streamlit.app)

*End-to-end credit risk model — data loading → EDA → feature engineering → 5 ML models → evaluation → live scoring app.*

</div>

---

## 📌 Project Overview

**Client:** Bank GoodCredit &nbsp;|&nbsp; **Category:** Banking — Risk &nbsp;

Bank GoodCredit wants to predict a credit score for current credit card customers. The score denotes creditworthiness and helps the bank reduce credit default risk.

| Item | Detail |
|---|---|
| **Target Variable** | `Bad_label` — 0 = Good Credit, 1 = Bad Credit (30 DPD+) |
| **Bad Rate in Data** | 17.8% (891 bad out of 5000 customers) |
| **Benchmark Gini** | 37.9 |
| **Best Model** | Ensemble (XGB + LightGBM + RF) — Gini **93.03** |
| **Benchmark Improvement** | +55.13 Gini points above benchmark |
| **Data Sources** | 3 MySQL tables — Cust_Account, Cust_Enquiry, Cust_Demographics |
| **Total Features Engineered** | 105 features |

---

## 🚀 Live Demo

👉 **[Open the Credit Risk Scoring App](https://credit-risk-model-sa6lrta9dysc9kdwsqegqx.streamlit.app)**

Enter customer details and get an instant credit score, risk band, and recommendation.

> **Note:** The Streamlit app uses a weighted scorecard to demonstrate the scoring logic interactively, as the production MySQL database is not publicly accessible from Streamlit Cloud. The full ML pipeline with all 5 models is in the Jupyter notebook.

---

## 📊 Model Results — From Jupyter Notebook

All 5 models were trained on 5000 customers with 105 features. Every model beat the benchmark Gini of 37.9.

| Model | AUC | Gini | Accuracy | Precision | Recall | F1 | Beats Benchmark |
|---|---|---|---|---|---|---|---|
| 🏆 **Ensemble (XGB+LGBM+RF)** | **0.9652** | **93.03** | **93.40%** | **85.00%** | **76.40%** | **80.47%** | ✅ YES |
| LightGBM | 0.9645 | 92.89 | 93.80% | 86.25% | 77.53% | 81.66% | ✅ YES |
| XGBoost | 0.9634 | 92.68 | 92.70% | 81.82% | 75.84% | 78.72% | ✅ YES |
| Random Forest | 0.9594 | 91.88 | 92.90% | 83.65% | 74.72% | 78.93% | ✅ YES |
| Logistic Regression | 0.9587 | 91.75 | 91.60% | 74.23% | 80.90% | 77.42% | ✅ YES |

**Best Model automatically selected:** `Ensemble (XGB+LGBM+RF)` with Gini **93.03**

---

## 🎯 Sample Prediction Output (Notebook Step 11)

```
=======================================================
   🏦  BANK GOODCREDIT — CREDIT RISK RESULT
=======================================================
   Model Used             : Ensemble (XGB+LGBM+RF)
   Bad Label Prediction   : 0  (✅ GOOD CREDIT)
   Probability of Default : 1.5%
   Credit Score           : 841 / 850
   Risk Band              : LOW
=======================================================
   Score Bar  : [█████████████████████████████░]  841/850
               300(Poor)          850(Excellent)
=======================================================
   📋 RECOMMENDATION : ✅ APPROVE — Low default risk
=======================================================
```

**Credit Score Formula:** `credit_score = 300 + (1 - prob_bad) × 550`

---

## 📈 Rank Ordering — Decile Analysis (Notebook Step 10)

Decile analysis confirms the model correctly separates risky customers. Decile 10 captures **53.37%** of all bad customers.

| Decile | Total | Bads | Goods | Bad Rate | Cumulative Bad Capture |
|---|---|---|---|---|---|
| **10 (Highest Risk)** | 100 | 95 | 5 | **95.0%** | 53.37% |
| 9 | 100 | 58 | 42 | 58.0% | 85.96% |
| 8 | 100 | 15 | 85 | 15.0% | 94.38% |
| 7 | 100 | 4 | 96 | 4.0% | 96.63% |
| 6 | 100 | 4 | 96 | 4.0% | 98.88% |
| 5 | 100 | 1 | 99 | 1.0% | 99.44% |
| 4 | 100 | 1 | 99 | 1.0% | 100.00% |
| 3 | 100 | 0 | 100 | 0.0% | 100.00% |
| 2 | 100 | 0 | 100 | 0.0% | 100.00% |
| **1 (Lowest Risk)** | 100 | 0 | 100 | **0.0%** | 100.00% |

✅ **Good rank ordering confirmed** — bad rate decreases consistently from decile 10 → 1.

---

##  Notebook Steps

| Step | Description | Key Output |
|---|---|---|
| Step 1 | Install libraries | All packages installed |
| Step 2 | Import libraries | pandas, numpy, sklearn, xgboost, lightgbm, imbalanced-learn |
| Step 3 | Load data | 5000 customers, 15000 account rows, 25000 enquiry rows |
| Step 4 | EDA — distributions, account types, enquiry purposes | Target: 82% Good, 18% Bad |
| Step 5 | Feature engineering from 3 tables | 105 features built |
| Step 6 | Train/test split + SMOTE | 6574 rows after SMOTE (50/50 balanced) |
| Step 7 | Train & evaluate all 5 models individually | Confusion matrices, ROC curves, classification reports |
| Step 8 | Model comparison + best model selection | Ensemble wins with Gini 93.03 |
| Step 9 | Save best model to pickle files | best_credit_risk_model.pkl, imputer.pkl |
| Step 10 | Rank ordering / decile analysis | Decile 10 captures 53.37% of all bads |
| Step 11 | Predict for a new customer | Change inputs, get credit score instantly |

---

## 🔧 Feature Engineering (Notebook Step 5)

Features engineered from 3 source tables, merged at customer level. Reference date: `2018-06-01`.

### From Cust_Account Table
| Feature | Description |
|---|---|
| `Ratio_currbalance_creditlimit` | Total current balance / total credit limit |
| `utilisation_trend` | Mean utilisation ratio across all accounts |
| `acct_ph_ok_mean` | Fraction of on-time payments (parsed from `paymenthistory1` string) |
| `acct_ph_bad_mean` | Fraction of 30+ DPD payments (parsed from `paymenthistory1` string) |
| `acct_diff_lastpaymt_opened_mean` | Avg months between last payment and account open date |
| `acct_amt_past_due_sum/mean/max` | Past due amount aggregations across accounts |
| `acct_util_ratio_mean/max` | Utilisation ratio aggregations |
| `acct_actualpaymentamount_sum/mean` | Payment amount aggregations |
| `acct_count` | Number of accounts per customer |

### From Cust_Enquiry Table
| Feature | Description |
|---|---|
| `count_enquiry_rec_90` | Number of credit enquiries in last 90 days |
| `count_enquiry_rec_365` | Number of credit enquiries in last 365 days |
| `count_enquiry_total` | Total enquiry count |
| `mean_enq_amt / max_enq_amt` | Enquiry amount statistics |
| `perc_unsecured` | Ratio of unsecured enquiry purposes (non-HL/AL) |

### From Cust_Demographics Table
- 79 anonymised customer features (`feature_1` to `feature_79`)
- Features 1–5 show significant separation between good and bad customers

---

## ⚖️ Class Imbalance — SMOTE (Notebook Step 6)

The dataset had a bad rate of only 17.8%. SMOTE oversampling was applied to balance the training set.

| Split | Rows | Features | Bad Rate |
|---|---|---|---|
| Original Train | 4000 | 105 | 17.8% |
| **After SMOTE** | **6574** | **105** | **50.0%** |
| Test Set | 1000 | 105 | 17.8% |

```
SMOTE resampled class counts:
  Good (0): 3287
  Bad  (1): 3287
```

---

## 🏗️ Model Hyperparameters (Notebook Step 7)

```python
# Logistic Regression
LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

# Random Forest
RandomForestClassifier(n_estimators=200, max_depth=10,
                       class_weight="balanced", random_state=42)

# XGBoost
XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8,
              scale_pos_weight=ratio, eval_metric="auc", random_state=42)

# LightGBM
LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
               num_leaves=63, class_weight="balanced", random_state=42)

# Ensemble — Soft Voting
VotingClassifier(estimators=[("xgb",xgb),("lgbm",lgbm),("rf",rf)],
                 voting="soft")
```

---

## 📁 Data Sources

```
MySQL Database: project_banking (Host: 18.136.157.135, Port: 3306)

├── Cust_Account      → opened_dt, last_paymt_dt, cur_balance_amt,
│                       creditlimit, cashlimit, amt_past_due,
│                       paymenthistory1, rateofinterest, actualpaymentamount
│
├── Cust_Enquiry      → enquiry_dt, enq_purpose, enq_amt
│
└── Cust_Demographics → feature_1 to feature_79, Bad_label (target)
```

> The notebook attempts real DB connection first. If unreachable, it automatically falls back to synthetic data with `np.random.seed(42)`.

---

## 🗂️ Project Structure

```
Credit-Risk-Model/
├── Credit_Risk_Model.ipynb   ← Full pipeline: Steps 1–11
├── streamlit_app.py          ← Live scoring app
├── requirements.txt          ← Python dependencies
├── .streamlit/
│   └── config.toml           ← App theme
└── artifacts/                ← Saved model files (from Step 9)
    ├── best_credit_risk_model.pkl
    ├── imputer.pkl
    ├── scaler.pkl
    ├── feature_columns.pkl
    └── model_metadata.json
```

---

##  How to Run the Notebook Locally

### 1. Clone the repo
```bash
git clone https://github.com/AnchalGada/Credit-Risk-Model.git
cd Credit-Risk-Model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Open the notebook
```bash
jupyter notebook Credit_Risk_Model.ipynb
```

Run **Kernel → Restart & Run All**

### 4. Change customer values in Step 11
Edit the values at the top of the last cell and press **Shift+Enter** to get a new credit score.

---

## 🏗️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Best Model** | Ensemble (XGBoost + LightGBM + Random Forest — Soft Voting) |
| **Other Models** | Logistic Regression, Random Forest, XGBoost, LightGBM |
| **Imbalance Handling** | SMOTE (imbalanced-learn) |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualisation** | Matplotlib, Seaborn |
| **Deployment** | Streamlit, Streamlit Community Cloud |
| **Database** | MySQL (SQLAlchemy + PyMySQL) |
| **Serialisation** | Pickle |

---

## 👩‍💻 Author

**Anchal Gada**
GitHub: [@AnchalGada](https://github.com/AnchalGada)
