"""
train.py — Bank GoodCredit Credit Risk Model
=============================================
Trains all models (LR, RF, XGBoost, LightGBM, Ensemble), selects the best
by Gini, and saves the following artifacts:
  - best_credit_risk_model.pkl
  - imputer.pkl
  - scaler.pkl
  - feature_columns.pkl
  - model_metadata.json

Run:
    python train.py                        # uses synthetic data
    python train.py --use-db               # connects to real DB
"""

import argparse
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import lightgbm as lgb

# ── Reference date (matches notebook) ────────────────────────
REF = pd.Timestamp("2018-06-01")

# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_from_db():
    from sqlalchemy import create_engine
    import pymysql  # noqa
    engine = create_engine(
        "mysql+pymysql://dm_team1:DM!$Team&279@20!@18.136.157.135:3306/project_banking",
        connect_args={"connect_timeout": 10},
    )
    df_account      = pd.read_sql("SELECT * FROM Cust_Account",      engine)
    df_enquiry      = pd.read_sql("SELECT * FROM Cust_Enquiry",       engine)
    df_demographics = pd.read_sql("SELECT * FROM Cust_Demographics",  engine)
    return df_account, df_enquiry, df_demographics


def load_synthetic():
    np.random.seed(42)
    N = 5000
    customer_nos = np.arange(1, N + 1)

    df_demographics = pd.DataFrame({
        "customer_no": customer_nos,
        **{f"feature_{i}": np.random.randn(N) for i in range(1, 80)},
        "Bad_label": np.random.choice([0, 1], size=N, p=[0.82, 0.18]),
    })
    bad = df_demographics["Bad_label"] == 1
    for c in ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]:
        df_demographics.loc[bad, c] += 1.2

    n_acc = N * 3
    ac = np.random.choice(customer_nos, n_acc)
    df_account = pd.DataFrame({
        "customer_no":          ac,
        "opened_dt":            pd.date_range("2010-01-01", periods=n_acc, freq="4h"),
        "last_paymt_dt":        pd.date_range("2017-01-01", periods=n_acc, freq="3h"),
        "acct_type":            np.random.choice(["CC", "PL", "HL", "AL"], n_acc),
        "cur_balance_amt":      np.random.uniform(0, 300_000, n_acc),
        "creditlimit":          np.random.uniform(20_000, 1_000_000, n_acc),
        "cashlimit":            np.random.uniform(5_000, 200_000, n_acc),
        "high_credit_amt":      np.random.uniform(10_000, 500_000, n_acc),
        "amt_past_due":         np.where(np.random.rand(n_acc) > 0.8,
                                         np.random.uniform(100, 50_000, n_acc), 0),
        "actualpaymentamount":  np.random.uniform(0, 50_000, n_acc),
        "rateofinterest":       np.random.uniform(0.10, 0.42, n_acc),
        "paymenthistory1":      ["".join(np.random.choice(list("0123X"), 24))
                                 for _ in range(n_acc)],
    })

    n_enq = N * 5
    ec = np.random.choice(customer_nos, n_enq)
    df_enquiry = pd.DataFrame({
        "customer_no": ec,
        "enquiry_dt":  pd.date_range("2016-01-01", periods=n_enq, freq="1h"),
        "enq_purpose": np.random.choice(["CC", "PL", "HL", "AL", "SL", "GL"], n_enq),
        "enq_amt":     np.random.uniform(5_000, 500_000, n_enq),
    })

    return df_account, df_enquiry, df_demographics


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def parse_payment_history(s):
    """Return (on-time fraction, bad-payment fraction) from payment history string."""
    if not isinstance(s, str):
        return 0.0, 0.0
    nums = [int(c) for c in s if c.isdigit()]
    if not nums:
        return 0.0, 0.0
    return (sum(1 for x in nums if x < 2) / len(nums),
            sum(1 for x in nums if x >= 2) / len(nums))


def build_account_features(df_account):
    df = df_account.copy()
    df["opened_dt"]     = pd.to_datetime(df["opened_dt"],     errors="coerce")
    df["last_paymt_dt"] = pd.to_datetime(df["last_paymt_dt"], errors="coerce")
    df["diff_lastpaymt_opened"] = (
        (df["last_paymt_dt"] - df["opened_dt"]).dt.days / 30
    ).clip(0)
    df["util_ratio"] = (
        df["cur_balance_amt"] / df["creditlimit"].replace(0, np.nan)
    ).clip(0, 5)

    if "paymenthistory1" in df.columns:
        ph = df["paymenthistory1"].apply(
            lambda x: pd.Series(parse_payment_history(x), index=["ph_ok", "ph_bad"])
        )
        df = pd.concat([df, ph], axis=1)

    agg_dict = {
        "diff_lastpaymt_opened":   ["sum", "mean"],
        "util_ratio":              ["mean", "max"],
        "cur_balance_amt":         ["sum", "mean"],
        "creditlimit":             ["sum", "mean"],
        "cashlimit":               ["sum", "mean"],
        "amt_past_due":            ["sum", "mean", "max"],
        "actualpaymentamount":     ["sum", "mean"],
    }
    if "ph_ok" in df.columns:
        agg_dict["ph_ok"]  = "mean"
        agg_dict["ph_bad"] = "mean"

    acct_agg = df.groupby("customer_no").agg(agg_dict)
    acct_agg.columns = ["acct_" + "_".join(c) for c in acct_agg.columns]
    acct_agg = acct_agg.reset_index()

    acct_cnt = df.groupby("customer_no").size().reset_index(name="acct_count")
    acct_agg = acct_agg.merge(acct_cnt, on="customer_no")

    acct_agg["Ratio_currbalance_creditlimit"] = (
        acct_agg.get("acct_cur_balance_amt_sum", pd.Series(0)) /
        acct_agg.get("acct_creditlimit_sum", pd.Series(1)).replace(0, np.nan)
    ).clip(0, 5)
    acct_agg["utilisation_trend"] = acct_agg.get(
        "acct_util_ratio_mean", pd.Series(0)
    ).clip(0, 10)

    return acct_agg


def build_enquiry_features(df_enquiry):
    df = df_enquiry.copy()
    df["enquiry_dt"] = pd.to_datetime(df["enquiry_dt"], errors="coerce")
    df["days_since"] = (REF - df["enquiry_dt"]).dt.days.clip(0)
    df["in_90"]  = (df["days_since"] <= 90).astype(int)
    df["in_365"] = (df["days_since"] <= 365).astype(int)

    if "enq_purpose" in df.columns:
        df["is_secured"] = df["enq_purpose"].isin(["HL", "AL"]).astype(int)

    enq_agg = df.groupby("customer_no").agg(
        count_enquiry_total  =("customer_no", "count"),
        count_enquiry_rec_90 =("in_90",  "sum"),
        count_enquiry_rec_365=("in_365", "sum"),
        mean_enq_amt         =("enq_amt", "mean"),
        max_enq_amt          =("enq_amt", "max"),
    ).reset_index()

    if "is_secured" in df.columns:
        pu = df.groupby("customer_no")["is_secured"].mean().reset_index()
        pu.columns = ["customer_no", "perc_unsecured"]
        enq_agg = enq_agg.merge(pu, on="customer_no", how="left")

    return enq_agg


def build_feature_matrix(df_account, df_enquiry, df_demographics):
    acct_agg = build_account_features(df_account)
    enq_agg  = build_enquiry_features(df_enquiry)

    df = df_demographics.copy()
    if "customer no" in df.columns:
        df.rename(columns={"customer no": "customer_no"}, inplace=True)

    df = df.merge(acct_agg, on="customer_no", how="left")
    df = df.merge(enq_agg,  on="customer_no", how="left")

    drop_cols = [c for c in ["customer_no", "dt_opened", "entry_time"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    X = df.drop(columns=["Bad_label"])
    y = df["Bad_label"]

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    high_miss = X.columns[X.isnull().mean() > 0.6].tolist()
    X.drop(columns=high_miss, inplace=True)

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X_imp, y, imputer, list(X_imp.columns)


# ─────────────────────────────────────────────────────────────
# 3. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate(name, model, X_te, y_te):
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = model.predict(X_te)
    auc  = roc_auc_score(y_te, y_prob)
    gini = (2 * auc - 1) * 100
    return {
        "name":      name,
        "auc":       round(auc, 4),
        "gini":      round(gini, 2),
        "accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_te, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_te, y_pred,        zero_division=0), 4),
        "model":     model,
        "needs_scaling": False,
    }


def train_all_models(X_tr, y_tr, X_te, y_te, scaler):
    results = []

    # ── Logistic Regression ──────────────────────────────────
    print("  Training Logistic Regression ...")
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_test_sc = scaler.transform(X_te)

    class _LRWrapper:
        def __init__(self, m): self.m = m
        def predict_proba(self, X): return self.m.predict_proba(X)
        def predict(self, X):       return self.m.predict(X)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42, n_jobs=-1)
    lr.fit(X_tr_sc, y_tr)
    res = evaluate("Logistic Regression", _LRWrapper(lr), X_test_sc, y_te)
    res["needs_scaling"] = True
    res["model_obj"]     = lr          # raw model for persistence
    results.append(res)

    # ── Random Forest ────────────────────────────────────────
    print("  Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    res = evaluate("Random Forest", rf, X_te, y_te)
    res["model_obj"] = rf
    results.append(res)

    # ── XGBoost ──────────────────────────────────────────────
    print("  Training XGBoost ...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
        eval_metric="auc", random_state=42, n_jobs=-1,
    )
    xgb.fit(X_tr, y_tr)
    res = evaluate("XGBoost", xgb, X_te, y_te)
    res["model_obj"] = xgb
    results.append(res)

    # ── LightGBM ─────────────────────────────────────────────
    print("  Training LightGBM ...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        num_leaves=63, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_tr, y_tr)
    res = evaluate("LightGBM", lgbm, X_te, y_te)
    res["model_obj"] = lgbm
    results.append(res)

    # ── Ensemble ─────────────────────────────────────────────
    print("  Training Ensemble (XGB + LGBM + RF) ...")
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
        voting="soft",
    )
    ensemble.fit(X_tr, y_tr)
    res = evaluate("Ensemble (XGB+LGBM+RF)", ensemble, X_te, y_te)
    res["model_obj"] = ensemble
    results.append(res)

    return results


# ─────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────

def main(use_db: bool = False):
    print("=" * 60)
    print("  Bank GoodCredit — Credit Risk Model Training")
    print("=" * 60)

    # Load data
    if use_db:
        print("\n[1/5] Connecting to database ...")
        try:
            df_account, df_enquiry, df_demographics = load_from_db()
            print("  ✅ Connected to real database.")
        except Exception as e:
            print(f"  ⚠️  DB failed ({str(e)[:60]}). Falling back to synthetic data.")
            df_account, df_enquiry, df_demographics = load_synthetic()
    else:
        print("\n[1/5] Loading synthetic data ...")
        df_account, df_enquiry, df_demographics = load_synthetic()
    print(f"  Demographics: {df_demographics.shape} | "
          f"Account: {df_account.shape} | "
          f"Enquiry: {df_enquiry.shape}")

    # Feature engineering
    print("\n[2/5] Feature engineering ...")
    X, y, imputer, feature_columns = build_feature_matrix(
        df_account, df_enquiry, df_demographics
    )
    print(f"  Feature matrix: {X.shape} | Bad rate: {y.mean()*100:.1f}%")

    # Train/test split + SMOTE
    print("\n[3/5] Train/test split & SMOTE ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_tr, y_tr = smote.fit_resample(X_train, y_train)
    print(f"  Train (after SMOTE): {X_tr.shape} | Test: {X_test.shape}")

    # Train models
    print("\n[4/5] Training models ...")
    scaler  = StandardScaler()
    results = train_all_models(X_tr, y_tr, X_test, y_test, scaler)

    # Compare & select best
    print("\n[5/5] Model comparison:")
    print(f"  {'Model':<30} {'Gini':>8} {'AUC':>8} {'F1':>8}  Beats Benchmark?")
    print("  " + "-" * 70)
    for r in sorted(results, key=lambda x: x["gini"], reverse=True):
        beat = "✅ YES" if r["gini"] > 37.9 else "❌ NO"
        print(f"  {r['name']:<30} {r['gini']:>8.2f} {r['auc']:>8.4f} "
              f"{r['f1']:>8.4f}  {beat}")

    best = max(results, key=lambda x: x["gini"])
    print(f"\n  🏆 Best model : {best['name']}")
    print(f"     Gini       : {best['gini']:.2f}  (Benchmark: 37.9)")
    print(f"     AUC        : {best['auc']:.4f}")

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    pickle.dump(best["model_obj"],  open("artifacts/best_credit_risk_model.pkl", "wb"))
    pickle.dump(imputer,            open("artifacts/imputer.pkl",                "wb"))
    pickle.dump(feature_columns,    open("artifacts/feature_columns.pkl",        "wb"))
    pickle.dump(scaler,             open("artifacts/scaler.pkl",                 "wb"))

    metadata = {
        "model_name":    best["name"],
        "needs_scaling": best["needs_scaling"],
        "gini":          best["gini"],
        "auc":           best["auc"],
        "accuracy":      best["accuracy"],
        "precision":     best["precision"],
        "recall":        best["recall"],
        "f1":            best["f1"],
        "benchmark_gini": 37.9,
        "beats_benchmark": best["gini"] > 37.9,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }
    with open("artifacts/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n  ✅ Artifacts saved to ./artifacts/")
    print("     - best_credit_risk_model.pkl")
    print("     - imputer.pkl")
    print("     - scaler.pkl")
    print("     - feature_columns.pkl")
    print("     - model_metadata.json")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bank GoodCredit credit risk model")
    parser.add_argument("--use-db", action="store_true",
                        help="Connect to the real MySQL database instead of synthetic data")
    args = parser.parse_args()
    main(use_db=args.use_db)
