"""
predict.py — Bank GoodCredit Credit Risk Predictor
====================================================
Loads saved artifacts and scores a single customer or a batch.

CLI usage:
    python predict.py                         # interactive single customer
    python predict.py --batch input.csv       # batch from CSV

Import usage (from app.py):
    from predict import CreditRiskPredictor
    predictor = CreditRiskPredictor()
    result    = predictor.predict(customer_dict)
"""

import argparse
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

BAND_HIGH   = 0.60
BAND_MEDIUM = 0.35
SCORE_MIN   = 300
SCORE_MAX   = 850


class CreditRiskPredictor:
    """Load artifacts once, score repeatedly."""

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self._dir = artifacts_dir
        self._load_artifacts()

    def _load_artifacts(self):
        with open(os.path.join(self._dir, "best_credit_risk_model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(self._dir, "imputer.pkl"), "rb") as f:
            self.imputer = pickle.load(f)
        with open(os.path.join(self._dir, "feature_columns.pkl"), "rb") as f:
            self.feature_columns = pickle.load(f)
        with open(os.path.join(self._dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self._dir, "model_metadata.json")) as f:
            self.metadata = json.load(f)

    def predict(self, customer: dict) -> dict:
        """
        Score a single customer.

        Parameters
        ----------
        customer : dict
            Keys should match feature_columns. Missing keys are imputed.

        Returns
        -------
        dict with keys: bad_label, prob_bad, prob_good, credit_score,
                        risk_band, recommendation, model_name
        """
        row = pd.DataFrame([customer])
        return self._score_dataframe(row).iloc[0].to_dict()

    def predict_batch(self, customers: list) -> pd.DataFrame:
        """Score a list of customer dicts. Returns a DataFrame."""
        df = pd.DataFrame(customers)
        return self._score_dataframe(df)

    def predict_csv(self, path: str) -> pd.DataFrame:
        """Score a CSV file. Returns original data with results appended."""
        df = pd.read_csv(path)
        scores = self._score_dataframe(df)
        return pd.concat([df.reset_index(drop=True), scores], axis=1)

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        X_imp = pd.DataFrame(
            self.imputer.transform(df), columns=self.feature_columns
        )
        if self.metadata.get("needs_scaling", False):
            return self.scaler.transform(X_imp)
        return X_imp.values

    def _score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        X      = self._preprocess(df.copy())
        probs  = self.model.predict_proba(X)[:, 1]
        labels = (probs >= 0.5).astype(int)
        scores = (SCORE_MIN + (1 - probs) * (SCORE_MAX - SCORE_MIN)).astype(int)
        bands  = np.where(probs >= BAND_HIGH,   "HIGH",
                 np.where(probs >= BAND_MEDIUM, "MEDIUM", "LOW"))
        recs   = np.where(bands == "HIGH",   "REJECT — High default risk",
                 np.where(bands == "MEDIUM", "CAUTION — Approve with conditions",
                                             "APPROVE — Low default risk"))
        return pd.DataFrame({
            "bad_label":      labels,
            "prob_bad":       probs.round(4),
            "prob_good":      (1 - probs).round(4),
            "credit_score":   scores,
            "risk_band":      bands,
            "recommendation": recs,
        })


def _build_sample_input():
    """Return a sample low-risk customer dict."""
    return {
        "Ratio_currbalance_creditlimit":    0.10,
        "utilisation_trend":               0.10,
        "acct_cur_balance_amt_sum":        90_000,
        "acct_cur_balance_amt_mean":       30_000,
        "acct_creditlimit_sum":           900_000,
        "acct_creditlimit_mean":          300_000,
        "acct_cashlimit_sum":             240_000,
        "acct_cashlimit_mean":             80_000,
        "acct_amt_past_due_sum":                0,
        "acct_amt_past_due_mean":               0,
        "acct_amt_past_due_max":                0,
        "acct_actualpaymentamount_sum":    45_000,
        "acct_actualpaymentamount_mean":   15_000,
        "acct_rateofinterest_mean":         0.14,
        "acct_util_ratio_mean":             0.10,
        "acct_util_ratio_max":              0.11,
        "acct_diff_lastpaymt_opened_sum":     144,
        "acct_diff_lastpaymt_opened_mean":     48,
        "acct_count":                           3,
        "acct_ph_ok_mean":                   0.95,
        "acct_ph_bad_mean":                  0.00,
        "count_enquiry_total":                  1,
        "count_enquiry_rec_90":                 0,
        "count_enquiry_rec_365":                1,
        "mean_enq_amt":                   50_000,
        "max_enq_amt":                    75_000,
        "perc_unsecured":                   0.20,
    }


def _print_result(result: dict):
    sep   = "=" * 55
    label = "BAD CREDIT"  if result["bad_label"] == 1 else "GOOD CREDIT"
    band  = result["risk_band"]
    score = result["credit_score"]
    bar_f = int((score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 30)
    print(sep)
    print("   BANK GOODCREDIT - CREDIT RISK RESULT")
    print(sep)
    print(f"   Bad Label Prediction   : {result['bad_label']}  ({label})")
    print(f"   Probability of Default : {result['prob_bad']*100:.1f}%")
    print(f"   Credit Score           : {score} / {SCORE_MAX}")
    print(f"   Risk Band              : {band}")
    print(f"   Score Bar  : [{'X'*bar_f}{'.'*(30-bar_f)}]  {score}/{SCORE_MAX}")
    print(sep)
    print(f"   RECOMMENDATION : {result['recommendation']}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Bank GoodCredit — credit risk prediction"
    )
    parser.add_argument("--batch",  metavar="CSV", help="CSV file for batch scoring")
    parser.add_argument("--output", metavar="OUT", default="predictions.csv",
                        help="Output CSV for batch mode (default: predictions.csv)")
    args = parser.parse_args()

    predictor = CreditRiskPredictor()
    print(f"Model loaded : {predictor.metadata['model_name']}")
    print(f"Gini: {predictor.metadata['gini']:.2f}  |  "
          f"AUC: {predictor.metadata['auc']:.4f}  |  "
          f"Beats benchmark: {predictor.metadata['beats_benchmark']}")

    if args.batch:
        print(f"\nBatch scoring: {args.batch}")
        out = predictor.predict_csv(args.batch)
        out.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}  ({len(out)} rows)")
    else:
        print("\n--- Single customer demo (low-risk sample) ---")
        result = predictor.predict(_build_sample_input())
        _print_result(result)


if __name__ == "__main__":
    main()
