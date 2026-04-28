"""
streamlit_app.py — Bank GoodCredit Credit Risk Scoring Tool
============================================================
Run:  streamlit run streamlit_app.py
"""

import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="GoodCredit Risk Scorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0f1e;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* Main background */
.main { background: #f4f6fa; }

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #1a4fff;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 0.5rem;
}
.metric-card h3 {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7a99;
    margin: 0 0 0.3rem 0;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-card p {
    font-size: 1.9rem;
    font-weight: 700;
    color: #0a0f1e;
    margin: 0;
}

/* Result panel */
.result-approve {
    background: linear-gradient(135deg, #e8fdf0 0%, #d0f5e2 100%);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-caution {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 2px solid #f59e0b;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-reject {
    background: linear-gradient(135deg, #fff1f1 0%, #ffe4e4 100%);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-approve h2, .result-caution h2, .result-reject h2 {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.result-approve h2 { color: #15803d; }
.result-caution h2 { color: #92400e; }
.result-reject h2  { color: #b91c1c; }

.score-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3rem;
    font-weight: 600;
}
.score-label-approve { color: #15803d; }
.score-label-caution  { color: #92400e; }
.score-label-reject   { color: #b91c1c; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #6b7a99;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2050 60%, #1a4fff 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero p {
    color: #93b4ff;
    margin: 0.5rem 0 0 0;
    font-size: 0.95rem;
}

/* Sticker badges */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-weight: 600;
    margin-right: 0.4rem;
}
.badge-blue   { background: #dbeafe; color: #1d4ed8; }
.badge-green  { background: #dcfce7; color: #15803d; }
.badge-yellow { background: #fef9c3; color: #854d0e; }
.badge-red    { background: #fee2e2; color: #b91c1c; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS  (cached so they load only once)
# ─────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
SCORE_MIN, SCORE_MAX = 300, 850
BAND_HIGH, BAND_MEDIUM = 0.60, 0.35


@st.cache_resource
def load_artifacts():
    """Load model artifacts once and cache them for the session."""
    try:
        model    = pickle.load(open(os.path.join(ARTIFACTS_DIR, "best_credit_risk_model.pkl"), "rb"))
        imputer  = pickle.load(open(os.path.join(ARTIFACTS_DIR, "imputer.pkl"),               "rb"))
        scaler   = pickle.load(open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"),                "rb"))
        features = pickle.load(open(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"),       "rb"))
        metadata = json.load(open(os.path.join(ARTIFACTS_DIR, "model_metadata.json")))
        return model, imputer, scaler, features, metadata, None
    except FileNotFoundError as e:
        return None, None, None, None, None, str(e)


model, imputer, scaler, feature_columns, metadata, load_error = load_artifacts()


# ─────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> np.ndarray:

    # 🔹 Step 1: Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0   # default value

    # 🔹 Step 2: Ensure correct column order
    df = df[feature_columns]

    # 🔹 Step 3: Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)

    # 🔹 Step 4: Apply trained imputer
    X_imp = pd.DataFrame(imputer.transform(df), columns=feature_columns)

    # 🔹 Step 5: Apply scaling if required
    if metadata.get("needs_scaling", False):
        return scaler.transform(X_imp)

    return X_imp.values

def score_customer(customer: dict) -> dict:
    row   = pd.DataFrame([customer])
    X     = preprocess(row)
    prob  = float(model.predict_proba(X)[0, 1])
    label = int(prob >= 0.5)
    score = int(SCORE_MIN + (1 - prob) * (SCORE_MAX - SCORE_MIN))
    band  = "HIGH" if prob >= BAND_HIGH else ("MEDIUM" if prob >= BAND_MEDIUM else "LOW")
    rec   = ("REJECT — High default risk"            if band == "HIGH"   else
             "CAUTION — Approve with conditions"     if band == "MEDIUM" else
             "APPROVE — Low default risk")
    return {"bad_label": label, "prob_bad": round(prob, 4),
            "prob_good": round(1 - prob, 4), "credit_score": score,
            "risk_band": band, "recommendation": rec}


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    X      = preprocess(df.copy())
    probs  = model.predict_proba(X)[:, 1]
    labels = (probs >= 0.5).astype(int)
    scores = (SCORE_MIN + (1 - probs) * (SCORE_MAX - SCORE_MIN)).astype(int)
    bands  = np.where(probs >= BAND_HIGH,   "HIGH",
             np.where(probs >= BAND_MEDIUM, "MEDIUM", "LOW"))
    recs   = np.where(bands == "HIGH",   "REJECT",
             np.where(bands == "MEDIUM", "CAUTION", "APPROVE"))
    return pd.DataFrame({
        "bad_label":      labels,
        "prob_bad":       probs.round(4),
        "prob_good":      (1 - probs).round(4),
        "credit_score":   scores,
        "risk_band":      bands,
        "recommendation": recs,
    })


# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 GoodCredit")
    st.markdown("**Credit Risk Scoring**")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Single Customer", "Batch Scoring", "Model Information"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if metadata:
        st.markdown(f"**Model**")
        st.markdown(f"`{metadata['model_name']}`")
        beat = "✅ Yes" if metadata["beats_benchmark"] else "❌ No"
        st.markdown(f"Gini: **{metadata['gini']:.1f}** (bench 37.9) {beat}")
        st.markdown(f"AUC: **{metadata['auc']:.4f}**")
    else:
        st.warning("Artifacts not loaded.\nRun `python train.py` first.")

# ─────────────────────────────────────────────────────────────
# LOAD ERROR GUARD
# ─────────────────────────────────────────────────────────────
if load_error:
    st.error(f"**Model artifacts not found.**\n\n"
             f"`{load_error}`\n\n"
             f"Run `python train.py` in the project folder first, then restart.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# PAGE 1 — SINGLE CUSTOMER
# ─────────────────────────────────────────────────────────────
if page == "Single Customer":

    st.markdown("""
    <div class="hero">
        <h1>🏦 Credit Risk Scorer</h1>
        <p>Bank GoodCredit · PM-PR-0015 · Enter customer details to get an instant credit assessment</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("customer_form"):

        # ── Account Details ──────────────────────────────────
        st.markdown('<p class="section-header">Account Details</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            cur_balance   = st.number_input("Current Balance (₹)", min_value=0, value=30_000, step=1_000)
            credit_limit  = st.number_input("Credit Limit (₹)",    min_value=1, value=300_000, step=5_000)
        with c2:
            cash_limit    = st.number_input("Cash Limit (₹)",      min_value=0, value=80_000,  step=1_000)
            past_due      = st.number_input("Amount Past Due (₹)", min_value=0, value=0,        step=500)
        with c3:
            last_payment  = st.number_input("Last Payment (₹)",    min_value=0, value=15_000,  step=500)
            n_accounts    = st.number_input("Number of Accounts",  min_value=1, value=3,        step=1)

        # ── Payment History ──────────────────────────────────
        st.markdown('<p class="section-header">Payment History</p>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            months_since_opened = st.slider("Months Since Account Opened", 1, 240, 48)
        with c5:
            ontime_rate  = st.slider("On-time Payment Rate (0–1)", 0.0, 1.0, 0.95, 0.01)
        with c6:
            dpd30_rate   = st.slider("30+ DPD Rate (0–1)",         0.0, 1.0, 0.00, 0.01)

        # ── Enquiry Data ─────────────────────────────────────
        st.markdown('<p class="section-header">Enquiry History</p>', unsafe_allow_html=True)
        c7, c8, c9 = st.columns(3)
        with c7:
            enq_90        = st.number_input("Enquiries — last 90 days",  min_value=0, value=0, step=1)
        with c8:
            enq_365       = st.number_input("Enquiries — last 365 days", min_value=0, value=1, step=1)
        with c9:
            avg_enq_amt   = st.number_input("Avg Enquiry Amount (₹)",   min_value=0, value=50_000, step=1_000)
        unsecured_ratio   = st.slider("Unsecured Enquiry Ratio (0=all secured, 1=all unsecured)",
                                      0.0, 1.0, 0.20, 0.01)

        submitted = st.form_submit_button("⚡ Score Customer", use_container_width=True)

    # ── Result ───────────────────────────────────────────────
    if submitted:
        cred = credit_limit if credit_limit > 0 else 1
        customer = {
            "Ratio_currbalance_creditlimit":    cur_balance / cred,
            "utilisation_trend":               cur_balance / cred,
            "acct_cur_balance_amt_sum":        cur_balance * n_accounts,
            "acct_cur_balance_amt_mean":       cur_balance,
            "acct_creditlimit_sum":            credit_limit * n_accounts,
            "acct_creditlimit_mean":           credit_limit,
            "acct_cashlimit_sum":              cash_limit * n_accounts,
            "acct_cashlimit_mean":             cash_limit,
            "acct_amt_past_due_sum":           past_due * n_accounts,
            "acct_amt_past_due_mean":          past_due,
            "acct_amt_past_due_max":           past_due,
            "acct_actualpaymentamount_sum":    last_payment * n_accounts,
            "acct_actualpaymentamount_mean":   last_payment,
            "acct_rateofinterest_mean":        0.14,
            "acct_util_ratio_mean":            cur_balance / cred,
            "acct_util_ratio_max":             min(cur_balance / cred * 1.1, 1.0),
            "acct_diff_lastpaymt_opened_sum":  months_since_opened * n_accounts,
            "acct_diff_lastpaymt_opened_mean": months_since_opened,
            "acct_count":                      n_accounts,
            "acct_ph_ok_mean":                 ontime_rate,
            "acct_ph_bad_mean":                dpd30_rate,
            "count_enquiry_total":             enq_365,
            "count_enquiry_rec_90":            enq_90,
            "count_enquiry_rec_365":           enq_365,
            "mean_enq_amt":                    avg_enq_amt,
            "max_enq_amt":                     avg_enq_amt * 1.5,
            "perc_unsecured":                  unsecured_ratio,
        }

        with st.spinner("Scoring..."):
            result = score_customer(customer)

        band  = result["risk_band"]
        score = result["credit_score"]
        prob  = result["prob_bad"] * 100
        css   = {"HIGH": "reject", "MEDIUM": "caution", "LOW": "approve"}[band]
        icon  = {"HIGH": "❌", "MEDIUM": "⚠️", "LOW": "✅"}[band]
        label = {"HIGH": "BAD CREDIT", "MEDIUM": "CAUTION", "LOW": "GOOD CREDIT"}[band]

        st.markdown("---")
        # Main result card
        bar_pct = int((score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 100)
        st.markdown(f"""
        <div class="result-{css}">
            <div style="font-size:2.5rem">{icon}</div>
            <h2>{label}</h2>
            <div class="score-label score-label-{css}">{score} / {SCORE_MAX}</div>
            <p style="color:#64748b; margin:0.5rem 0 0 0; font-family:'IBM Plex Mono',monospace;
                      font-size:0.8rem;">CREDIT SCORE</p>
        </div>
        """, unsafe_allow_html=True)

        # Score bar
        st.progress(bar_pct)
        st.caption(f"300 (Poor) {'─'*30} 850 (Excellent)  |  Your score: {score}")

        # Metrics row
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"""<div class="metric-card">
            <h3>Default Probability</h3><p>{prob:.1f}%</p></div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="metric-card">
            <h3>Safe Probability</h3><p>{result['prob_good']*100:.1f}%</p></div>""", unsafe_allow_html=True)
        m3.markdown(f"""<div class="metric-card">
            <h3>Risk Band</h3><p>{band}</p></div>""", unsafe_allow_html=True)
        m4.markdown(f"""<div class="metric-card">
            <h3>Bad Label</h3><p>{'1 — BAD' if result['bad_label'] else '0 — GOOD'}</p></div>""",
            unsafe_allow_html=True)

        st.info(f"📋 **Recommendation:** {result['recommendation']}")


# ─────────────────────────────────────────────────────────────
# PAGE 2 — BATCH SCORING
# ─────────────────────────────────────────────────────────────
elif page == "Batch Scoring":

    st.markdown("""
    <div class="hero">
        <h1>📂 Batch Scoring</h1>
        <p>Upload a CSV of customers and download results with credit scores and risk bands</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How to prepare your CSV")
    st.markdown(
        "Your CSV should include as many of the feature columns as possible. "
        "Any missing columns are **automatically imputed** using the training median. "
        "You can download a template below."
    )

    # Template download
    sample_data = {
        "Ratio_currbalance_creditlimit":    [0.10, 0.85, 0.45],
        "utilisation_trend":               [0.10, 0.82, 0.44],
        "acct_cur_balance_amt_sum":        [90000, 800000, 300000],
        "acct_cur_balance_amt_mean":       [30000, 266666, 100000],
        "acct_creditlimit_sum":            [900000, 940000, 670000],
        "acct_creditlimit_mean":           [300000, 313333, 223333],
        "acct_cashlimit_sum":              [240000, 140000, 190000],
        "acct_cashlimit_mean":             [80000, 46666,  63333],
        "acct_amt_past_due_sum":           [0, 45000, 5000],
        "acct_amt_past_due_mean":          [0, 15000, 1666],
        "acct_amt_past_due_max":           [0, 20000, 3000],
        "acct_actualpaymentamount_sum":    [45000, 5000, 20000],
        "acct_actualpaymentamount_mean":   [15000, 1666, 6666],
        "acct_rateofinterest_mean":        [0.14,  0.38, 0.22],
        "acct_util_ratio_mean":            [0.10,  0.82, 0.44],
        "acct_util_ratio_max":             [0.11,  0.90, 0.50],
        "acct_diff_lastpaymt_opened_sum":  [144, 72, 108],
        "acct_diff_lastpaymt_opened_mean": [48,  24, 36],
        "acct_count":                      [3, 3, 3],
        "acct_ph_ok_mean":                 [0.95, 0.30, 0.70],
        "acct_ph_bad_mean":                [0.00, 0.60, 0.15],
        "count_enquiry_total":             [1, 8, 3],
        "count_enquiry_rec_90":            [0, 4, 1],
        "count_enquiry_rec_365":           [1, 7, 2],
        "mean_enq_amt":                    [50000, 80000, 60000],
        "max_enq_amt":                     [75000, 120000, 90000],
        "perc_unsecured":                  [0.20, 0.90, 0.50],
    }
    template_df = pd.DataFrame(sample_data)
    st.download_button(
        "⬇️  Download CSV Template",
        data=template_df.to_csv(index=False),
        file_name="credit_risk_template.csv",
        mime="text/csv",
    )

    st.markdown("---")
    uploaded = st.file_uploader("Upload your customer CSV", type=["csv"])

    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_input):,} customers — {df_input.shape[1]} columns")
        st.dataframe(df_input.head(5), use_container_width=True)

        if st.button("⚡ Run Batch Scoring", use_container_width=True):
            with st.spinner(f"Scoring {len(df_input):,} customers..."):
                scores = score_dataframe(df_input)
                result_df = pd.concat([df_input.reset_index(drop=True), scores], axis=1)

            st.success(f"Done! {len(result_df):,} customers scored.")

            # Summary stats
            st.markdown("### Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Customers",  f"{len(result_df):,}")
            c2.metric("Approvals (LOW)",   f"{(scores['risk_band']=='LOW').sum():,}")
            c3.metric("Cautions (MEDIUM)", f"{(scores['risk_band']=='MEDIUM').sum():,}")
            c4.metric("Rejects (HIGH)",    f"{(scores['risk_band']=='HIGH').sum():,}")

            # Band breakdown
            band_counts = scores["risk_band"].value_counts()
            st.bar_chart(band_counts)

            # Score distribution
            st.markdown("### Credit Score Distribution")
            hist_data = pd.DataFrame({"Credit Score": scores["credit_score"]})
            st.bar_chart(hist_data["Credit Score"].value_counts().sort_index())

            # Preview
            st.markdown("### Results Preview (first 50 rows)")
            display_cols = ["bad_label", "prob_bad", "credit_score", "risk_band", "recommendation"]
            st.dataframe(result_df[display_cols].head(50), use_container_width=True)

            # Download
            csv_out = result_df.to_csv(index=False)
            st.download_button(
                "⬇️  Download Full Results CSV",
                data=csv_out,
                file_name="credit_risk_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────
# PAGE 3 — MODEL INFORMATION
# ─────────────────────────────────────────────────────────────
elif page == "Model Information":

    st.markdown("""
    <div class="hero">
        <h1>📊 Model Information</h1>
        <p>Bank GoodCredit · PM-PR-0015 · Model performance, features, and scoring logic</p>
    </div>
    """, unsafe_allow_html=True)

    # Model summary cards
    st.markdown("### Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="metric-card">
        <h3>Best Model</h3>
        <p style="font-size:1rem">{metadata['model_name']}</p></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card">
        <h3>Gini Score</h3><p>{metadata['gini']:.2f}</p></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card">
        <h3>AUC Score</h3><p>{metadata['auc']:.4f}</p></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card">
        <h3>Beats Benchmark</h3>
        <p>{'✅ Yes' if metadata['beats_benchmark'] else '❌ No'}</p></div>""",
        unsafe_allow_html=True)

    st.markdown("---")
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("### All Metrics")
        metrics_data = {
            "Metric": ["Gini", "AUC", "Accuracy", "Precision", "Recall", "F1"],
            "Value":  [
                f"{metadata['gini']:.2f}",
                f"{metadata['auc']:.4f}",
                f"{metadata['accuracy']*100:.2f}%",
                f"{metadata['precision']*100:.2f}%",
                f"{metadata['recall']*100:.2f}%",
                f"{metadata['f1']*100:.2f}%",
            ],
        }
        st.table(pd.DataFrame(metrics_data))

    with c_right:
        st.markdown("### Risk Band Thresholds")
        thresholds = pd.DataFrame({
            "Band":            ["LOW",         "MEDIUM",          "HIGH"],
            "Default Prob":    ["< 35%",        "35% – 59%",        ">= 60%"],
            "Decision":        ["APPROVE",      "CAUTION",         "REJECT"],
            "Credit Score":    ["~630–850",     "~468–629",        "300–467"],
        })
        st.table(thresholds)

    st.markdown("---")
    st.markdown("### Credit Score Scale")
    st.markdown("""
    The credit score is a linear mapping of the model's default probability:

    | Default Prob | Credit Score | Risk |
    |---|---|---|
    | 0% (safest) | 850 | Excellent |
    | 35% | 658 | Fair |
    | 60% | 465 | Poor |
    | 100% (riskiest) | 300 | Very Poor |

    **Formula:** `credit_score = 300 + (1 - prob_bad) × 550`
    """)

    st.markdown("---")
    st.markdown("### Project Feature Suggestions (from problem statement)")
    feature_table = pd.DataFrame({
        "Feature": [
            "payment_history_avg_dpd_0_29_bucket",
            "total_diff_lastpaymt_opened_dt",
            "min_months_last_30_plus",
            "utilisation_trend",
            "count_enquiry_recency_365",
            "Ratio_currbalance_creditlimit",
            "mean_diff_lastpaymt_opened_dt",
            "count_enquiry_recency_90",
            "perc_unsecured_others",
        ],
        "Gain": [0.0457, 0.0408, 0.0382, 0.0375, 0.0362, 0.0356, 0.0322, 0.0238, 0.0234],
        "Description": [
            "Mean accounts in 0-29 DPD bucket throughout payment history",
            "Total duration: last payment date → account opened date",
            "Smallest months before first 30+ DPD appeared",
            "[total cur_bal / total credit limit] / [mean cur_bal / (mean credit limit + cash limit)]",
            "# of enquiries in past 365 days",
            "Total current balance / total credit limit",
            "Average duration: last payment date → account opened date",
            "# of enquiries in past 90 days",
            "Ratio of secured loan enquiry purpose to total enquiries",
        ],
    })
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Feature Columns Used by Model")
    st.caption(f"Total: {metadata['feature_count']} features")
    cols = metadata["feature_columns"]
    col_chunks = [cols[i:i+4] for i in range(0, len(cols), 4)]
    for chunk in col_chunks:
        badges = " ".join([f'<span class="badge badge-blue">{c}</span>' for c in chunk])
        st.markdown(badges, unsafe_allow_html=True)
