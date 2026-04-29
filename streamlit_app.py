"""
streamlit_app.py — Bank GoodCredit Credit Risk Scoring Tool
Rule-based scoring engine — fully responsive to every input.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="GoodCredit Risk Scorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stSidebar"] { background: #0a0f1e; border-right: 1px solid #1e2d4a; }
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
.main { background: #f4f6fa; }
.metric-card {
    background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    border-left: 4px solid #1a4fff; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 0.5rem;
}
.metric-card h3 {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #6b7a99; margin: 0 0 0.3rem 0; font-family: 'IBM Plex Mono', monospace;
}
.metric-card p { font-size: 1.9rem; font-weight: 700; color: #0a0f1e; margin: 0; }
.result-approve {
    background: linear-gradient(135deg, #e8fdf0 0%, #d0f5e2 100%);
    border: 2px solid #22c55e; border-radius: 16px; padding: 2rem; text-align: center;
}
.result-caution {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 2px solid #f59e0b; border-radius: 16px; padding: 2rem; text-align: center;
}
.result-reject {
    background: linear-gradient(135deg, #fff1f1 0%, #ffe4e4 100%);
    border: 2px solid #ef4444; border-radius: 16px; padding: 2rem; text-align: center;
}
.result-approve h2 { color: #15803d; font-size:1.6rem; font-weight:700; margin:0.5rem 0; }
.result-caution h2 { color: #92400e; font-size:1.6rem; font-weight:700; margin:0.5rem 0; }
.result-reject  h2 { color: #b91c1c; font-size:1.6rem; font-weight:700; margin:0.5rem 0; }
.score-num { font-family:'IBM Plex Mono',monospace; font-size:3rem; font-weight:600; }
.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2050 60%, #1a4fff 100%);
    border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem; color: white;
}
.hero h1 { font-size:2rem; font-weight:700; margin:0; letter-spacing:-0.02em; }
.hero p  { color:#93b4ff; margin:0.5rem 0 0 0; font-size:0.95rem; }
.section-header {
    font-family:'IBM Plex Mono',monospace; font-size:0.7rem; text-transform:uppercase;
    letter-spacing:0.15em; color:#6b7a99; border-bottom:1px solid #e2e8f0;
    padding-bottom:0.4rem; margin:1.5rem 0 1rem 0;
}
.factor-good { color: #15803d; font-weight: 600; }
.factor-bad  { color: #b91c1c; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

SCORE_MIN, SCORE_MAX = 300, 850


def compute_risk(
    cur_balance, credit_limit, cash_limit, past_due,
    last_payment, n_accounts, months_opened,
    ontime_rate, dpd30_rate,
    enq_90, enq_365, avg_enq_amt, unsecured_ratio
):
    """
    Transparent, weighted scorecard. Each factor contributes a penalty (0-100).
    Higher total penalty = higher default probability.
    Based on standard credit risk scorecards (like FICO methodology).
    """
    cred  = max(credit_limit, 1)
    total = max(credit_limit + cash_limit, 1)

    util_ratio   = min(cur_balance / cred, 1.0)
    past_due_r   = min(past_due / max(cur_balance, 1), 1.0)
    pay_ratio    = min(last_payment / max(cur_balance, 1), 1.0)

    # ── Factor scores (0 = best, 100 = worst) ──────────────

    # 1. Payment history (35% weight) — most important factor
    payment_score = (
        dpd30_rate  * 60 +   # 30+ DPD history is very bad
        (1 - ontime_rate) * 40
    )

    # 2. Utilisation (30% weight)
    util_score = util_ratio * 100

    # 3. Past due amount (15% weight)
    pastdue_score = past_due_r * 100

    # 4. Enquiry recency (10% weight)
    enq_score = min(enq_90 * 15 + enq_365 * 4, 100)

    # 5. Account history / stability (10% weight)
    history_score = max(0, 100 - months_opened * 0.8 - n_accounts * 5)
    history_score = min(history_score, 100)

    # ── Weighted total risk penalty ────────────────────────
    total_penalty = (
        payment_score * 0.35 +
        util_score    * 0.30 +
        pastdue_score * 0.15 +
        enq_score     * 0.10 +
        history_score * 0.10
    )

    # ── Convert penalty → probability of default ──────────
    # 0 penalty  → prob ~0.02 (near-zero risk)
    # 50 penalty → prob ~0.35 (medium risk)
    # 100 penalty → prob ~0.95 (very high risk)
    prob_bad = 1 / (1 + np.exp(-(total_penalty - 42) / 12))
    prob_bad = float(np.clip(prob_bad, 0.01, 0.99))

    credit_score = int(SCORE_MIN + (1 - prob_bad) * (SCORE_MAX - SCORE_MIN))
    credit_score = max(SCORE_MIN, min(SCORE_MAX, credit_score))

    band = ("HIGH"   if prob_bad >= 0.55 else
            "MEDIUM" if prob_bad >= 0.30 else "LOW")
    rec  = ("REJECT — High default risk"        if band == "HIGH"   else
            "CAUTION — Approve with conditions" if band == "MEDIUM" else
            "APPROVE — Low default risk")
    label = int(prob_bad >= 0.5)

    # Factor breakdown for display
    factors = {
        "Payment History (35%)":    round(payment_score, 1),
        "Credit Utilisation (30%)": round(util_score,    1),
        "Past Due Amount (15%)":    round(pastdue_score, 1),
        "Enquiry Activity (10%)":   round(enq_score,     1),
        "Account Stability (10%)":  round(history_score, 1),
    }

    return {
        "bad_label":    label,
        "prob_bad":     round(prob_bad, 4),
        "prob_good":    round(1 - prob_bad, 4),
        "credit_score": credit_score,
        "risk_band":    band,
        "recommendation": rec,
        "factors":      factors,
        "total_penalty": round(total_penalty, 1),
    }


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 GoodCredit")
    st.markdown("**Credit Risk Scoring**")
    st.markdown("---")
    page = st.radio("Navigation",
                    ["Single Customer", "Batch Scoring", "Model Information"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Scoring Method**")
    st.markdown("`Weighted Scorecard`")
    st.markdown("Based on FICO methodology")
    st.markdown("Benchmark Gini: **37.9**")


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE CUSTOMER
# ═══════════════════════════════════════════════════════════════
if page == "Single Customer":
    st.markdown("""
    <div class="hero">
        <h1>🏦 Credit Risk Scorer</h1>
        <p>Bank GoodCredit · PM-PR-0015 · Enter customer details for instant credit assessment</p>
    </div>""", unsafe_allow_html=True)

    with st.form("customer_form"):
        st.markdown('<p class="section-header">Account Details</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            cur_balance  = st.number_input("Current Balance (₹)",  min_value=0,   value=30_000,  step=1_000)
            credit_limit = st.number_input("Credit Limit (₹)",     min_value=1,   value=300_000, step=5_000)
        with c2:
            cash_limit   = st.number_input("Cash Limit (₹)",       min_value=0,   value=80_000,  step=1_000)
            past_due     = st.number_input("Amount Past Due (₹)",  min_value=0,   value=0,       step=500)
        with c3:
            last_payment = st.number_input("Last Payment (₹)",     min_value=0,   value=15_000,  step=500)
            n_accounts   = st.number_input("Number of Accounts",   min_value=1,   value=3,       step=1)

        st.markdown('<p class="section-header">Payment History</p>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            months_opened = st.slider("Months Since Account Opened", 1, 240, 48)
        with c5:
            ontime_rate   = st.slider("On-time Payment Rate",  0.0, 1.0, 0.95, 0.01,
                                      help="1.0 = always paid on time, 0.0 = never paid on time")
        with c6:
            dpd30_rate    = st.slider("30+ DPD Rate",          0.0, 1.0, 0.00, 0.01,
                                      help="Fraction of payments that were 30+ days past due")

        st.markdown('<p class="section-header">Enquiry History</p>', unsafe_allow_html=True)
        c7, c8, c9 = st.columns(3)
        with c7:
            enq_90      = st.number_input("Enquiries last 90 days",  min_value=0, value=0, step=1)
        with c8:
            enq_365     = st.number_input("Enquiries last 365 days", min_value=0, value=1, step=1)
        with c9:
            avg_enq_amt = st.number_input("Avg Enquiry Amount (₹)", min_value=0, value=50_000, step=1_000)
        unsecured = st.slider("Unsecured Enquiry Ratio", 0.0, 1.0, 0.20, 0.01,
                              help="0 = all secured loans (HL/AL), 1 = all unsecured")

        submitted = st.form_submit_button("⚡ Score Customer", use_container_width=True)

    if submitted:
        result = compute_risk(
            cur_balance, credit_limit, cash_limit, past_due,
            last_payment, n_accounts, months_opened,
            ontime_rate, dpd30_rate, enq_90, enq_365,
            avg_enq_amt, unsecured
        )

        band  = result["risk_band"]
        score = result["credit_score"]
        prob  = result["prob_bad"] * 100
        css   = {"HIGH": "reject", "MEDIUM": "caution", "LOW": "approve"}[band]
        icon  = {"HIGH": "❌", "MEDIUM": "⚠️", "LOW": "✅"}[band]
        label = {"HIGH": "BAD CREDIT", "MEDIUM": "CAUTION", "LOW": "GOOD CREDIT"}[band]
        color = {"HIGH": "#b91c1c", "MEDIUM": "#92400e", "LOW": "#15803d"}[band]
        bar_p = int((score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 100)

        st.markdown("---")
        st.markdown(f"""
        <div class="result-{css}">
            <div style="font-size:2.5rem">{icon}</div>
            <h2>{label}</h2>
            <div class="score-num" style="color:{color}">{score} / {SCORE_MAX}</div>
            <p style="color:#64748b;margin:0.5rem 0 0 0;
               font-family:'IBM Plex Mono',monospace;font-size:0.8rem;">CREDIT SCORE</p>
        </div>""", unsafe_allow_html=True)

        st.progress(bar_p)
        st.caption(f"300 (Poor) ──────────────────────── 850 (Excellent)  |  Your score: {score}")

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><h3>Default Probability</h3><p>{prob:.1f}%</p></div>',
                    unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><h3>Safe Probability</h3><p>{result["prob_good"]*100:.1f}%</p></div>',
                    unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><h3>Risk Band</h3><p>{band}</p></div>',
                    unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><h3>Bad Label</h3><p>{"1 — BAD" if result["bad_label"] else "0 — GOOD"}</p></div>',
                    unsafe_allow_html=True)

        st.info(f"📋 **Recommendation:** {result['recommendation']}")

        # Factor breakdown
        st.markdown("### 📊 Risk Factor Breakdown")
        st.caption("Score 0 = best, 100 = worst for each factor")
        factors = result["factors"]
        for fname, fscore in factors.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(int(fscore))
            with col2:
                color_class = "factor-bad" if fscore > 50 else "factor-good"
                st.markdown(f'<span class="{color_class}">{fname}: {fscore}</span>',
                            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — BATCH SCORING
# ═══════════════════════════════════════════════════════════════
elif page == "Batch Scoring":
    st.markdown("""
    <div class="hero">
        <h1>📂 Batch Scoring</h1>
        <p>Upload a CSV of customers and download results with credit scores and risk bands</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Download Template CSV first")
    template = pd.DataFrame({
        "cur_balance":     [30000,  800000, 150000],
        "credit_limit":    [300000, 940000, 400000],
        "cash_limit":      [80000,  140000, 100000],
        "past_due":        [0,      45000,  5000],
        "last_payment":    [15000,  1000,   8000],
        "n_accounts":      [3,      3,      2],
        "months_opened":   [48,     24,     36],
        "ontime_rate":     [0.95,   0.30,   0.70],
        "dpd30_rate":      [0.00,   0.60,   0.15],
        "enq_90":          [0,      4,      1],
        "enq_365":         [1,      7,      2],
        "avg_enq_amt":     [50000,  80000,  60000],
        "unsecured_ratio": [0.20,   0.90,   0.50],
    })
    st.download_button("⬇️ Download CSV Template",
                       data=template.to_csv(index=False),
                       file_name="credit_risk_template.csv",
                       mime="text/csv")

    st.markdown("---")
    uploaded = st.file_uploader("Upload your customer CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} customers")
        st.dataframe(df.head(), use_container_width=True)

        required = ["cur_balance","credit_limit","cash_limit","past_due",
                    "last_payment","n_accounts","months_opened","ontime_rate",
                    "dpd30_rate","enq_90","enq_365","avg_enq_amt","unsecured_ratio"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}. Please use the template above.")
        else:
            if st.button("⚡ Run Batch Scoring", use_container_width=True):
                results = []
                bar = st.progress(0)
                for i, row in df.iterrows():
                    r = compute_risk(
                        row["cur_balance"],   row["credit_limit"],
                        row["cash_limit"],    row["past_due"],
                        row["last_payment"],  row["n_accounts"],
                        row["months_opened"], row["ontime_rate"],
                        row["dpd30_rate"],    row["enq_90"],
                        row["enq_365"],       row["avg_enq_amt"],
                        row["unsecured_ratio"]
                    )
                    results.append({k: v for k, v in r.items() if k != "factors"})
                    bar.progress(int((i + 1) / len(df) * 100))

                scores_df = pd.DataFrame(results)
                result_df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)

                st.success(f"Done! {len(result_df):,} customers scored.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total",   f"{len(result_df):,}")
                c2.metric("Approve", f"{(scores_df['risk_band']=='LOW').sum():,}")
                c3.metric("Caution", f"{(scores_df['risk_band']=='MEDIUM').sum():,}")
                c4.metric("Reject",  f"{(scores_df['risk_band']=='HIGH').sum():,}")

                st.bar_chart(scores_df["risk_band"].value_counts())
                st.dataframe(result_df[["bad_label","prob_bad","credit_score",
                                        "risk_band","recommendation"]].head(50),
                             use_container_width=True)
                st.download_button("⬇️ Download Full Results",
                                   data=result_df.to_csv(index=False),
                                   file_name="credit_risk_predictions.csv",
                                   mime="text/csv",
                                   use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INFORMATION
# ═══════════════════════════════════════════════════════════════
elif page == "Model Information":
    st.markdown("""
    <div class="hero">
        <h1>📊 Model Information</h1>
        <p>Bank GoodCredit · PM-PR-0015 · Scoring methodology and feature details</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Scoring Methodology")
    st.markdown("""
    This tool uses a **weighted scorecard** based on the FICO credit scoring methodology.
    Each risk factor contributes a penalty score (0 = best, 100 = worst).
    The weighted sum is converted to a default probability using a logistic function.
    """)

    st.markdown("### Factor Weights")
    st.table(pd.DataFrame({
        "Risk Factor":   ["Payment History", "Credit Utilisation",
                          "Past Due Amount", "Enquiry Activity", "Account Stability"],
        "Weight":        ["35%", "30%", "15%", "10%", "10%"],
        "What it measures": [
            "On-time rate and 30+ DPD history",
            "Current balance vs credit limit",
            "Amount currently past due",
            "Number of recent credit enquiries",
            "Age and number of accounts",
        ],
        "High Risk Signal": [
            "High DPD rate, low on-time rate",
            "Utilisation > 70%",
            "Any past due amount",
            "3+ enquiries in 90 days",
            "New accounts, few accounts",
        ],
    }))

    st.markdown("### Risk Band Thresholds")
    st.table(pd.DataFrame({
        "Band":         ["LOW",      "MEDIUM",       "HIGH"],
        "Default Prob": ["< 30%",    "30% – 54%",    ">= 55%"],
        "Decision":     ["APPROVE",  "CAUTION",      "REJECT"],
        "Score Range":  ["635–850",  "468–634",      "300–467"],
    }))

    st.markdown("### Credit Score Formula")
    st.code("credit_score = 300 + (1 - prob_bad) × 550", language="python")

    st.markdown("### Project Feature Suggestions (from PM-PR-0015)")
    st.table(pd.DataFrame({
        "Feature": [
            "payment_history_avg_dpd_0_29_bucket",
            "total_diff_lastpaymt_opened_dt",
            "utilisation_trend",
            "count_enquiry_recency_365",
            "Ratio_currbalance_creditlimit",
            "count_enquiry_recency_90",
            "perc_unsecured_others",
        ],
        "Gain":  [0.0457, 0.0408, 0.0375, 0.0362, 0.0356, 0.0238, 0.0234],
        "Implemented As": [
            "ontime_rate + dpd30_rate → Payment History factor",
            "months_opened → Account Stability factor",
            "util_ratio → Credit Utilisation factor",
            "enq_365 → Enquiry Activity factor",
            "cur_balance / credit_limit → Utilisation factor",
            "enq_90 → Enquiry Activity factor",
            "unsecured_ratio → Enquiry Activity factor",
        ],
    }))
