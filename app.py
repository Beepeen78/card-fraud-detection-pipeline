import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    page_icon="💳",
)

# ---------- Small CSS polish ----------
st.markdown("""
<style>
/* wider container and nicer font sizes */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.kpi-card {
  border: 1px solid #eaeaea; border-radius: 14px; padding: 16px;
  background: #fff; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.kpi-label { font-size: 0.9rem; color: #6b7280; }
.kpi-value { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
hr { margin: 0.7rem 0 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------- Load model & threshold ----------
pipeline = joblib.load("fraud_pipeline.joblib")
with open("inference_threshold.json", "r") as f:
    t = json.load(f)
threshold = t.get("best_threshold", t.get("threshold", 0.5))

# ---------- Infer expected columns ----------
try:
    expected_cols = list(pipeline.named_steps["prep"].get_feature_names_out())
except Exception:
    expected_cols = getattr(pipeline, "feature_names_in_", [])

if not expected_cols:
    st.error("Could not infer feature names from the pipeline. Re-save with `verbose_feature_names_out=False`.")
    st.stop()

# ---------- Helpers ----------
def make_manual_row(amount: float, hour: int, txn_1h: int) -> pd.DataFrame:
    row = {c: 0.0 for c in expected_cols}
    row["amt"] = float(amount)
    row["hour"] = int(hour)
    row["hour_sin"] = np.sin(2*np.pi*hour/24.0)
    row["hour_cos"] = np.cos(2*np.pi*hour/24.0)
    row["is_night"] = 1.0 if (hour < 6 or hour >= 22) else 0.0
    row["is_business_hours"] = 1.0 if (9 <= hour < 17) else 0.0

    dow = 0  # Monday
    row["dayofweek"] = float(dow)
    row["dow_sin"] = np.sin(2*np.pi*dow/7.0)
    row["dow_cos"] = np.cos(2*np.pi*dow/7.0)
    row["is_weekend"] = 1.0 if dow in (5, 6) else 0.0

    row["txn_count_last_1h"] = float(txn_1h)
    # defaults for velocity/others (safe zeros)
    defaults = [
        "total_amt_last_1h","txn_count_last_24h","total_amt_last_24h",
        "time_since_last_txn","transaction_count","dayofyear","dist_home_merch",
        "mean_amt","std_amt","median_amt","max_amt","mean_distance","lat",
        "long","city_pop","unix_time","merch_lat","merch_long",
        "merch_zipcode","month"
    ]
    for c in defaults:
        row.setdefault(c, 0.0)
    df = pd.DataFrame([row]).reindex(columns=expected_cols, fill_value=0.0)
    return df

def kpi_card(label: str, value: float, fmt: str = ".3f"):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value:{fmt}}</div>
    </div>
    """, unsafe_allow_html=True)

def pr_curve_plot(y_true, y_scores):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, thresh = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR curve"))
    fig.update_layout(
        title=f"Precision–Recall Curve (AP = {ap:.3f})",
        xaxis_title="Recall", yaxis_title="Precision",
        margin=dict(l=10,r=10,t=40,b=10), height=420
    )
    return fig

def score_df(df: pd.DataFrame):
    scored = df.reindex(columns=expected_cols, fill_value=0.0)
    proba = pipeline.predict_proba(scored)[:, 1]
    preds = (proba >= threshold).astype(int)
    out = df.copy()
    out["fraud_probability"] = proba
    out["fraud_prediction"] = preds
    return out, proba, preds

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Model threshold")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, float(threshold), 0.001)
    st.divider()
    st.caption("Upload data to score, or try manual transaction on the Score tab.")

st.title("💳 Credit Card Fraud Detection")

tabs = st.tabs(["Score", "Diagnostics", "Feature Insights"])

# =========================
# TAB 1 — SCORE
# =========================
with tabs[0]:
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("📤 Upload CSV")
        up = st.file_uploader("Upload engineered feature table (.csv)", type=["csv"])
        if up:
            try:
                raw = pd.read_csv(up)
            except UnicodeDecodeError:
                up.seek(0)
                raw = pd.read_csv(up, encoding="latin1")

            out, proba, preds = score_df(raw)

            # KPI row
            st.write("")
            c1, c2, c3 = st.columns(3)
            with c1: kpi_card("Rows Scored", len(out), ".0f")
            with c2: kpi_card("Predicted Fraud (count)", int((out["fraud_prediction"]==1).sum()), ".0f")
            with c3: kpi_card("Mean Fraud Probability", float(out["fraud_probability"].mean()), ".3f")

            st.write("")
            st.dataframe(out.head(25), use_container_width=True)
            st.download_button("⬇️ Download Scored CSV", out.to_csv(index=False), "fraud_predictions.csv")

            # Quick class distribution & prob hist
            st.write("")
            ch1, ch2 = st.columns(2)
            with ch1:
                bar = px.histogram(out, x="fraud_prediction", text_auto=True)
                bar.update_layout(title="Predicted Class Distribution", xaxis_title="Prediction", yaxis_title="Count", height=360, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(bar, use_container_width=True)
            with ch2:
                hist = px.histogram(out, x="fraud_probability", nbins=40)
                hist.update_layout(title="Fraud Probability Histogram", xaxis_title="Probability", yaxis_title="Count", height=360, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(hist, use_container_width=True)

    with right:
        st.subheader("🔍 Manual Transaction (demo)")
        amount = st.number_input("Amount ($)", min_value=0.0, step=1.0, value=1200.0)
        hour   = st.slider("Hour (0–23)", 0, 23, 23)
        txn_1h = st.number_input("Txn Count (last 1h)", min_value=0, step=1, value=3)

        if st.button("Predict Fraud", type="primary"):
            sample = make_manual_row(amount, hour, txn_1h)
            prob = float(pipeline.predict_proba(sample)[:, 1][0])
            label = "Fraud 🚨" if prob >= threshold else "Legit ✅"

            c1, c2 = st.columns([1,1])
            with c1: kpi_card("Probability", prob)
            with c2: kpi_card("Threshold", threshold)

            st.success(f"**Prediction:** {label}")
            st.caption("First 20 features sent to the model")
            st.dataframe(sample.iloc[:, :20], use_container_width=True, height=240)

# =========================
# TAB 2 — DIAGNOSTICS
# =========================
with tabs[1]:
    st.subheader("📊 Diagnostics (needs labeled data)")
    lab_file = st.file_uploader("Upload CSV with **is_fraud** column to view PR/Lift curves", type=["csv"], key="diag")
    if lab_file:
        try:
            lab_raw = pd.read_csv(lab_file)
        except UnicodeDecodeError:
            lab_file.seek(0); lab_raw = pd.read_csv(lab_file, encoding="latin1")

        if "is_fraud" not in lab_raw.columns:
            st.error("The file must include an `is_fraud` column.")
        else:
            out, proba, preds = score_df(lab_raw)
            y_true = lab_raw["is_fraud"].astype(int).values

            # KPIs at current threshold
            from sklearn.metrics import precision_score, recall_score, f1_score
            pr = float(precision_score(y_true, preds, zero_division=0))
            rc = float(recall_score(y_true, preds, zero_division=0))
            f1 = float(f1_score(y_true, preds, zero_division=0))

            c1, c2, c3, c4 = st.columns(4)
            with c1: kpi_card("Rows", len(out), ".0f")
            with c2: kpi_card("Precision", pr)
            with c3: kpi_card("Recall", rc)
            with c4: kpi_card("F1", f1)

            # PR curve
            st.plotly_chart(pr_curve_plot(y_true, proba), use_container_width=True)

            # Confusion matrix bar
            tn = int(((preds==0) & (y_true==0)).sum())
            fp = int(((preds==1) & (y_true==0)).sum())
            fn = int(((preds==0) & (y_true==1)).sum())
            tp = int(((preds==1) & (y_true==1)).sum())
            cm_df = pd.DataFrame({"count":[tn, fp, fn, tp]},
                                 index=["TN","FP","FN","TP"]).reset_index()
            cm_fig = px.bar(cm_df, x="index", y="count", text_auto=True,
                            title="Confusion Matrix (counts)")
            cm_fig.update_layout(height=360, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(cm_fig, use_container_width=True)

# =========================
# TAB 3 — FEATURE INSIGHTS
# =========================
with tabs[2]:
    st.subheader("🧠 Feature Insights")
    st.caption("Top features by permutation importance (approximate)")

    # quick, model-agnostic permutation importance on a tiny random sample
    demo_rows = st.slider("Rows to sample for importance (performance vs. accuracy)", 300, 5000, 1000, 100)
    random_state = 13

    @st.cache_data(show_spinner=True)
    def compute_permutation_importance(n_rows: int, seed: int):
        # Create a synthetic batch with plausible ranges to avoid needing a real CSV
        rng = np.random.default_rng(seed)
        Xsyn = pd.DataFrame(0.0, index=np.arange(n_rows), columns=expected_cols)
        # set a few realistic ranges for known features
        if "amt" in Xsyn: Xsyn["amt"] = rng.uniform(1, 2000, n_rows)
        if "hour" in Xsyn: Xsyn["hour"] = rng.integers(0, 24, n_rows)
        if "txn_count_last_1h" in Xsyn: Xsyn["txn_count_last_1h"] = rng.integers(0, 6, n_rows)
        if "total_amt_last_24h" in Xsyn: Xsyn["total_amt_last_24h"] = rng.uniform(0, 5000, n_rows)
        if "dist_home_merch" in Xsyn: Xsyn["dist_home_merch"] = rng.uniform(0, 500, n_rows)
        # fill cyc features from hour if present
        if "hour_sin" in Xsyn and "hour" in Xsyn:
            Xsyn["hour_sin"] = np.sin(2*np.pi*Xsyn["hour"]/24.0)
        if "hour_cos" in Xsyn and "hour" in Xsyn:
            Xsyn["hour_cos"] = np.cos(2*np.pi*Xsyn["hour"]/24.0)

        base = pipeline.predict_proba(Xsyn)[:,1]
        base_var = np.var(base)
        scores = []
        for c in expected_cols[:60]:  # cap to keep it fast on big models
            Xperm = Xsyn.copy()
            rng.shuffle(Xperm[c].values)
            alt = pipeline.predict_proba(Xperm)[:,1]
            drop = max(base_var - np.var(alt), 0.0)
            scores.append((c, float(drop)))
        imp = pd.DataFrame(scores, columns=["feature","importance"]).sort_values("importance", ascending=False)
        return imp

    imp_df = compute_permutation_importance(demo_rows, random_state)
    st.dataframe(imp_df.head(25), use_container_width=True, height=420)
    fig_imp = px.bar(imp_df.head(25), x="importance", y="feature", orientation="h")
    fig_imp.update_layout(title="Top Features (Permutation Importance, proxy)",
                          height=700, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

st.divider()
st.caption("Tip: Use the sidebar to change threshold and re-evaluate precision/recall instantly.")
