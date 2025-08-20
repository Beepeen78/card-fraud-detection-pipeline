import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- optional plotting (matplotlib) ---
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# ---- Load pipeline and threshold (with fallback key) ----
pipeline = joblib.load("fraud_pipeline.joblib")
with open("inference_threshold.json", "r") as f:
    t = json.load(f)
threshold = t.get("best_threshold", t.get("threshold", 0.5))

st.title("💳 Credit Card Fraud Detection App")
st.caption(f"Deployed threshold = **{threshold:.3f}**")

# ---- Determine expected feature columns from the pipeline ----
expected_cols = []
try:
    expected_cols = list(pipeline.named_steps["prep"].get_feature_names_out())
except Exception:
    expected_cols = getattr(pipeline, "feature_names_in_", [])

if not expected_cols:
    st.error("Could not infer feature names from the pipeline. "
             "Re-save the pipeline with verbose_feature_names_out=False, "
             "or include a feature list file.")
    st.stop()

# ---- Helpers ----
def make_manual_row(amount: float, hour: int, txn_1h: int) -> pd.DataFrame:
    row = {c: 0.0 for c in expected_cols}
    # Core
    row["amt"] = float(amount)
    row["hour"] = int(hour)

    # Time-derived
    row["hour_sin"] = np.sin(2*np.pi*hour/24.0)
    row["hour_cos"] = np.cos(2*np.pi*hour/24.0)
    row["is_night"] = 1.0 if (hour < 6 or hour >= 22) else 0.0
    row["is_business_hours"] = 1.0 if (9 <= hour < 17) else 0.0

    dow = 0  # Monday (demo)
    row["dayofweek"] = float(dow)
    row["dow_sin"] = np.sin(2*np.pi*dow/7.0)
    row["dow_cos"] = np.cos(2*np.pi*dow/7.0)
    row["is_weekend"] = 1.0 if dow in (5, 6) else 0.0

    # Velocity (demo defaults)
    row["txn_count_last_1h"] = float(txn_1h)
    row.setdefault("total_amt_last_1h", 0.0)
    row.setdefault("txn_count_last_24h", 0.0)
    row.setdefault("total_amt_last_24h", 0.0)
    row.setdefault("time_since_last_txn", 0.0)
    row.setdefault("transaction_count", 0.0)

    # Other engineered defaults
    defaults = {
        "dayofyear": 1.0, "dist_home_merch": 0.0, "mean_amt": 0.0, "std_amt": 0.0,
        "median_amt": 0.0, "max_amt": 0.0, "mean_distance": 0.0, "lat": 0.0,
        "long": 0.0, "city_pop": 0.0, "unix_time": 0.0, "merch_lat": 0.0,
        "merch_long": 0.0, "merch_zipcode": 0.0, "month": 1.0
    }
    for k, v in defaults.items():
        row.setdefault(k, v)

    df = pd.DataFrame([row])
    return df.reindex(columns=expected_cols, fill_value=0.0)

def precision_recall_at_thresholds(y_true, y_score, thresholds):
    """Vectorized prec/rec for a list of thresholds; y_true can be absent (returns None)."""
    if y_true is None:
        return None
    y_true = np.asarray(y_true).astype(int)
    out = []
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out.append((thr, prec, rec))
    return pd.DataFrame(out, columns=["threshold", "precision", "recall"])

def lift_recall_curve(y_true, y_score):
    """Compute recall captured vs. top-k% of transactions ranked by score."""
    if y_true is None:
        return None
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_score)  # descending
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    total_pos = y_true.sum()
    n = len(y_true)
    k = np.arange(1, n+1)
    recall = cum_tp / total_pos if total_pos > 0 else np.zeros_like(cum_tp, dtype=float)
    frac = k / n
    return pd.DataFrame({"top_fraction": frac, "recall": recall})

def top_feature_importances(pipeline, topn=15):
    try:
        clf = pipeline.named_steps["clf"]
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return None
        feat_names = pipeline.named_steps["prep"].get_feature_names_out()
        df_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
        df_imp = df_imp.sort_values("importance", ascending=False).head(topn)
        return df_imp
    except Exception:
        return None

# =========================
# CSV UPLOAD + SCORING
# =========================
st.subheader("📤 Upload a CSV")
uploaded_file = st.file_uploader(
    "Upload CSV with the engineered feature columns (same schema as training)",
    type=["csv"]
)

y_true_available = False
y_true = None
proba = None
out = None

if uploaded_file:
    try:
        raw = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file, encoding="latin1")

    # Try to capture true labels if present
    if "is_fraud" in raw.columns:
        y_true = raw["is_fraud"].values
        y_true_available = True

    # Align/Order columns as the model expects; fill missing with 0
    df = raw.reindex(columns=expected_cols, fill_value=0.0)

    proba = pipeline.predict_proba(df)[:, 1]
    preds = (proba >= threshold).astype(int)

    out = raw.copy()
    out["fraud_probability"] = proba
    out["fraud_prediction"] = preds

    left, right = st.columns([1, 1])
    with left:
        st.success(f"Scored {len(out):,} rows.")
        st.dataframe(out.head(25), use_container_width=True)
        st.download_button("⬇️ Download Results", out.to_csv(index=False), "fraud_predictions.csv")

    # ---------- Charts & diagnostics ----------
    with right:
        st.markdown("### 📊 Diagnostics")

        if HAVE_MPL:
            # 1) Probability histogram
            fig, ax = plt.subplots()
            ax.hist(proba, bins=40)
            ax.axvline(threshold, linestyle="--")
            ax.set_title("Fraud Probability Distribution")
            ax.set_xlabel("Probability")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("Install matplotlib to view charts: `pip install matplotlib`")

        # 2) Lift / recall@top-k
        if y_true_available and proba is not None and HAVE_MPL:
            curve = lift_recall_curve(y_true, proba)
            fig2, ax2 = plt.subplots()
            ax2.plot(curve["top_fraction"] * 100.0, curve["recall"])
            ax2.set_xlabel("Top % of transactions by score")
            ax2.set_ylabel("Recall")
            ax2.set_title("Recall captured vs. Top-% ranked by risk")
            st.pyplot(fig2)

        # 3) Precision/Recall at thresholds
        if y_true_available and proba is not None:
            thr_list = np.round(np.linspace(0.05, 0.95, 10), 3)
            prt = precision_recall_at_thresholds(y_true, proba, thr_list)
            st.dataframe(prt.style.format({"threshold": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"}),
                         use_container_width=True)

# Feature importances (global)
st.write("---")
st.subheader("🧠 Model Feature Importances")
imp = top_feature_importances(pipeline, topn=15)
if imp is None:
    st.info("Feature importances not available for the loaded model.")
else:
    if HAVE_MPL:
        fig3, ax3 = plt.subplots()
        ax3.barh(imp["feature"][::-1], imp["importance"][::-1])
        ax3.set_title("Top Feature Importances")
        ax3.set_xlabel("Importance")
        st.pyplot(fig3)
    st.dataframe(imp, use_container_width=True)

# =========================
# DEMO: MANUAL SINGLE TXN
# =========================
st.write("---")
st.subheader("🔍 Manual Transaction Check (demo)")
col1, col2, col3 = st.columns(3)
with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=1.0, value=500.0)
with col2:
    hour = st.slider("Hour of Transaction", 0, 23, 12)
with col3:
    txn_count = st.number_input("Transactions in Last 1h", min_value=0, step=1, value=0)

if st.button("Predict Fraud"):
    sample = make_manual_row(amount, hour, txn_count)
    proba_1 = float(pipeline.predict_proba(sample)[:, 1][0])
    label = "Fraud 🚨" if proba_1 >= threshold else "Legit ✅"
    st.success(f"**Prediction:** {label} — probability = {proba_1:.3f} (threshold = {threshold:.3f})")

    if HAVE_MPL:
        fig4, ax4 = plt.subplots()
        ax4.hist([proba_1], bins=[0, 0.25, 0.5, 0.75, 1.0], rwidth=0.9)
        ax4.axvline(threshold, linestyle="--")
        ax4.set_title("This transaction’s probability bucket")
        ax4.set_xlabel("Probability")
        st.pyplot(fig4)

    with st.expander("Debug: first 20 features sent to the model"):
        st.write(sample.reindex(columns=expected_cols).iloc[:, :20])
