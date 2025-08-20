# app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# -----------------------------
# Load pipeline + threshold
# -----------------------------
PIPE_PATH = "fraud_pipeline.joblib"
pipeline = joblib.load(PIPE_PATH)

with open("inference_threshold.json", "r") as f:
    t = json.load(f)
THRESHOLD = float(t.get("best_threshold", t.get("threshold", 0.5)))

st.title("💳 Credit Card Fraud Detection")

# -----------------------------
# Determine expected features
# -----------------------------
def get_expected_columns_from_pipeline(p):
    # Works if your ColumnTransformer was saved with verbose_feature_names_out=False
    try:
        return list(p.named_steps["prep"].get_feature_names_out())
    except Exception:
        pass
    # Fallback (works for plain sklearn models)
    return list(getattr(p, "feature_names_in_", []))

EXPECTED_COLS = get_expected_columns_from_pipeline(pipeline)
if not EXPECTED_COLS:
    st.error(
        "Could not infer feature names from the saved pipeline. "
        "Please re-save the pipeline with a ColumnTransformer and "
        "`verbose_feature_names_out=False`, or provide a feature list file."
    )
    st.stop()

# -----------------------------
# Utilities
# -----------------------------
def align_numeric_df(df_in: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """
    Reindex to expected columns, coerce to numeric, fill missing with 0.
    Extra columns (not used by the model) are dropped automatically.
    """
    df = df_in.copy()
    # Keep only expected columns (drop extras)
    df = df.reindex(columns=expected_cols)
    # Coerce to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0.0)
    # Order exactly as the model expects
    return df.loc[:, expected_cols]

def make_manual_row(amount: float,
                    hour: int,
                    txn_1h: int,
                    advanced: dict | None = None) -> pd.DataFrame:
    """
    Build a single-row dataframe matching EXPECTED_COLS.
    Unspecified features default to 0.
    """
    row = {c: 0.0 for c in EXPECTED_COLS}

    # Core
    row["amt"] = float(amount)
    row["hour"] = int(hour)
    row["txn_count_last_1h"] = float(txn_1h)

    # Cyclic time (if present in your feature set)
    if "hour_sin" in row: row["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    if "hour_cos" in row: row["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Night / business hours (if present)
    if "is_night" in row: row["is_night"] = 1.0 if (hour < 6 or hour >= 22) else 0.0
    if "is_business_hours" in row: row["is_business_hours"] = 1.0 if (9 <= hour < 17) else 0.0

    # Day-of-week defaults to Monday (0) unless advanced overrides it
    if "dayofweek" in row and ("dayofweek" not in (advanced or {})):
        row["dayofweek"] = 0.0
    if "dow_sin" in row and ("dayofweek" in (advanced or {})):
        dow = int(advanced["dayofweek"])
        row["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        row["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Apply advanced overrides (only if those columns exist)
    if advanced:
        for k, v in advanced.items():
            if k in row:
                row[k] = float(v)

    df = pd.DataFrame([row])
    return align_numeric_df(df, EXPECTED_COLS)

def score(df_features: pd.DataFrame) -> np.ndarray:
    """Return probability of class 1 (fraud)."""
    return pipeline.predict_proba(df_features)[:, 1]

# =========================================================
# 1) CSV UPLOAD + SCORING (adds probability + prediction)
# =========================================================
st.header("📤 Upload CSV to Score")

uploaded = st.file_uploader(
    "Upload a CSV whose columns match the engineered features expected by the model.",
    type=["csv"],
)

if uploaded:
    # Try UTF-8, then Latin-1 on failure
    try:
        raw = pd.read_csv(uploaded, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded.seek(0)
        raw = pd.read_csv(uploaded, encoding="latin1")

    # Let the user know if columns are missing; we’ll still fill them with 0’s
    missing = [c for c in EXPECTED_COLS if c not in raw.columns]
    extra = [c for c in raw.columns if c not in EXPECTED_COLS]
    if missing:
        st.warning(f"{len(missing)} expected columns not found and will be filled with 0: {missing[:10]}{' …' if len(missing)>10 else ''}")
    if extra:
        st.info(f"{len(extra)} extra columns will be ignored: {extra[:10]}{' …' if len(extra)>10 else ''}")

    X = align_numeric_df(raw, EXPECTED_COLS)
    proba = score(X)
    preds = (proba >= THRESHOLD).astype(int)

    out = raw.copy()
    out["fraud_probability"] = proba
    out["fraud_prediction"] = preds

    n_fraud = int((out["fraud_prediction"] == 1).sum())
    st.success(f"Scored {len(out):,} rows — **flagged {n_fraud:,} as fraud** at threshold {THRESHOLD:.3f}.")

    # Show a small preview
    st.dataframe(out.head(50), use_container_width=True)
    st.download_button(
        "⬇️ Download Results (with predictions)",
        data=out.to_csv(index=False),
        file_name="scored_results.csv",
        mime="text/csv",
    )

st.divider()

# =========================================================
# 2) MANUAL DEMO CHECK
# =========================================================
st.header("🔍 Manual Transaction Check (demo)")

colA, colB, colC = st.columns(3)
with colA:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=5000.0, step=10.0)
with colB:
    hour = st.slider("Hour of Transaction", 0, 23, 12)
with colC:
    txn_1h = st.number_input("Transactions in Last 1h", min_value=0, value=0, step=1)

with st.expander("Advanced signals (optional)"):
    a1, a2, a3 = st.columns(3)
    a4, a5, a6 = st.columns(3)
    a7, a8, a9 = st.columns(3)

    # Only set keys that might actually exist in your feature list
    adv = {}
    adv["total_amt_last_24h"] = a1.number_input("Total Amount Last 24h", min_value=0.0, value=0.0, step=100.0)
    adv["mean_amt"]            = a2.number_input("Mean Amount (user history)", min_value=0.0, value=0.0, step=10.0)
    adv["std_amt"]             = a3.number_input("Std Amount (user history)", min_value=0.0, value=0.0, step=10.0)
    adv["median_amt"]          = a4.number_input("Median Amount (user history)", min_value=0.0, value=0.0, step=10.0)
    adv["max_amt"]             = a5.number_input("Max Amount (user history)", min_value=0.0, value=0.0, step=100.0)
    adv["city_pop"]            = a6.number_input("City Population", min_value=0.0, value=0.0, step=1000.0)
    adv["time_since_last_txn"] = a7.number_input("Time Since Last Txn (sec)", min_value=0.0, value=0.0, step=1.0)
    adv["transaction_count"]   = a8.number_input("Txn Count Last 24h (user)", min_value=0.0, value=0.0, step=1.0)
    adv["dayofweek"]           = int(a9.selectbox("Day of Week (Mon=0…Sun=6)", list(range(7)), index=0))

    # Remove keys for columns your model doesn't use to keep it clean
    adv = {k: v for k, v in adv.items() if k in EXPECTED_COLS}

if st.button("Predict Fraud"):
    sample = make_manual_row(amount, hour, txn_1h, advanced=adv)
    proba = float(score(sample)[0])
    label = "Fraud 🚨" if proba >= THRESHOLD else "Legit ✅"
    st.success(f"**Prediction:** {label} — probability = {proba:.3f} (threshold = {THRESHOLD:.3f})")

    with st.expander("Debug: first 20 features sent to the model"):
        st.dataframe(sample.iloc[:, :20], use_container_width=True)
