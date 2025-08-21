# app.py
# -----------------------------
# Credit Card Fraud Detection – Streamlit App
# - Loads trained pipeline (LightGBM inside)
# - Auto-engineers missing features for raw CSVs
# - Aligns columns to model expectations
# - Scores, explains, and visualizes
# -----------------------------

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------- Page config -------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ------------- Constants -------------
MODEL_PATHS = [
    Path("fraud_pipeline.joblib"),
    Path("notebooks/fraud_pipeline.joblib"),
]
THRESHOLD_PATHS = [
    Path("inference_threshold.json"),
    Path("notebooks/inference_threshold.json"),
]

# Raw columns the app will look for when auto-engineering features
RAW_TIME_COLS = [
    "trans_date_trans_time", "transaction_time", "timestamp", "datetime"
]
RAW_AMOUNT_COLS = ["amt", "amount", "transaction_amount"]
RAW_ID_COLS = ["trans_num", "transaction_id", "id"]
RAW_CC_COLS = ["cc_num", "customer_id", "cust_id", "user_id"]

# Distance (haversine) columns
LAT_HOME_COLS = ["lat", "home_lat"]
LON_HOME_COLS = ["long", "home_long", "lng"]
LAT_MERCH_COLS = ["merch_lat", "merchant_lat"]
LON_MERCH_COLS = ["merch_long", "merchant_long", "merch_lng"]
ZIP_MERCH_COLS = ["merch_zipcode", "merchant_zip"]

# ------------- Cached loaders -------------
@st.cache_resource(show_spinner=False)
def load_pipeline() -> object:
    for p in MODEL_PATHS:
        if p.exists():
            return joblib.load(p)
    st.error("❌ Could not find `fraud_pipeline.joblib` (root or notebooks/).")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_threshold() -> float:
    for p in THRESHOLD_PATHS:
        if p.exists():
            obj = json.loads(Path(p).read_text())
            return float(obj.get("best_threshold", obj.get("threshold", 0.5)))
    return 0.5


@st.cache_resource(show_spinner=False)
def load_expected_feature_list(_pipeline) -> List[str]:
    """
    Try to discover the model's expected feature list in a robust order:
      1) feature_columns.pkl (root or notebooks)
      2) ColumnTransformer.get_feature_names_out() if available
      3) estimator.feature_names_in_ if available

    NOTE: `_pipeline` has a leading underscore so Streamlit won't try to hash it.
    """
    for p in [Path("feature_columns.pkl"), Path("notebooks/feature_columns.pkl")]:
        if p.exists():
            try:
                cols = joblib.load(p)
                if isinstance(cols, (list, np.ndarray, pd.Index)):
                    return list(cols)
            except Exception:
                pass

    try:
        if hasattr(_pipeline, "named_steps") and "prep" in _pipeline.named_steps:
            cols = _pipeline.named_steps["prep"].get_feature_names_out()
            return list(cols)
    except Exception:
        pass

    try:
        cols = getattr(_pipeline, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    except Exception:
        pass

    st.error(
        "❌ Unable to infer expected feature columns. "
        "Please save `feature_columns.pkl` during training or ensure your pipeline "
        "exposes column names via a ColumnTransformer."
    )
    st.stop()


# ------------- Feature engineering helpers -------------
def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_datetime(df: pd.DataFrame) -> Optional[pd.Series]:
    tcol = _first_present(df, RAW_TIME_COLS)
    if tcol is None:
        return None
    s = pd.to_datetime(df[tcol], errors="coerce", infer_datetime_format=True, utc=False)
    if s.notna().sum() == 0:
        try:
            s = pd.to_datetime(df[tcol], unit="s", errors="coerce")
        except Exception:
            pass
    return s


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = (lat2 - lat1)
    dlon = (lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def basic_feature_engineering(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # ---- Amount ----
    a_col = _first_present(df, RAW_AMOUNT_COLS)
    if a_col and "amt" not in df.columns:
        df["amt"] = pd.to_numeric(df[a_col], errors="coerce").fillna(0.0)
    elif "amt" not in df.columns:
        df["amt"] = 0.0

    # ---- ID columns ----
    id_col = _first_present(df, RAW_ID_COLS)
    if id_col and "trans_num" not in df.columns:
        df["trans_num"] = df[id_col].astype(str)
    elif "trans_num" not in df.columns:
        df["trans_num"] = np.arange(len(df)).astype(str)

    cc_col = _first_present(df, RAW_CC_COLS)
    if cc_col and "cc_num" not in df.columns:
        df["cc_num"] = df[cc_col]
    elif "cc_num" not in df.columns:
        df["cc_num"] = 0

    # ---- Time features ----
    t_series = _ensure_datetime(df)
    if t_series is None:
        df["hour"] = 0.0
        df["dayofweek"] = 0.0
        df["month"] = 1.0
        df["is_weekend"] = 0.0
        df["is_night"] = 0.0
        df["is_business_hours"] = 1.0
        df["hour_sin"] = 0.0
        df["hour_cos"] = 1.0
        df["dow_sin"] = 0.0
        df["dow_cos"] = 1.0
        df["dayofyear"] = 1.0
        df["unix_time"] = 0.0
    else:
        df["hour"] = t_series.dt.hour.astype(float)
        df["dayofweek"] = t_series.dt.dayofweek.astype(float)
        df["month"] = t_series.dt.month.astype(float)
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(float)
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(float)
        df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] < 17)).astype(float)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
        df["dayofyear"] = t_series.dt.dayofyear.astype(float)
        try:
            df["unix_time"] = (t_series.view("int64") // 10**9).astype(float)
        except Exception:
            df["unix_time"] = t_series.astype("int64", errors="ignore") // 10**9

    # ---- Geo/distance ----
    lat_h = _first_present(df, LAT_HOME_COLS)
    lon_h = _first_present(df, LON_HOME_COLS)
    lat_m = _first_present(df, LAT_MERCH_COLS)
    lon_m = _first_present(df, LON_MERCH_COLS)

    for c in ["lat", "long", "merch_lat", "merch_long"]:
        if c not in df.columns:
            df[c] = 0.0
    if lat_h and "lat" not in raw.columns:
        df["lat"] = pd.to_numeric(df[lat_h], errors="coerce").fillna(0.0)
    if lon_h and "long" not in raw.columns:
        df["long"] = pd.to_numeric(df[lon_h], errors="coerce").fillna(0.0)
    if lat_m and "merch_lat" not in raw.columns:
        df["merch_lat"] = pd.to_numeric(df[lat_m], errors="coerce").fillna(0.0)
    if lon_m and "merch_long" not in raw.columns:
        df["merch_long"] = pd.to_numeric(df[lon_m], errors="coerce").fillna(0.0)

    df["dist_home_merch"] = _haversine(
        df["lat"].astype(float),
        df["long"].astype(float),
        df["merch_lat"].astype(float),
        df["merch_long"].astype(float),
    )

    # Zip defaults if required by pipeline
    zipm = _first_present(df, ZIP_MERCH_COLS)
    if "merch_zipcode" not in df.columns:
        if zipm is not None:
            df["merch_zipcode"] = pd.to_numeric(df[zipm], errors="coerce").fillna(0.0)
        else:
            df["merch_zipcode"] = 0.0

    # ---- Per-card aggregates (windowless proxies) ----
    g = df.groupby("cc_num", dropna=False)["amt"]
    df["mean_amt"] = g.transform("mean").fillna(0.0)
    df["std_amt"] = g.transform("std").fillna(0.0)
    df["median_amt"] = g.transform("median").fillna(0.0)
    df["max_amt"] = g.transform("max").fillna(0.0)
    df["mean_distance"] = df.groupby("cc_num", dropna=False)["dist_home_merch"].transform("mean").fillna(0.0)
    df["transaction_count"] = df.groupby("cc_num", dropna=False)["amt"].transform("count").fillna(0.0)

    # ---- Velocity features (coarse but effective) ----
    if _ensure_datetime(df) is not None:
        df = df.sort_values(["cc_num", "unix_time"], kind="mergesort")
        df["time_since_last_txn"] = df.groupby("cc_num")["unix_time"].diff().fillna(0.0)

        grp = df.groupby("cc_num", group_keys=False)
        df["txn_count_last_1h"] = grp["unix_time"].apply(
            lambda s: pd.Series((s.values[:, None] - s.values[None, :] <= 3600).sum(axis=1), index=s.index)
        ).astype("float32")
        df["txn_count_last_24h"] = grp["unix_time"].apply(
            lambda s: pd.Series((s.values[:, None] - s.values[None, :] <= 86400).sum(axis=1), index=s.index)
        ).astype("float32")

        amt = df["amt"].astype(float).values
        df["total_amt_last_1h"] = grp["unix_time"].apply(
            lambda s: pd.Series(((s.values[:, None] - s.values[None, :] <= 3600) * amt[s.index]).sum(axis=1), index=s.index)
        ).astype("float32")
        df["total_amt_last_24h"] = grp["unix_time"].apply(
            lambda s: pd.Series(((s.values[:, None] - s.values[None, :] <= 86400) * amt[s.index]).sum(axis=1), index=s.index)
        ).astype("float32")
    else:
        df["time_since_last_txn"] = 0.0
        df["txn_count_last_1h"] = 0.0
        df["txn_count_last_24h"] = 0.0
        df["total_amt_last_1h"] = 0.0
        df["total_amt_last_24h"] = 0.0

    # Final hygiene
    for c in df.columns:
        if df[c].dtype == "O" and c not in {"trans_num"}:
            try:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception:
                pass
    return df


def align_to_expected(df_feat: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    aligned = df_feat.reindex(columns=expected_cols, fill_value=0.0)
    for c in aligned.columns:
        if aligned[c].dtype == "O":
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce").fillna(0.0)
    return aligned


# ------------- Scoring -------------
def score_dataframe(
    df_raw: pd.DataFrame,
    pipeline,
    expected_cols: List[str],
    threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_feat = basic_feature_engineering(df_raw)
    model_df = align_to_expected(df_feat, expected_cols)

    try:
        proba = pipeline.predict_proba(model_df)[:, 1]
    except Exception as e:
        st.exception(e)
        st.stop()
    preds = (proba >= threshold).astype(int)

    out = df_raw.copy()
    out["fraud_probability"] = proba
    out["fraud_prediction"] = preds
    return out, model_df


# ------------- UI -------------
st.title("💳 Credit Card Fraud Detection")

with st.sidebar:
    st.header("Model")
    pipeline = load_pipeline()
    threshold = load_threshold()
    st.markdown(f"**Decision threshold:** `{threshold:.3f}`")
    exp_cols = load_expected_feature_list(pipeline)
    st.caption(f"Model expects **{len(exp_cols)}** features.")

st.subheader("📤 Upload CSV")
uploaded = st.file_uploader(
    "Upload raw transactions (no need to pre-engineer). CSV only.",
    type=["csv"],
    help="The app will auto-engineer time/geo/velocity features, align to the trained model, and score.",
)

# Sample template download
template_cols = ["transaction_id", "trans_date_trans_time", "cc_num", "amount", "lat", "long", "merch_lat", "merch_long"]
buf = io.StringIO()
pd.DataFrame(columns=template_cols).to_csv(buf, index=False)
st.download_button("📄 Download a sample template", buf.getvalue(), "sample_template.csv", use_container_width=False)

if uploaded is None:
    st.info("Upload a CSV to start scoring. Keep at least a timestamp, amount, and some IDs for better features.")
    st.stop()

# Read uploaded
try:
    df_in = pd.read_csv(uploaded)
except UnicodeDecodeError:
    uploaded.seek(0)
    df_in = pd.read_csv(uploaded, encoding="latin1")

# Score
with st.spinner("Scoring…"):
    scored, X_model = score_dataframe(df_in, pipeline, exp_cols, threshold)

# KPIs
total_rows = len(scored)
fraud_count = int(scored["fraud_prediction"].sum())
mean_prob = float(scored["fraud_probability"].mean())

k1, k2, k3 = st.columns(3)
k1.metric("Rows Scored", f"{total_rows:,}")
k2.metric("Predicted Fraud (count)", f"{fraud_count:,}")
k3.metric("Mean Fraud Probability", f"{mean_prob:.3f}")

# ----------------- Top suspicious table (safe: no duplicate column names) -----------------
st.subheader("🔎 Top Suspicious Transactions")
topn = scored.nlargest(200, "fraud_probability").copy()

id_actual   = _first_present(scored, RAW_ID_COLS) or "trans_num"
amt_actual  = _first_present(scored, RAW_AMOUNT_COLS) or "amt"
cc_actual   = _first_present(scored, RAW_CC_COLS) or "cc_num"
hour_actual = "hour" if "hour" in scored.columns else None
label_actual = "is_fraud" if "is_fraud" in scored.columns else None

pairs = [
    ("transaction_id", id_actual),
    ("amount", amt_actual),
    ("hour", hour_actual),
    ("customer_id", cc_actual),
    ("is_fraud", label_actual),
    ("fraud_probability", "fraud_probability"),
    ("fraud_prediction", "fraud_prediction"),
]
pairs = [(lbl, col) for (lbl, col) in pairs if col is not None]

seen_actual = set()
safe_pairs = []
for lbl, col in pairs:
    if col not in seen_actual:
        safe_pairs.append((lbl, col))
        seen_actual.add(col)

disp = topn[[col for _, col in safe_pairs]].copy()
rename_map = {col: lbl for (lbl, col) in safe_pairs}
disp = disp.rename(columns=rename_map)

st.dataframe(disp, use_container_width=True, height=420)
# ------------------------------------------------------------------------------------------

# Download scored results
csv_io = io.StringIO()
scored.to_csv(csv_io, index=False)
st.download_button(
    "💾 Download Scored CSV",
    csv_io.getvalue(),
    file_name="scored_transactions.csv",
    use_container_width=False,
)

# Visuals
st.subheader("📊 Visual Insights")

c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(scored, x="fraud_probability", nbins=40, title="Fraud Probability Distribution")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # ---- SAFE scatter: color derived from the *same* sampled frame ----
    amt_col = _first_present(scored, RAW_AMOUNT_COLS) or "amt"
    sample = scored.sample(min(len(scored), 5000), random_state=42).copy()
    sample["pred_label"] = sample["fraud_prediction"].map({0: "Legit", 1: "Fraud"}).astype("category")

    fig2 = px.scatter(
        sample,
        x=amt_col,
        y="fraud_probability",
        color="pred_label",
        opacity=0.6,
        title="Amount vs Probability",
        category_orders={"pred_label": ["Legit", "Fraud"]},
        labels={"pred_label": "Prediction"},
    )
    fig2.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text="Prediction")
    st.plotly_chart(fig2, use_container_width=True)

# Explain simple drivers (non-SHAP, safe & fast)
if "amt" in scored.columns and "fraud_probability" in scored.columns:
    st.subheader("⚙️ Simple Drivers (univariate)")
    amt_q = pd.qcut(scored["amt"], q=10, duplicates="drop")
    by_bucket = scored.groupby(amt_q)["fraud_probability"].mean().reset_index()
    fig3 = px.bar(by_bucket, x="amt", y="fraud_probability", title="Avg Fraud Prob by Amount Decile")
    fig3.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)

st.caption("Tip: If predictions look off on *raw* data, include timestamp and per-card history so auto-engineering can build better velocity features.")
