
# app.py — Patched: builds minimal engineered features so raw CSVs work
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from google.cloud import bigquery

st.set_page_config(page_title="Credit Card Fraud – Scoring", layout="wide")

MODEL_PATH = Path("fraud_pipeline.joblib")
PBI_OUT_DIR = Path("powerbi/out"); PBI_OUT_DIR.mkdir(parents=True, exist_ok=True)

BQ_PROJECT   = os.getenv("BQ_PROJECT", "credit-card-fraud-pipeline")
BQ_DATASET   = os.getenv("BQ_DATASET", "fraud_prod")
BQ_TABLE_TX  = f"{BQ_PROJECT}.{BQ_DATASET}.transactions_scored"
BQ_TABLE_MET = f"{BQ_PROJECT}.{BQ_DATASET}.metrics_daily"

ID_CANDS   = ["transaction_id","trans_num","id"]
CC_CANDS   = ["cc_num","customer_id","cust_id","user_id"]
AMT_CANDS  = ["amount","amt","transaction_amount"]
TS_CANDS   = ["trans_date_trans_time","timestamp","datetime","transaction_time"]
LABELS     = ["is_fraud","label","target","Class"]

@st.cache_resource(show_spinner=False)
def load_pipe():
    pipe = joblib.load(MODEL_PATH)
    exp = None
    try:
        if hasattr(pipe,"named_steps") and "prep" in pipe.named_steps:
            exp = list(pipe.named_steps["prep"].get_feature_names_out())
    except Exception:
        pass
    if exp is None and hasattr(pipe,"feature_names_in_"):
        exp = list(pipe.feature_names_in_)
    return pipe, exp

def first(df, cols):
    for c in cols:
        if c in df.columns: return c
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    # vectorized haversine
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Create a broad set of engineered features w/ safe defaults.
       Rolling/TE features that require history are set to 0.
    """
    df = df_raw.copy()

    # Core column names
    id_col  = first(df, ID_CANDS)  or "transaction_id"
    cc_col  = first(df, CC_CANDS)  or "cc_num"
    amt_col = first(df, AMT_CANDS) or "amount"
    ts_col  = first(df, TS_CANDS)  or "trans_date_trans_time"

    if id_col not in df.columns:
        df[id_col] = np.arange(len(df)).astype(str)

    # Coerce
    df[amt_col] = pd.to_numeric(df.get(amt_col, 0.0), errors="coerce").fillna(0.0)
    ts = pd.to_datetime(df.get(ts_col, datetime.utcnow()), errors="coerce", utc=True)
    df["unix_time"] = (ts.view("int64") // 10**9).astype("int64")
    df["hour"]      = ts.dt.hour.fillna(0).astype(int)
    df["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    df["dayofyear"] = ts.dt.dayofyear.fillna(1).astype(int)

    # Geo
    lat = pd.to_numeric(df.get("lat", 0), errors="coerce").fillna(0.0)
    lon = pd.to_numeric(df.get("long", 0), errors="coerce").fillna(0.0)
    mlat= pd.to_numeric(df.get("merch_lat", 0), errors="coerce").fillna(0.0)
    mlon= pd.to_numeric(df.get("merch_long", 0), errors="coerce").fillna(0.0)

    df["mean_distance"]   = haversine_km(lat, lon, mlat, mlon)
    df["dist_home_merch"] = df["mean_distance"]  # proxy if no home coords

    # Sine/cosine time encodings
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dayofweek"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dayofweek"]/7)

    # Flags
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    df["is_night"]   = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_business_hours"] = df["hour"].between(9,17).astype(int)

    # Basic amt stats per row (fallbacks; real training used windows)
    df["max_amt"]    = df[amt_col]
    df["median_amt"] = df[amt_col]
    df["std_amt"]    = 0.0
    df["mean_amt"]   = df[amt_col]

    # Window features (no history in single file) -> zeros
    for c in [
        "txn_count_last_1h","txn_count_last_24h","txn_count_last_1h_category",
        "txn_count_last_24h_category","total_amt_last_1h","total_amt_last_24h",
        "total_amt_last_1h_category","total_amt_last_24h_category",
        "time_since_last_txn","transaction_count"
    ]:
        df[c] = 0.0

    # Simple categorical encodings (fallbacks)
    if "gender" in df.columns:
        df["gender_bin"] = df["gender"].astype(str).str.lower().map({"m":1,"male":1,"f":0,"female":0}).fillna(0).astype(int)
    else:
        df["gender_bin"] = 0

    # Distance category / job target-encoding placeholders
    df["te_dist_category"] = 0.0
    df["te_job"] = 0.0

    # Direct passthroughs if present
    for passthru in ["city_pop","merch_zipcode","month"]:
        if passthru not in df.columns: df[passthru] = 0

    # Ensure original needed columns are present
    if "amt" not in df.columns: df["amt"] = df[amt_col]

    return df

def build_metrics_daily(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["date"] = pd.to_datetime(scored["score_time"], utc=True).dt.date
    out = (
        scored.groupby("date", as_index=False)
        .agg(
            transactions=("transaction_id","count"),
            flagged=("fraud_prediction","sum"),
            avg_risk=("fraud_probability","mean"),
            total_amount=("amount","sum"),
            actual_fraud=("is_fraud","sum")
        )
    )
    out["transactions"]=out["transactions"].astype("int64")
    out["flagged"]=out["flagged"].astype("int64")
    out["actual_fraud"]=out["actual_fraud"].astype("int64")
    return out

def upload_df_to_bq(df: pd.DataFrame, table: str) -> str:
    if df is None or df.empty: return "Skipped (empty)"
    client = bigquery.Client(project=BQ_PROJECT)
    job = client.load_table_from_dataframe(df, table,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"))
    job.result()
    return f"Appended {len(df)} rows to {table}"

# UI
st.title("💳 Credit Card Fraud – Scoring & Export")
thr = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.001)

uploaded = st.file_uploader("Upload a CSV of raw transactions", type=["csv"])
if not uploaded: st.stop()

raw = pd.read_csv(uploaded)
st.write(f"Rows uploaded: **{len(raw):,}**")

pipe, expected = load_pipe()

# Build features then align to model's expected columns
feat = build_features(raw)

# If the model expects a specific column set, align/fill zeros
if expected is not None:
    X = feat.reindex(columns=expected, fill_value=0.0)
else:
    X = feat

# Score
proba = pipe.predict_proba(X)[:,1]
preds = (proba >= thr).astype(int)

id_col = first(raw, ID_CANDS) or "transaction_id"
cc_col = first(raw, CC_CANDS) or "cc_num"
amt_col= first(raw, AMT_CANDS) or "amount"

scored = pd.DataFrame({
    "transaction_id": raw[id_col].astype(str) if id_col in raw.columns else np.arange(len(raw)).astype(str),
    "customer_id": raw.get(cc_col, ""),
    "amount": pd.to_numeric(raw.get(amt_col, 0.0), errors="coerce").fillna(0.0),
    "fraud_probability": proba.astype(float),
    "fraud_prediction": preds.astype(int),
    "is_fraud": pd.to_numeric(raw.get("is_fraud", 0), errors="coerce").fillna(0).astype(int),
    "score_time": pd.Timestamp.utcnow().tz_convert("UTC"),
})

# KPIs
c1,c2,c3 = st.columns(3)
c1.metric("Rows Scored", f"{len(scored):,}")
c2.metric("Predicted Fraud", f"{int(scored['fraud_prediction'].sum()):,}")
c3.metric("Mean Prob", f"{scored['fraud_probability'].mean():.3f}")

# Table & chart
st.subheader("🔎 Top Suspicious")
st.dataframe(scored.sort_values("fraud_probability", ascending=False).head(50).reset_index(drop=True))
st.subheader("Risk Distribution")
st.plotly_chart(px.histogram(scored, x="fraud_probability", nbins=40), use_container_width=True)

# Metrics + Export
metrics_daily = build_metrics_daily(scored)

st.subheader("📦 Export")
tx_csv = scored.to_csv(index=False).encode("utf-8")
met_csv = metrics_daily.to_csv(index=False).encode("utf-8")
c1, c2 = st.columns(2)
c1.download_button("Download transactions_scored.csv", data=tx_csv, file_name="transactions_scored.csv", mime="text/csv")
c1.download_button("Download metrics_daily.csv", data=met_csv, file_name="metrics_daily.csv", mime="text/csv")
scored.to_csv(PBI_OUT_DIR / "transactions_scored.csv", index=False)
metrics_daily.to_csv(PBI_OUT_DIR / "metrics_daily.csv", index=False)
c1.success(f"Saved CSVs to {PBI_OUT_DIR.as_posix()}")

with c2:
    if st.button("Export to BigQuery (append)"):
        try:
            m1 = upload_df_to_bq(scored[["transaction_id","customer_id","amount","fraud_probability","fraud_prediction","is_fraud","score_time"]], BQ_TABLE_TX)
            m2 = upload_df_to_bq(metrics_daily[["date","transactions","flagged","avg_risk","total_amount","actual_fraud"]], BQ_TABLE_MET)
            st.success(f"✅ BigQuery upload complete:\n- {m1}\n- {m2}")
        except Exception as e:
            st.error(f"BigQuery upload failed: {e}")
