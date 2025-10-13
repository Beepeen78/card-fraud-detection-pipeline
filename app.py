
# app.py — Patched: builds minimal engineered features so raw CSVs work
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from google.cloud import bigquery
from snowflake_config import SnowflakeConfig, upload_to_snowflake
import argparse
import sys
from sklearn.metrics import f1_score, precision_score, recall_score

st.set_page_config(page_title="Credit Card Fraud – Scoring", layout="wide")

MODEL_PATH = Path("fraud_pipeline.joblib")
PBI_OUT_DIR = Path("powerbi/out"); PBI_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load tuning defaults if present
TUNING_FILE = Path("tuning_results.json")
_TUNED = {"heuristic_alpha": 0.0, "threshold": 0.5}
if TUNING_FILE.exists():
    try:
        import json
        with open(TUNING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            _TUNED.update({k: float(v) for k, v in data.items() if k in _TUNED})
    except Exception:
        pass

BQ_PROJECT   = os.getenv("BQ_PROJECT", "credit-card-fraud-pipeline")
BQ_DATASET   = os.getenv("BQ_DATASET", "fraud_prod")
BQ_TABLE_TX  = f"{BQ_PROJECT}.{BQ_DATASET}.transactions_scored"
BQ_TABLE_MET = f"{BQ_PROJECT}.{BQ_DATASET}.metrics_daily"

# Snowflake configuration
SNOWFLAKE_CONFIG = SnowflakeConfig()

ID_CANDS   = ["transaction_id","trans_num","id"]
CC_CANDS   = ["cc_num","customer_id","cust_id","user_id"]
AMT_CANDS  = ["amount","amt","transaction_amount"]
TS_CANDS   = ["trans_date_trans_time","timestamp","datetime","transaction_time"]
LABELS     = ["is_fraud","label","target","Class"]

@st.cache_resource(show_spinner=False)
def load_pipe():
    try:
        pipe = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}\n" \
                           "If the model requires extra packages (e.g. lightgbm), install them in your environment.")
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
    if amt_col in df.columns:
        df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
    else:
        df[amt_col] = 0.0
    # Ensure ts is a Series (not a scalar) so downstream .astype / .dt work consistently
    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    else:
        ts = pd.to_datetime(pd.Series([datetime.now(timezone.utc)] * len(df)), errors="coerce", utc=True)
    df["unix_time"] = (ts.astype("int64") // 10**9).astype("int64")
    df["hour"]      = ts.dt.hour.fillna(0).astype(int)
    df["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    df["dayofyear"] = ts.dt.dayofyear.fillna(1).astype(int)

    # Geo — ensure we always produce Series so downstream vector ops work
    def _num_series(col: str, default: float = 0.0) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
        return pd.Series([default] * len(df), dtype=float)

    lat = _num_series("lat", 0.0)
    lon = _num_series("long", 0.0)
    mlat = _num_series("merch_lat", 0.0)
    mlon = _num_series("merch_long", 0.0)

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

def upload_df_to_snowflake(df: pd.DataFrame, table_name: str) -> str:
    """Upload DataFrame to Snowflake."""
    if df is None or df.empty: return "Skipped (empty)"
    return upload_to_snowflake(df, table_name, SNOWFLAKE_CONFIG)

# Tuning and evaluation helpers (defined before UI so Streamlit can call them)
def compute_heuristic_score(feat: pd.DataFrame) -> pd.Series:
    """Compute a simple heuristic fraud score in [0,1] from engineered features.

    Uses amount, mean_distance, is_night and merchant category (if present) to
    compute a lightweight signal suitable for blending with model probabilities.
    """
    # amount contribution: softcap using tanh to avoid extreme influence
    amt = feat.get("amt", feat.get("amount", 0.0)).astype(float)
    amt_score = np.tanh(amt / (amt.std() + 1e-9) / 2.0).clip(0, 1)

    # distance contribution
    dist = feat.get("mean_distance", pd.Series([0.0] * len(feat))).astype(float)
    # scale distance so that 0-2000km maps to 0-1 roughly
    dist_score = (dist / 2000.0).clip(0, 1)

    # night-time contribution
    night = feat.get("is_night", pd.Series([0] * len(feat))).astype(int)

    # merchant category risk
    cat = feat.get("merchant_category", pd.Series([""] * len(feat))).astype(str).str.lower()
    risk_map = {"online": 0.7, "travel": 0.6, "entertainment": 0.5}
    cat_score = cat.map(risk_map).fillna(0.0).astype(float)

    # combine with chosen weights
    score = 0.45 * amt_score + 0.35 * dist_score + 0.15 * night + 0.05 * cat_score
    # normalize to [0,1]
    return score.clip(0, 1)


def tune_on_csv(csv_path: str, search_thresholds=None, search_alphas=None, target_flag_rate: float | None = None):
    """Grid-search threshold and heuristic_alpha on a labelled CSV to maximize F1 or match a target flag rate.

    Returns (best_threshold, best_alpha, metrics_dict)
    """
    if search_thresholds is None:
        search_thresholds = list(np.linspace(0.0, 0.9, 10))
    if search_alphas is None:
        search_alphas = list(np.linspace(0.0, 1.0, 11))

    raw = pd.read_csv(csv_path)
    pipe, expected = load_pipe()
    feat = build_features(raw)
    X = feat.reindex(columns=expected, fill_value=0.0) if expected is not None else feat
    model_proba = pipe.predict_proba(X)[:, 1]
    heuristic = compute_heuristic_score(feat)

    y_true = pd.to_numeric(raw.get("is_fraud", 0), errors="coerce").fillna(0).astype(int)

    best = None
    results = []
    for alpha in search_alphas:
        combined = (1.0 - alpha) * model_proba + alpha * heuristic.to_numpy()
        for thr in search_thresholds:
            preds = (combined >= thr).astype(int)
            if target_flag_rate is not None:
                flagged_rate = preds.mean()
                # score by closeness to target flag rate
                score = -abs(flagged_rate - target_flag_rate)
                metrics = {"flagged_rate": flagged_rate}
            else:
                f1 = f1_score(y_true, preds, zero_division=0)
                prec = precision_score(y_true, preds, zero_division=0)
                rec = recall_score(y_true, preds, zero_division=0)
                score = f1
                metrics = {"f1": f1, "precision": prec, "recall": rec, "flagged_rate": preds.mean()}

            results.append((alpha, thr, score, metrics))
            if best is None or score > best[0]:
                best = (score, alpha, thr, metrics)

    return best, results


def save_tuning_results(alpha: float, threshold: float, path: Path | str = "tuning_results.json") -> None:
    import json
    out = {"heuristic_alpha": float(alpha), "threshold": float(threshold)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def generate_evaluation_report(raw: pd.DataFrame, scored: pd.DataFrame, out_dir: Path | str = "eval_out") -> dict:
    """Generate evaluation artifacts (metrics JSON, PR/ROC plots HTML, confusion matrix PNG) and return metrics."""
    from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = pd.to_numeric(raw.get("is_fraud", 0), errors="coerce").fillna(0).astype(int)
    y_scores = scored["fraud_probability"].astype(float)
    y_pred = scored["fraud_prediction"].astype(int)

    prec, rec, pr_thr = precision_recall_curve(y_true, y_scores)
    fpr, tpr, roc_thr = roc_curve(y_true, y_scores)
    pr_auc = auc(rec, prec)
    roc_auc = auc(fpr, tpr)

    # Save PR and ROC as simple HTML using plotly
    import plotly.graph_objects as go
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name='PR'))
    pr_fig.update_layout(title=f'Precision-Recall curve (AUC={pr_auc:.4f})', xaxis_title='Recall', yaxis_title='Precision')
    pr_html = out_dir / 'pr_curve.html'
    pr_fig.write_html(pr_html)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='rand', line=dict(dash='dash')))
    roc_fig.update_layout(title=f'ROC curve (AUC={roc_auc:.4f})', xaxis_title='FPR', yaxis_title='TPR')
    roc_html = out_dir / 'roc_curve.html'
    roc_fig.write_html(roc_html)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['neg','pos'], columns=['pred_neg','pred_pos'])
    cm_csv = out_dir / 'confusion_matrix.csv'
    cm_df.to_csv(cm_csv)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        'pr_auc': float(pr_auc), 'roc_auc': float(roc_auc), 'confusion_matrix': str(cm.tolist()), 'classification_report': report
    }
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Save scored with predictions for inspection
    scored.to_csv(out_dir / 'scored_with_preds.csv', index=False)

    return metrics


# UI
def _running_in_streamlit() -> bool:
    """Return True when executing inside a Streamlit script run context.
    We try several import paths to be compatible with Streamlit versions.
    """
    try:
        # Newer Streamlit
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        try:
            # Older Streamlit
            from streamlit.script_run_context import get_script_run_ctx
            return get_script_run_ctx() is not None
        except Exception:
            return False


def compute_heuristic_score(feat: pd.DataFrame) -> pd.Series:
    """Compute a simple heuristic fraud score in [0,1] from engineered features.

    Uses amount, mean_distance, is_night and merchant category (if present) to
    compute a lightweight signal suitable for blending with model probabilities.
    """
    # amount contribution: softcap using tanh to avoid extreme influence
    amt = feat.get("amt", feat.get("amount", 0.0)).astype(float)
    amt_score = np.tanh(amt / (amt.std() + 1e-9) / 2.0).clip(0, 1)

    # distance contribution
    dist = feat.get("mean_distance", pd.Series([0.0] * len(feat))).astype(float)
    # scale distance so that 0-2000km maps to 0-1 roughly
    dist_score = (dist / 2000.0).clip(0, 1)

    # night-time contribution
    night = feat.get("is_night", pd.Series([0] * len(feat))).astype(int)

    # merchant category risk
    cat = feat.get("merchant_category", pd.Series([""] * len(feat))).astype(str).str.lower()
    risk_map = {"online": 0.7, "travel": 0.6, "entertainment": 0.5}
    cat_score = cat.map(risk_map).fillna(0.0).astype(float)

    # combine with chosen weights
    score = 0.45 * amt_score + 0.35 * dist_score + 0.15 * night + 0.05 * cat_score
    # normalize to [0,1]
    return score.clip(0, 1)


def _score_dataframe(raw: pd.DataFrame, thr: float = 0.5, heuristic_alpha: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a DataFrame of raw transactions and return (scored, metrics_daily).

    Note: This function currently returns model probabilities. To combine with a
    lightweight rule-based heuristic, call `compute_heuristic_score` and mix
    externally (or pass heuristic_alpha via caller and combine here).
    """
    pipe, expected = load_pipe()
    feat = build_features(raw)
    X = feat.reindex(columns=expected, fill_value=0.0) if expected is not None else feat
    proba = pipe.predict_proba(X)[:, 1]
    # Compute heuristic score from the engineered features
    heuristic = compute_heuristic_score(feat)

    # Blend model probability with heuristic according to alpha
    heuristic_alpha = float(heuristic_alpha or 0.0)
    combined_proba = (1.0 - heuristic_alpha) * proba + heuristic_alpha * heuristic.to_numpy()
    preds = (combined_proba >= thr).astype(int)

    id_col = first(raw, ID_CANDS) or "transaction_id"
    cc_col = first(raw, CC_CANDS) or "cc_num"
    amt_col = first(raw, AMT_CANDS) or "amount"

    # Helpers to ensure we always produce Series (not scalars) when columns are missing
    def _col_series(col: str, default="") -> pd.Series:
        if col in raw.columns:
            return raw[col]
        return pd.Series([default] * len(raw))

    def _num_col(col: str, default: float = 0.0) -> pd.Series:
        if col in raw.columns:
            return pd.to_numeric(raw[col], errors="coerce").fillna(default).astype(float)
        return pd.Series([default] * len(raw), dtype=float)

    txn_ids = _col_series(id_col).astype(str) if id_col in raw.columns else pd.Series(np.arange(len(raw)).astype(str))
    cust_ids = _col_series(cc_col, "").astype(str)
    amounts = _num_col(amt_col, 0.0)
    is_fraud_series = _num_col("is_fraud", 0).astype(int)

    scored = pd.DataFrame({
        "transaction_id": txn_ids,
        "customer_id": cust_ids,
        "amount": amounts,
        # keep raw model probability and also the blended probability
        "fraud_probability_model": proba.astype(float),
        "fraud_probability": combined_proba.astype(float),
        "fraud_prediction": preds.astype(int),
        "is_fraud": is_fraud_series,
        "score_time": pd.Timestamp.now(tz="UTC"),
    })

    # attach heuristic for diagnostics
    scored["heuristic_score"] = heuristic.values

    metrics_daily = build_metrics_daily(scored)
    return scored, metrics_daily


# Detect if user intends to run CLI mode (pass --csv on command line) and we're not in
# a Streamlit run context. In that case, skip constructing Streamlit UI so the script
# can execute the CLI flow below.
_is_cli_request = ("--csv" in sys.argv) and (not _running_in_streamlit())

if not _is_cli_request:
    # Streamlit UI (kept minimal here so CLI mode can reuse scoring logic)
    st.title("💳 Credit Card Fraud – Scoring & Export")
    thr = st.sidebar.slider("Decision threshold", 0.0, 1.0, float(_TUNED.get("threshold", 0.5)), 0.001)
    heuristic_alpha = st.sidebar.slider("Heuristic blend (alpha)", 0.0, 1.0, float(_TUNED.get("heuristic_alpha", 0.0)), 0.01)

    uploaded = st.file_uploader("Upload a CSV of raw transactions", type=["csv"])

    if not uploaded:
        # If we're not inside a Streamlit run context, exit with a helpful message so running
        # `python app.py` doesn't continue and try to call pandas.read_csv(None).
        if not _running_in_streamlit():
            print("No file uploaded. To use the interactive UI run: streamlit run app.py")
            print("Or run in CLI mode: python app.py --csv path/to/file.csv")
            raise SystemExit(1)
        # Inside Streamlit: stop execution until the user uploads a file
        st.stop()

    raw = pd.read_csv(uploaded)
    st.write(f"Rows uploaded: **{len(raw):,}**")

    # Allow auto-tune: if pressed, run tuner on the uploaded CSV and update defaults
    if st.sidebar.button("Auto-tune on uploaded CSV"):
        with st.spinner("Running tuner — this may take a minute..."):
            # save uploaded temp to disk so tuner can read it
            tmp_path = Path(".uploaded_tmp.csv")
            raw.to_csv(tmp_path, index=False)
            best, _ = tune_on_csv(str(tmp_path))
            if best is not None:
                _, a, t, _ = best
                save_tuning_results(a, t)
                st.sidebar.success(f"Tuned: alpha={a:.2f}, threshold={t:.2f}")
                heuristic_alpha = float(a)
                thr = float(t)
            tmp_path.unlink(missing_ok=True)

    scored, metrics_daily = _score_dataframe(raw, thr=thr, heuristic_alpha=heuristic_alpha)

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows Scored", f"{len(scored):,}")
    c2.metric("Predicted Fraud", f"{int(scored['fraud_prediction'].sum()):,}")
    c3.metric("Mean Prob", f"{scored['fraud_probability'].mean():.3f}")

    # Table & chart
    st.subheader("🔎 Top Suspicious")
    st.dataframe(scored.sort_values("fraud_probability", ascending=False).head(50).reset_index(drop=True))
    st.subheader("Risk Distribution")
    st.plotly_chart(px.histogram(scored, x="fraud_probability", nbins=40), use_container_width=True)

    # Export UI
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
        # BigQuery export
        if st.button("Export to BigQuery (append)"):
            try:
                m1 = upload_df_to_bq(scored[["transaction_id", "customer_id", "amount", "fraud_probability", "fraud_prediction", "is_fraud", "score_time"]], BQ_TABLE_TX)
                m2 = upload_df_to_bq(metrics_daily[["date", "transactions", "flagged", "avg_risk", "total_amount", "actual_fraud"]], BQ_TABLE_MET)
                st.success(f"✅ BigQuery upload complete:\n- {m1}\n- {m2}")
            except Exception as e:
                st.error(f"BigQuery upload failed: {e}")

        # Snowflake export
        if st.button("Export to Snowflake (append)"):
            try:
                if not SNOWFLAKE_CONFIG.validate_config():
                    st.error("❌ Snowflake not configured. Please set environment variables:\n- SNOWFLAKE_ACCOUNT\n- SNOWFLAKE_USER\n- SNOWFLAKE_PASSWORD")
                else:
                    m1 = upload_df_to_snowflake(scored[["transaction_id", "customer_id", "amount", "fraud_probability", "fraud_prediction", "is_fraud", "score_time"]], SNOWFLAKE_CONFIG.transactions_table)
                    m2 = upload_df_to_snowflake(metrics_daily[["date", "transactions", "flagged", "avg_risk", "total_amount", "actual_fraud"]], SNOWFLAKE_CONFIG.metrics_table)
                    st.success(f"✅ Snowflake upload complete:\n- {m1}\n- {m2}")
            except Exception as e:
                st.error(f"Snowflake upload failed: {e}")


def _run_cli(args: argparse.Namespace) -> int:
    """Run a non-interactive scoring flow for a CSV and write outputs to disk."""
    csv_path = args.csv
    out_dir = Path(args.out_dir) if args.out_dir else PBI_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(csv_path)
    heuristic_alpha = getattr(args, "heuristic_alpha", 0.0)
    scored, metrics_daily = _score_dataframe(raw, thr=args.threshold, heuristic_alpha=heuristic_alpha)

    tx_path = out_dir / "transactions_scored.csv"
    metrics_path = out_dir / "metrics_daily.csv"
    scored.to_csv(tx_path, index=False)
    metrics_daily.to_csv(metrics_path, index=False)

    print(f"Wrote scored transactions to: {tx_path}")
    print(f"Wrote daily metrics to:     {metrics_path}")
    return 0


def tune_on_csv(csv_path: str, search_thresholds=None, search_alphas=None, target_flag_rate: float | None = None):
    """Grid-search threshold and heuristic_alpha on a labelled CSV to maximize F1 or match a target flag rate.

    Returns (best_threshold, best_alpha, metrics_dict)
    """
    if search_thresholds is None:
        search_thresholds = list(np.linspace(0.0, 0.9, 10))
    if search_alphas is None:
        search_alphas = list(np.linspace(0.0, 1.0, 11))

    raw = pd.read_csv(csv_path)
    pipe, expected = load_pipe()
    feat = build_features(raw)
    X = feat.reindex(columns=expected, fill_value=0.0) if expected is not None else feat
    model_proba = pipe.predict_proba(X)[:, 1]
    heuristic = compute_heuristic_score(feat)

    y_true = pd.to_numeric(raw.get("is_fraud", 0), errors="coerce").fillna(0).astype(int)

    best = None
    results = []
    for alpha in search_alphas:
        combined = (1.0 - alpha) * model_proba + alpha * heuristic.to_numpy()
        for thr in search_thresholds:
            preds = (combined >= thr).astype(int)
            if target_flag_rate is not None:
                flagged_rate = preds.mean()
                # score by closeness to target flag rate
                score = -abs(flagged_rate - target_flag_rate)
                metrics = {"flagged_rate": flagged_rate}
            else:
                f1 = f1_score(y_true, preds, zero_division=0)
                prec = precision_score(y_true, preds, zero_division=0)
                rec = recall_score(y_true, preds, zero_division=0)
                score = f1
                metrics = {"f1": f1, "precision": prec, "recall": rec, "flagged_rate": preds.mean()}

            results.append((alpha, thr, score, metrics))
            if best is None or score > best[0]:
                best = (score, alpha, thr, metrics)

    return best, results


def save_tuning_results(alpha: float, threshold: float, path: Path | str = "tuning_results.json") -> None:
    import json
    out = {"heuristic_alpha": float(alpha), "threshold": float(threshold)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def generate_evaluation_report(raw: pd.DataFrame, scored: pd.DataFrame, out_dir: Path | str = "eval_out") -> dict:
    """Generate evaluation artifacts (metrics JSON, PR/ROC plots HTML, confusion matrix PNG) and return metrics."""
    from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = pd.to_numeric(raw.get("is_fraud", 0), errors="coerce").fillna(0).astype(int)
    y_scores = scored["fraud_probability"].astype(float)
    y_pred = scored["fraud_prediction"].astype(int)

    prec, rec, pr_thr = precision_recall_curve(y_true, y_scores)
    fpr, tpr, roc_thr = roc_curve(y_true, y_scores)
    pr_auc = auc(rec, prec)
    roc_auc = auc(fpr, tpr)

    # Save PR and ROC as simple HTML using plotly
    import plotly.graph_objects as go
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name='PR'))
    pr_fig.update_layout(title=f'Precision-Recall curve (AUC={pr_auc:.4f})', xaxis_title='Recall', yaxis_title='Precision')
    pr_html = out_dir / 'pr_curve.html'
    pr_fig.write_html(pr_html)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='rand', line=dict(dash='dash')))
    roc_fig.update_layout(title=f'ROC curve (AUC={roc_auc:.4f})', xaxis_title='FPR', yaxis_title='TPR')
    roc_html = out_dir / 'roc_curve.html'
    roc_fig.write_html(roc_html)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['neg','pos'], columns=['pred_neg','pred_pos'])
    cm_csv = out_dir / 'confusion_matrix.csv'
    cm_df.to_csv(cm_csv)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        'pr_auc': float(pr_auc), 'roc_auc': float(roc_auc), 'confusion_matrix': str(cm.tolist()), 'classification_report': report
    }
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Save scored with predictions for inspection
    scored.to_csv(out_dir / 'scored_with_preds.csv', index=False)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score transactions CSV (CLI mode) or run Streamlit UI")
    parser.add_argument("--csv", help="Path to CSV to score (enables CLI mode)")
    parser.add_argument("--out-dir", help="Directory to write scored outputs (defaults to powerbi/out)")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for fraud prediction")
    parser.add_argument("--heuristic-alpha", type=float, default=None, help="Blend weight for heuristic score (0=no blend, 1=only heuristic)")
    parsed = parser.parse_args()

    # Only run CLI mode when a CSV path is provided and we're not inside a Streamlit run context
    if parsed.csv and not _running_in_streamlit():
        # fill in tuned defaults when CLI args are omitted
        if parsed.threshold is None:
            parsed.threshold = float(_TUNED.get("threshold", 0.5))
        if parsed.heuristic_alpha is None:
            parsed.heuristic_alpha = float(_TUNED.get("heuristic_alpha", 0.0))
        raise SystemExit(_run_cli(parsed))
