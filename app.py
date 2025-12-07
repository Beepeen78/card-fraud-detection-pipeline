# app.py
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from evaluate_model import evaluate
from predict_fraud_batch import score_batch
from powerbi_export import export_powerbi_csvs, POWERBI_OUT


# -------------------------
# Streamlit page settings
# -------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection")


# -------------------------
# Helpers
# -------------------------
def _safe_get_bq_project() -> str | None:
    try:
        return st.secrets.get("bq_project", os.getenv("BQ_PROJECT", "")) or None
    except FileNotFoundError:
        return os.getenv("BQ_PROJECT") or None


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in ["transaction_id", "customer_id"]:
        if c in out.columns:
            out[c] = out[c].astype(str)

    if "unix_time" in out.columns:
        out["unix_time"] = pd.to_numeric(out["unix_time"], errors="coerce").fillna(0).astype("int64")

    # Support 'amt' alias
    if "amount" not in out.columns and "amt" in out.columns:
        out["amount"] = out["amt"]
    if "amount" in out.columns:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)

    if "is_fraud" in out.columns:
        out["is_fraud"] = pd.to_numeric(out["is_fraud"], errors="coerce").fillna(0).astype("int64")

    return out


def _export_to_bigquery(scored: pd.DataFrame, project: str) -> None:
    from google.cloud import bigquery
    from google.api_core.exceptions import NotFound, BadRequest

    client = bigquery.Client(project=project)
    dataset_id = f"{project}.fraud_prod"
    table_id = f"{dataset_id}.transactions_scored"

    try:
        client.get_dataset(dataset_id)
    except NotFound:
        client.create_dataset(bigquery.Dataset(dataset_id))

    base_fields = [
        bigquery.SchemaField("transaction_id", "STRING"),
        bigquery.SchemaField("amount", "NUMERIC"),
        bigquery.SchemaField("fraud_probability", "FLOAT64"),
        bigquery.SchemaField("fraud_prediction", "INT64"),
        bigquery.SchemaField("is_fraud", "INT64"),
        bigquery.SchemaField("score_time", "TIMESTAMP"),
    ]

    try:
        table = client.get_table(table_id)
        table_exists = True
    except NotFound:
        table_exists = False

    df_has_customer = "customer_id" in scored.columns
    schema = ([bigquery.SchemaField("customer_id", "STRING")] + base_fields) if df_has_customer else base_fields

    if not table_exists:
        client.create_table(bigquery.Table(table_id, schema=schema))

    df_out = scored.copy()
    if "transaction_id" not in df_out.columns:
        df_out["transaction_id"] = ""

    if df_has_customer:
        df_out["customer_id"] = df_out["customer_id"].astype(str)

    if "amount" in df_out.columns:
        df_out["amount"] = pd.to_numeric(df_out["amount"], errors="coerce").round(2)

    for c in ["fraud_prediction", "is_fraud"]:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce").fillna(0).astype("Int64")

    if "fraud_probability" in df_out.columns:
        df_out["fraud_probability"] = pd.to_numeric(df_out["fraud_probability"], errors="coerce")

    if "score_time" in df_out.columns:
        df_out["score_time"] = pd.to_datetime(df_out["score_time"], utc=True)

    ordered_cols = [f.name for f in schema if f.name in df_out.columns]
    df_out = df_out[ordered_cols]

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        source_format=bigquery.SourceFormat.PARQUET,
    )

    job = client.load_table_from_dataframe(df_out, table_id, job_config=job_config)
    job.result()  # raises on error


# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Decision threshold", min_value=0.01, max_value=0.95, value=0.50, step=0.01,
    help="Transactions with fraud_probability ≥ threshold are flagged as fraud."
)

if st.sidebar.button("Download sample template"):
    template_cols = [
        "transaction_id", "customer_id", "amount",
        "lat", "long", "city_pop", "unix_time",
        "merch_lat", "merch_long", "merch_zipcode", "is_fraud"
    ]
    sample_csv = pd.DataFrame(columns=template_cols).to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV template",
        data=sample_csv,
        file_name="sample_credit_card_transactions.csv",
        mime="text/csv",
    )


# -------------------------
# Main Upload & Scoring
# -------------------------
uploaded = st.file_uploader(
    "Upload a CSV of raw transactions (max ~200MB)",
    type=["csv"],
    help="Minimum: transaction_id + amount (or amt). More features = better accuracy."
)

BQ_PROJECT = _safe_get_bq_project()
if not BQ_PROJECT:
    st.info(
        "BigQuery export is optional → set `bq_project` in `.streamlit/secrets.toml` "
        "or the `BQ_PROJECT` env var to enable it."
    )

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Failed to read CSV. Make sure it's a valid file.")
        st.exception(e)
        st.stop()

    if raw_df.empty:
        st.error("Uploaded CSV is empty.")
        st.stop()

    raw_df = _coerce_types(raw_df)
    
    # === ADD THESE LINES HERE (right after _coerce_types) ===
    # Auto-map columns from the official kartik2112 fraudTest.csv / fraudTrain.csv
    if "trans_num" in raw_df.columns:
        raw_df["transaction_id"] = raw_df["trans_num"].astype(str)
    elif "transaction_id" not in raw_df.columns:
        raw_df["transaction_id"] = raw_df.index.astype(str)

    if "cc_num" in raw_df.columns and "customer_id" not in raw_df.columns:
        raw_df["customer_id"] = raw_df["cc_num"].astype(str)

    if "amt" in raw_df.columns and "amount" not in raw_df.columns:
        raw_df["amount"] = raw_df["amt"]
    # ====================================================

    # === Column validation continues below (no changes needed) ===
    required_cols = ["transaction_id"]
    # ...

    # === Column validation ===
    required_cols = ["transaction_id"]
    if "amount" not in raw_df.columns and "amt" not in raw_df.columns:
        required_cols.append("amount (or amt)")

    missing_required = [c for c in required_cols if c not in raw_df.columns and c != "amount (or amt)"]
    if missing_required:
        st.error(f"Missing required column(s): {', '.join(missing_required)}")
        st.stop()

    expected_features = ["lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]
    missing_features = [c for c in expected_features if c not in raw_df.columns]
    if missing_features:
        st.warning(
            f"Missing {len(missing_features)} recommended feature(s) → accuracy may be lower:\n"
            f"`{', '.join(missing_features[:6])}{'...' if len(missing_features)>6 else ''}`"
        )

    # === Scoring ===
    with st.spinner(f"Scoring {len(raw_df):,} transactions..."):
        try:
            scored = score_batch(raw_df, threshold=threshold)
        except Exception as e:
            st.error("Scoring failed — check your model (`predict_fraud_batch.py`) and column names.")
            st.exception(e)
            st.stop()

    # Timestamp all rows with current UTC time
    scored["score_time"] = pd.Timestamp.now(tz="UTC")

    # === Results ===
    st.success(f"Scored {len(scored):,} transactions in batch")

    st.subheader("Top Suspicious Transactions")
    display_cols = [c for c in [
        "transaction_id", "customer_id", "amount",
        "fraud_probability", "fraud_prediction", "is_fraud", "score_time"
    ] if c in scored.columns]

    top_n = scored.sort_values("fraud_probability", ascending=False)[display_cols].head(200)
    st.dataframe(top_n, use_container_width=True)

    # Score distribution
    if len(scored) > 1 and "fraud_probability" in scored.columns:
        fig = px.histogram(
            scored,
            x="fraud_probability",
            nbins=50,
            color="fraud_prediction",
            title="Fraud Probability Distribution",
            marginal="box",
            hover_data=["amount"]
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    # Evaluation metrics if ground truth exists
    metrics = evaluate(scored)
    if metrics:
        st.subheader("Model Performance (on uploaded labels)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC-ROC", f"{metrics.get('auc', 0):.3f}")
        col2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        col3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        col4.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
        with st.expander("Full metrics JSON"):
            st.json(metrics)

    # === Exports ===
    st.subheader("Export Results")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Export CSVs for Power BI"):
            try:
                p1, p2 = export_powerbi_csvs(scored)
                st.success(f"Saved to `{POWERBI_OUT}`")
                st.code(f"{p1}\n{p2}")
            except Exception as e:
                st.error(f"Power BI export failed: {e}")

    with c2:
        if BQ_PROJECT:
            if st.button("Export to BigQuery (append)"):
                with st.spinner("Uploading to BigQuery..."):
                    try:
                        _export_to_bigquery(scored, BQ_PROJECT)
                        st.success(f"Appended {len(scored)} rows to BigQuery!")
                    except Exception as e:
                        st.error(f"BigQuery export failed: {e}")
        else:
            st.caption("BigQuery export disabled (no project configured)")

else:
    st.info(
        """
        ### How to use
        1. Upload a CSV with at least `transaction_id` and `amount` (or `amt`)  
        2. Add geographic/time features for best accuracy  
        3. Adjust threshold → explore results → export

        **Minimum columns**: `transaction_id`, `amount`  
        **Recommended**: `lat`, `long`, `city_pop`, `unix_time`, `merch_lat`, `merch_long`
        """
    )