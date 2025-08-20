# app.py
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Optional: nicer charts with Plotly (falls back to Streamlit charts if not installed)
try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Card Fraud Detection")
st.caption("Upload engineered transactions (columns matching the training features). The app scores each row and highlights risky ones.")

# -------------------------------------------------
# Load model & threshold (cached)
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(path: str = "fraud_pipeline.joblib"):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_threshold(path: str = "inference_threshold.json", default_value: float = 0.5) -> float:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return float(data.get("best_threshold", data.get("threshold", default_value)))
    except Exception:
        return default_value

pipeline = load_pipeline()
THRESHOLD = load_threshold()

# -------------------------------------------------
# Get expected feature names from the pipeline
# -------------------------------------------------
def get_expected_columns_from_pipeline(pl) -> list[str]:
    # Try ColumnTransformer first (what we saved earlier)
    try:
        prep = pl.named_steps.get("prep")
        if prep is not None:
            return list(prep.get_feature_names_out())
    except Exception:
        pass

    # Fallback to sklearn's feature_names_in_
    try:
        return list(pl.feature_names_in_)
    except Exception:
        return []

EXPECTED_COLS = get_expected_columns_from_pipeline(pipeline)
if not EXPECTED_COLS:
    st.error("❌ Could not infer expected feature columns from the pipeline. "
             "Please re-save the pipeline with `verbose_feature_names_out=False` or include a feature list.")
    st.stop()

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def robust_read_csv(upload) -> pd.DataFrame:
    """Read CSV with a UTF-8 first, then latin-1 fallback."""
    try:
        return pd.read_csv(upload, encoding="utf-8")
    except UnicodeDecodeError:
        upload.seek(0)
        return pd.read_csv(upload, encoding="latin1")

def align_to_expected(df_raw: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Return numeric, ordered frame ready for the pipeline."""
    # Keep only expected columns for scoring
    X = df_raw.reindex(columns=expected_cols, fill_value=0)

    # Coerce everything to numeric if possible
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    # Ensure order and dtype
    X = X.reindex(columns=expected_cols, fill_value=0.0).astype(float)
    return X

def score_dataframe(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score and return (aligned_scored, original_with_scores)."""
    X = align_to_expected(df_raw, EXPECTED_COLS)
    proba = pipeline.predict_proba(X)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)

    X_scored = X.copy()
    X_scored["fraud_probability"] = proba
    X_scored["fraud_prediction"] = preds

    out = df_raw.copy()
    out["fraud_probability"] = proba
    out["fraud_prediction"] = preds
    return X_scored, out

def make_sample_template(expected_cols: list[str]) -> pd.DataFrame:
    """Two-row sample with benign and suspicious patterns using the expected schema."""
    base = {c: 0 for c in expected_cols}

    # Helper to safely set a value if the column exists
    def s(d, key, val):
        if key in d:
            d[key] = val

    # Row 1: benign-ish, mid-day small amount
    r1 = base.copy()
    s(r1, "amt", 35)
    s(r1, "hour", 11)
    s(r1, "dayofweek", 2)
    s(r1, "is_weekend", 0)
    s(r1, "txn_count_last_1h", 0)
    s(r1, "total_amt_last_24h", 0)
    s(r1, "dist_home_merch", 2)
    s(r1, "city_pop", 50000)

    # Row 2: suspicious, night large amount, recent activity spike
    r2 = base.copy()
    s(r2, "amt", 12000)
    s(r2, "hour", 2)
    s(r2, "is_weekend", 1)
    s(r2, "is_night", 1)
    s(r2, "txn_count_last_1h", 4)
    s(r2, "total_amt_last_24h", 350000)
    s(r2, "dist_home_merch", 220)
    s(r2, "city_pop", 50000)
    s(r2, "month", 8)
    s(r2, "dayofweek", 6)

    return pd.DataFrame([r1, r2], columns=expected_cols)

# -------------------------------------------------
# Header stats bar (empty containers; will populate after scoring)
# -------------------------------------------------
stats_cols = st.columns(4)
box_rows, box_fraud, box_mean, box_thresh = stats_cols

# -------------------------------------------------
# Upload area
# -------------------------------------------------
st.subheader("📤 Upload CSV")
uploaded = st.file_uploader("Upload a CSV with engineered feature columns", type=["csv"])

with st.expander("📄 Download a sample template", expanded=False):
    template_df = make_sample_template(EXPECTED_COLS)
    st.dataframe(template_df.head(10), height=220, use_container_width=True)
    st.download_button(
        "Download sample_template.csv",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_template.csv",
        mime="text/csv",
    )
    st.caption("Template includes two rows: one benign-like, one suspicious-like.")

# -------------------------------------------------
# Scoring flow
# -------------------------------------------------
if uploaded:
    df_raw = robust_read_csv(uploaded)

    # Hint user about missing features
    missing = [c for c in EXPECTED_COLS if c not in df_raw.columns]
    if missing:
        st.warning(
            f"⚠️ {len(missing)} expected feature(s) are missing and will be filled with 0: "
            + ", ".join(missing[:25])
            + (" ..." if len(missing) > 25 else "")
        )

    X_scored, out = score_dataframe(df_raw)

    # Summary metrics
    rows_scored = len(out)
    fraud_count = int(out["fraud_prediction"].sum())
    mean_p = float(out["fraud_probability"].mean()) if rows_scored else 0.0

    box_rows.metric("Rows Scored", f"{rows_scored:,}")
    box_fraud.metric("Predicted Fraud (count)", f"{fraud_count:,}")
    box_mean.metric("Mean Fraud Probability", f"{mean_p:0.3f}")
    box_thresh.metric("Threshold", f"{THRESHOLD:0.3f}")

    st.divider()

    # ---- Results Table(s)
    st.subheader("🔎 Top Suspicious Transactions")
    topn = out.sort_values("fraud_probability", ascending=False).head(200)
    st.dataframe(topn, use_container_width=True, height=280)

    st.download_button(
        "⬇️ Download Scored CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions_scored.csv",
        mime="text/csv",
    )

    st.divider()

    # ---- Charts
    st.subheader("📊 Visual Insights")

    c1, c2 = st.columns(2)

    # Probability distribution
    with c1:
        st.markdown("**Fraud probability distribution**")
        if px is not None:
            fig = px.histogram(out, x="fraud_probability", nbins=40, opacity=0.85)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(np.histogram(out["fraud_probability"], bins=40)[0])

    # Amount vs probability (if amt exists)
    with c2:
        if "amt" in out.columns:
            st.markdown("**Amount vs. Probability**")
            if px is not None:
                fig = px.scatter(
                    out,
                    x="amt",
                    y="fraud_probability",
                    opacity=0.7,
                    trendline="lowess",
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(out[["amt", "fraud_probability"]])
        else:
            st.info("Column `amt` not found; skipping amount/probability chart.")

    # Hour-wise mean probability (if hour exists)
    if "hour" in out.columns:
        st.markdown("**Average fraud probability by hour**")
        hour_agg = out.groupby("hour", dropna=False)["fraud_probability"].mean().reset_index()
        if px is not None:
            fig = px.bar(hour_agg, x="hour", y="fraud_probability")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(hour_agg.set_index("hour"))

    # If labels present, show quick precision/recall/F1
    if "is_fraud" in out.columns:
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score

            p = precision_score(out["is_fraud"], out["fraud_prediction"], zero_division=0)
            r = recall_score(out["is_fraud"], out["fraud_prediction"], zero_division=0)
            f1 = f1_score(out["is_fraud"], out["fraud_prediction"], zero_division=0)
            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{p:0.3f}")
            m2.metric("Recall", f"{r:0.3f}")
            m3.metric("F1", f"{f1:0.3f}")
        except Exception:
            st.info("`is_fraud` present but could not compute metrics (sklearn missing?).")

    with st.expander("🛠️ Debug / Scoring matrix"):
        st.caption("Aligned matrix sent to the model (first rows).")
        st.dataframe(X_scored.head(20), use_container_width=True, height=260)

else:
    st.info("Upload a CSV to score transactions. Or download the sample template above to try.")
