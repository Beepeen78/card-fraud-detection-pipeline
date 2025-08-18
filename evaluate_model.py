# evaluate_model.py — clean version
# Builds features on FULL DF (then splits), supports distance bucket + category windows,
# uses expected feature names from feature_columns.pkl (preferred), else model.

import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# --------------------
# Config
# --------------------
ROOT = Path(__file__).resolve().parent
ART  = ROOT / "notebooks"
OUT  = ROOT / "eval_out"
OUT.mkdir(parents=True, exist_ok=True)

CSV        = ROOT / r"dataset\credit_card_transactions.csv"
TIME_COL   = "trans_date_trans_time"
LABEL      = "is_fraud"
ID_COL     = "trans_num"
TEST_START = "2020-06-01"   # hold out last month

# --------------------
# Helpers
# --------------------
def haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

def get_expected_features(artifacts_dir: Path):
    """Prefer feature_columns.pkl; else try to read from model (LightGBM booster)."""
    p = artifacts_dir / "feature_columns.pkl"
    if p.exists():
        try:
            cols = list(joblib.load(p))
            return cols, "feature_columns.pkl"
        except Exception:
            pass
    # fallback via model
    model_path = artifacts_dir / "fraud_lgbm_calibrated.pkl"
    if not model_path.exists():
        model_path = artifacts_dir / "fraud_lgbm_model.pkl"
    m = joblib.load(model_path)
    for accessor in [
        "calibrated_classifiers_[0].estimator.booster_.feature_name",
        "booster_.feature_name",
    ]:
        try:
            names = eval(f"m.{accessor}()")
            if names:
                return list(names), f"model:{accessor}"
        except Exception:
            continue
    raise RuntimeError("Cannot determine expected feature names.")

def _window_counts_and_totals(sub: pd.DataFrame, window_seconds: int, prefix: str):
    """O(n) sliding window over time for counts/totals (within a group)."""
    t = sub["_ts"].to_numpy()
    a = sub["amt"].to_numpy()
    n = len(sub)
    cnt = np.zeros(n, dtype=np.int32)
    tot = np.zeros(n, dtype=float)
    j = 0
    for i in range(n):
        while t[i] - t[j] > window_seconds:
            j += 1
        cnt[i] = i - j + 1
        tot[i] = a[j:i+1].sum()
    return pd.DataFrame(
        {f"txn_count_last_{prefix}": cnt, f"total_amt_last_{prefix}": tot},
        index=sub.index,
    )

def build_features(df: pd.DataFrame, want_cols: set) -> pd.DataFrame:
    """Compute only the features the model expects."""
    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    sort_cols = [c for c in (ID_COL, TIME_COL) if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols).reset_index(drop=True)

    # --- time parts ---
    if "hour" in want_cols:       d["hour"] = d[TIME_COL].dt.hour
    if "dayofweek" in want_cols:  d["dayofweek"] = d[TIME_COL].dt.dayofweek
    if "dayofyear" in want_cols:  d["dayofyear"] = d[TIME_COL].dt.dayofyear
    if "month" in want_cols:      d["month"] = d[TIME_COL].dt.month
    if "is_weekend" in want_cols: d["is_weekend"] = d["dayofweek"].isin([5, 6]).astype(int)
    if "is_night" in want_cols:   d["is_night"] = d["hour"].isin(range(0, 6)).astype(int)
    if "is_business_hours" in want_cols:
        d["is_business_hours"] = d["hour"].between(9, 17, inclusive="both").astype(int)
    if "hour_sin" in want_cols:   d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    if "hour_cos" in want_cols:   d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    if "dow_sin" in want_cols:    d["dow_sin"] = np.sin(2 * np.pi * d["dayofweek"] / 7)
    if "dow_cos" in want_cols:    d["dow_cos"] = np.cos(2 * np.pi * d["dayofweek"] / 7)

    # --- distances ---
    if {"lat", "long", "merch_lat", "merch_long"}.issubset(d.columns) and "dist_home_merch" in want_cols:
        d["dist_home_merch"] = haversine(d["lat"], d["long"], d["merch_lat"], d["merch_long"])

    if "dist_category_bucket_idx" in want_cols and "dist_home_merch" in d.columns:
        bins = [-np.inf, 1, 10, 50, 100, np.inf]
        d["dist_category_bucket_idx"] = pd.cut(
            d["dist_home_merch"].fillna(-1), bins=bins, labels=False
        )

    # --- per-user rolling features (compute on FULL DF first) ---
    if ID_COL in d.columns and TIME_COL in d.columns:
        if "time_since_last_txn" in want_cols:
            d["time_since_last_txn"] = d.groupby(ID_COL)[TIME_COL].diff().dt.total_seconds().fillna(0)
        if "transaction_count" in want_cols:
            d["transaction_count"] = d.groupby(ID_COL).cumcount() + 1

        if "amt" in d.columns:
            g_amt = d.groupby(ID_COL)["amt"]
            if "mean_amt" in want_cols:   d["mean_amt"]   = g_amt.transform(lambda x: x.rolling(50, min_periods=1).mean())
            if "std_amt" in want_cols:    d["std_amt"]    = g_amt.transform(lambda x: x.rolling(50, min_periods=2).std().fillna(0))
            if "median_amt" in want_cols: d["median_amt"] = g_amt.transform(lambda x: x.rolling(50, min_periods=1).median())
            if "max_amt" in want_cols:    d["max_amt"]    = g_amt.transform(lambda x: x.rolling(50, min_periods=1).max())

        if "mean_distance" in want_cols and "dist_home_merch" in d.columns:
            d["mean_distance"] = d.groupby(ID_COL)["dist_home_merch"].transform(
                lambda x: x.rolling(50, min_periods=1).mean()
            )

        # Sliding windows (1h and 24h)
        if "amt" in d.columns:
            d["_ts"] = d[TIME_COL].astype("int64") // 10**9

            need_1h  = {"txn_count_last_1h","total_amt_last_1h"} & want_cols
            need_24h = {"txn_count_last_24h","total_amt_last_24h"} & want_cols
            if need_1h:
                tmp = d.groupby(ID_COL, group_keys=False).apply(
                    lambda sub: _window_counts_and_totals(sub, 3600, "1h"), include_groups=False
                )
                for c in need_1h:  d[c] = tmp[c]
            if need_24h:
                tmp = d.groupby(ID_COL, group_keys=False).apply(
                    lambda sub: _window_counts_and_totals(sub, 86400, "24h"), include_groups=False
                )
                for c in need_24h: d[c] = tmp[c]

            # Category-aware windows if expected
            if "category" in d.columns:
                need_1h_c  = {"txn_count_last_1h_category","total_amt_last_1h_category"} & want_cols
                need_24h_c = {"txn_count_last_24h_category","total_amt_last_24h_category"} & want_cols
                if need_1h_c:
                    tmpc = d.groupby([ID_COL, "category"], group_keys=False).apply(
                        lambda sub: _window_counts_and_totals(sub, 3600, "1h"), include_groups=False
                    )
                    d["txn_count_last_1h_category"]  = tmpc["txn_count_last_1h"]
                    d["total_amt_last_1h_category"]   = tmpc["total_amt_last_1h"]
                if need_24h_c:
                    tmpc = d.groupby([ID_COL, "category"], group_keys=False).apply(
                        lambda sub: _window_counts_and_totals(sub, 86400, "24h"), include_groups=False
                    )
                    d["txn_count_last_24h_category"] = tmpc["txn_count_last_24h"]
                    d["total_amt_last_24h_category"]  = tmpc["total_amt_last_24h"]

            d.drop(columns=["_ts"], errors="ignore", inplace=True)

    # --- simple encodings ---
    if "gender_bin" in want_cols and "gender" in d.columns:
        d["gender_bin"] = d["gender"].map({"M": 1, "F": 0}).fillna(0).astype(int)

    # drop any alt label names if present
    d.drop(columns=[c for c in ["Class","label","target"] if c in d.columns], errors="ignore", inplace=True)
    return d

def align_numeric(X: pd.DataFrame, expected: list) -> pd.DataFrame:
    """Ensure all expected columns exist and are numeric."""
    X2 = X.copy()
    for c in expected:
        if c not in X2.columns:
            X2[c] = 0.0
    X2 = X2.reindex(columns=expected)
    for c in X2.columns:
        if X2[c].dtype == "O":
            X2[c] = pd.to_numeric(X2[c], errors="coerce").fillna(0.0)
    return X2

# --------------------
# Load data & model
# --------------------
df = pd.read_csv(CSV)
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

# pick calibrated model first
calib = ART / "fraud_lgbm_calibrated.pkl"
base  = ART / "fraud_lgbm_model.pkl"
model_path = calib if calib.exists() else base
model = joblib.load(model_path)

expected, source = get_expected_features(ART)
print(f"Using expected features from: {source} ({len(expected)})")
if len(expected) > 0:
    print("First 10 features:", expected[:10])

# --------------------
# Build features on FULL DF, then split
# --------------------
X_all = build_features(df, set(expected))
y_all = df[LABEL].astype(int).values

mask_test  = df[TIME_COL] >= pd.Timestamp(TEST_START)
mask_train = ~mask_test

X_train, y_train = X_all[mask_train], y_all[mask_train]
X_test,  y_test  = X_all[mask_test],  y_all[mask_test]

X_test = align_numeric(X_test, expected)

print(f"Train={mask_train.sum():,}  Test={mask_test.sum():,}")

# --------------------
# Predict & metrics
# --------------------
y_prob = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_prob)
pr  = average_precision_score(y_test, y_prob)
print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")

# --------------------
# Evaluate at current policy thresholds
# --------------------
policy = json.loads((ART / "operating_policy.json").read_text())
thr_review = float(policy.get("review", 0.60))
thr_block  = float(policy.get("block", 0.90))

def tier(p):
    return "block" if p >= thr_block else ("review" if p >= thr_review else "allow")

pred_tier  = np.array([tier(p) for p in y_prob])
y_pred_pos = (pred_tier != "allow").astype(int)

prec = precision_score(y_test, y_pred_pos, zero_division=0)
rec  = recall_score(y_test, y_pred_pos, zero_division=0)
print(f"At policy thresholds -> Precision={prec:.3f} | Recall={rec:.3f} | Flags={(y_pred_pos.mean()*100):.2f}%")

# --------------------
# Suggest thresholds for target precision levels
# --------------------
target_block_prec  = 0.98
target_review_prec = 0.75

prec_arr, rec_arr, thr = precision_recall_curve(y_test, y_prob)
cand_block  = [t for p, t in zip(prec_arr[:-1], thr) if p >= target_block_prec]
best_block  = max(cand_block) if cand_block else float(thr[-1])
cand_review = [t for p, t in zip(prec_arr[:-1], thr) if (p >= target_review_prec) and (t < best_block)]
best_review = max(cand_review) if cand_review else min(best_block, 0.5)

suggest = {"block": float(best_block), "review": float(best_review), "allow": 0.0}
(OUT / "threshold_suggestion.json").write_text(json.dumps(suggest, indent=2))
print("Wrote:", OUT / "threshold_suggestion.json")
