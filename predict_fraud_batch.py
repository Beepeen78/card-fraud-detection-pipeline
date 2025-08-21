#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_fraud_batch.py  (auto-feature + safe alignment)

Usage (Windows PowerShell):
  python predict_fraud_batch.py `
    --input "dataset/credit_card_transactions.csv" `
    --output "eval_out/predictions_calibrated.csv" `
    --artifacts_dir "notebooks" `
    --id_col trans_num
"""

import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

DEFAULT_POLICY = {"block": 0.90, "review": 0.60}

# --- CONFIG (adjust if needed) ---
ID_COL   = "cc_num"                   # default ID column used for rolling features
TIME_COL = "trans_date_trans_time"
AMT_COL  = "amt"

# ---------- small helpers ----------
def load_policy(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return DEFAULT_POLICY

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))  # km

def _group_apply(df, key, func):
    """pandas compatibility: try include_groups=False if available."""
    try:
        return df.groupby(key, group_keys=False).apply(func, include_groups=False)
    except TypeError:
        return df.groupby(key, group_keys=False).apply(func)

# ---------- feature builder ----------
def build_features(df: pd.DataFrame, want_cols: set) -> pd.DataFrame:
    """Build ONLY the features present in want_cols (subset-safe)."""
    d = df.copy()
    if TIME_COL in d.columns:
        d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d = d.sort_values([c for c in [ID_COL, TIME_COL] if c in d.columns]).reset_index(drop=True)

    # time parts
    if "hour" in want_cols and TIME_COL in d.columns:         d["hour"] = d[TIME_COL].dt.hour
    if "dayofweek" in want_cols and TIME_COL in d.columns:    d["dayofweek"] = d[TIME_COL].dt.dayofweek
    if "dayofyear" in want_cols and TIME_COL in d.columns:    d["dayofyear"] = d[TIME_COL].dt.dayofyear
    if "month" in want_cols and TIME_COL in d.columns:        d["month"] = d[TIME_COL].dt.month
    if "is_weekend" in want_cols and "dayofweek" in d.columns:
        d["is_weekend"] = d["dayofweek"].isin([5, 6]).astype(int)
    if "is_night" in want_cols and "hour" in d.columns:
        d["is_night"] = d["hour"].isin(range(0, 6)).astype(int)
    if "is_business_hours" in want_cols and "hour" in d.columns:
        d["is_business_hours"] = d["hour"].between(9, 17, inclusive="both").astype(int)
    if "hour_sin" in want_cols and "hour" in d.columns:       d["hour_sin"] = np.sin(2*np.pi*d["hour"]/24)
    if "hour_cos" in want_cols and "hour" in d.columns:       d["hour_cos"] = np.cos(2*np.pi*d["hour"]/24)
    if "dow_sin" in want_cols and "dayofweek" in d.columns:   d["dow_sin"] = np.sin(2*np.pi*d["dayofweek"]/7)
    if "dow_cos" in want_cols and "dayofweek" in d.columns:   d["dow_cos"] = np.cos(2*np.pi*d["dayofweek"]/7)

    # distance home->merchant
    if "dist_home_merch" in want_cols and {"lat","long","merch_lat","merch_long"}.issubset(d.columns):
        d["dist_home_merch"] = haversine(d["lat"], d["long"], d["merch_lat"], d["merch_long"])

    # user-level rolling features
    if ID_COL in d.columns and TIME_COL in d.columns:
        if "time_since_last_txn" in want_cols:
            d["time_since_last_txn"] = d.groupby(ID_COL)[TIME_COL].diff().dt.total_seconds().fillna(0)
        if "transaction_count" in want_cols:
            d["transaction_count"] = d.groupby(ID_COL).cumcount() + 1
        if AMT_COL in d.columns:
            g_amt = d.groupby(ID_COL)[AMT_COL]
            if "mean_amt" in want_cols:   d["mean_amt"]   = g_amt.transform(lambda x: x.rolling(50, min_periods=1).mean())
            if "std_amt" in want_cols:    d["std_amt"]    = g_amt.transform(lambda x: x.rolling(50, min_periods=2).std().fillna(0))
            if "median_amt" in want_cols: d["median_amt"] = g_amt.transform(lambda x: x.rolling(50, min_periods=1).median())
            if "max_amt" in want_cols:    d["max_amt"]    = g_amt.transform(lambda x: x.rolling(50, min_periods=1).max())
        if "mean_distance" in want_cols and "dist_home_merch" in d.columns:
            d["mean_distance"] = d.groupby(ID_COL)["dist_home_merch"].transform(lambda x: x.rolling(50, min_periods=1).mean())
        if AMT_COL in d.columns:
            d["_ts"] = d[TIME_COL].astype("int64") // 10**9
            def window(sub, sec, prefix):
                t = sub["_ts"].to_numpy(); a = sub[AMT_COL].to_numpy(); n = len(sub)
                cnt = np.zeros(n, dtype=np.int32); tot = np.zeros(n, dtype=float); j = 0
                for i in range(n):
                    while t[i] - t[j] > sec: j += 1
                    cnt[i] = i - j + 1; tot[i] = a[j:i+1].sum()
                return pd.DataFrame({f"txn_count_last_{prefix}": cnt,
                                     f"total_amt_last_{prefix}": tot}, index=sub.index)
            need1  = {"txn_count_last_1h",  "total_amt_last_1h"}  & want_cols
            need24 = {"txn_count_last_24h", "total_amt_last_24h"} & want_cols
            if need1:
                tmp = _group_apply(d, ID_COL, lambda sub: window(sub, 3600, "1h"))
                for c in need1:  d[c] = tmp[c]
            if need24:
                tmp = _group_apply(d, ID_COL, lambda sub: window(sub, 86400, "24h"))
                for c in need24: d[c] = tmp[c]
            d.drop(columns=["_ts"], errors="ignore", inplace=True)

    # simple encodings
    if "gender_bin" in want_cols and "gender" in d.columns:
        d["gender_bin"] = d["gender"].map({"M": 1, "F": 0}).fillna(0).astype(int)

    # distance bucket
    if "dist_category_bucket_idx" in want_cols and "dist_home_merch" in d.columns:
        bins = [-np.inf, 1, 10, 50, 100, np.inf]
        d["dist_category_bucket_idx"] = pd.cut(d["dist_home_merch"].fillna(-1), bins=bins, labels=False)

    # drop obvious labels if present
    d.drop(columns=[c for c in ["is_fraud","label","target","Class"] if c in d.columns],
           errors="ignore", inplace=True)
    return d

# ---------- column alignment ----------
def expected_from_model(m):
    for accessor in [
        "calibrated_classifiers_[0].estimator.booster_.feature_name",
        "booster_.feature_name",
    ]:
        try:
            names = eval(f"m.{accessor}()")
            if names:
                return list(names)
        except Exception:
            pass
    return []

def align_numeric(X: pd.DataFrame, expected: list) -> pd.DataFrame:
    X2 = X.copy()
    for c in expected:
        if c not in X2.columns:
            X2[c] = 0.0
    X2 = X2.reindex(columns=expected)
    for c in X2.columns:
        if X2[c].dtype == "O":
            X2[c] = pd.to_numeric(X2[c], errors="coerce").fillna(0.0)
    return X2

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--artifacts_dir", default="notebooks")
    ap.add_argument("--id_col", default=None)  # optional: include an ID column in the output
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    policy = load_policy(art / "operating_policy.json")

    pipe_path  = art / "fraud_pipeline.joblib"
    calib_path = art / "fraud_lgbm_calibrated.pkl"
    base_path  = art / "fraud_lgbm_model.pkl"
    cols_pkl   = art / "feature_columns.pkl"

    df_raw = pd.read_csv(args.input)

    used = None
    y_prob = None

    if pipe_path.exists():
        # Try pipeline directly on raw columns
        pipe = joblib.load(pipe_path)
        try:
            X_try = df_raw.drop(columns=[c for c in ["is_fraud","label","target","Class"] if c in df_raw.columns],
                                errors="ignore")
            y_prob = pipe.predict_proba(X_try)[:, 1]
            used = "pipeline"
        except Exception:
            # Build a broad feature set and try again (covers both 25-col and 38-col schemas)
            common_cols = {
                "amt","city_pop","dayofweek","dayofyear","dist_category_bucket_idx","dist_home_merch",
                "dow_cos","dow_sin","hour","hour_cos","hour_sin","is_business_hours","is_night","is_weekend",
                "max_amt","mean_amt","median_amt","month","std_amt","time_since_last_txn",
                "total_amt_last_1h","total_amt_last_24h","transaction_count","txn_count_last_1h","txn_count_last_24h",
                "mean_distance","gender_bin"
            }
            Xb = build_features(df_raw, common_cols)
            y_prob = pipe.predict_proba(Xb)[:, 1]
            used = "pipeline(features)"

    if y_prob is None:
        # Use model directly with strict column alignment
        model = joblib.load(calib_path) if calib_path.exists() else joblib.load(base_path)
        expected = expected_from_model(model)
        if not expected and cols_pkl.exists():
            expected = list(joblib.load(cols_pkl))
        if not expected:
            raise RuntimeError("Cannot determine expected feature names (model & feature_columns.pkl missing).")

        Xb = build_features(df_raw, set(expected))
        Xb = align_numeric(Xb, expected)

        # sanity log
        if len(expected) != Xb.shape[1]:
            print(f"WARNING: expected {len(expected)} features, got {Xb.shape[1]}")
        missing = [c for c in expected if c not in Xb.columns]
        extra   = [c for c in Xb.columns if c not in expected]
        if missing or extra:
            print("Missing (filled):", missing[:5], " Extra (dropped):", extra[:5])

        y_prob = model.predict_proba(Xb)[:, 1]
        used = "calibrated" if calib_path.exists() else "base"

    def tier(p): 
        return "block" if p >= policy["block"] else ("review" if p >= policy["review"] else "allow")

    out = pd.DataFrame({"fraud_probability": y_prob,
                        "decision": [tier(float(p)) for p in y_prob]})
    if args.id_col and args.id_col in df_raw.columns:
        out.insert(0, args.id_col, df_raw[args.id_col].values)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    vc = out["decision"].value_counts().to_dict()
    rate = (out["decision"] != "allow").mean()
    print(f"Scored {len(out):,} rows using {used}. Policy: review>={policy['review']}, block>={policy['block']}")
    print(f"Tier counts: {vc} | % flagged: {rate:.2%}")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
