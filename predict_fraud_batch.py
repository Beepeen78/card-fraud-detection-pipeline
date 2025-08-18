#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_fraud_batch.py  (auto-feature version)

Usage:
  python predict_fraud_batch.py --input dataset/credit_card_transactions.csv \
    --output eval_out/predictions_calibrated.csv --artifacts_dir notebooks [--id_col cc_num]
"""

import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

DEFAULT_POLICY = {"block":0.90, "review":0.60}

# --- CONFIG: adjust if your column names differ ---
ID_COL   = "cc_num"
TIME_COL = "trans_date_trans_time"
AMT_COL  = "amt"

# ---------- helpers ----------
def load_policy(path: Path):
    if path.exists():
        try: return json.loads(path.read_text())
        except Exception: pass
    return DEFAULT_POLICY

def haversine(lat1, lon1, lat2, lon2):
    lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
    dlat = lat2-lat1; dlon = lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2*np.arcsin(np.sqrt(a))  # km

def build_features(df: pd.DataFrame, want_cols: set) -> pd.DataFrame:
    """Build only the features that the model expects (subset-safe)."""
    d = df.copy()
    # clean base
    if TIME_COL in d.columns:
        d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d = d.sort_values([c for c in [ID_COL, TIME_COL] if c in d.columns]).reset_index(drop=True)

    # time parts
    if "hour" in want_cols and TIME_COL in d.columns: d["hour"] = d[TIME_COL].dt.hour
    if "dayofweek" in want_cols and TIME_COL in d.columns: d["dayofweek"] = d[TIME_COL].dt.dayofweek
    if "dayofyear" in want_cols and TIME_COL in d.columns: d["dayofyear"] = d[TIME_COL].dt.dayofyear
    if "month" in want_cols and TIME_COL in d.columns: d["month"] = d[TIME_COL].dt.month
    if "is_weekend" in want_cols and "dayofweek" in d.columns: d["is_weekend"] = d["dayofweek"].isin([5,6]).astype(int)
    if "is_night" in want_cols and "hour" in d.columns: d["is_night"] = d["hour"].isin(range(0,6)).astype(int)
    if "is_business_hours" in want_cols and "hour" in d.columns: d["is_business_hours"] = d["hour"].between(9,17, inclusive="both").astype(int)
    if "hour_sin" in want_cols and "hour" in d.columns: d["hour_sin"] = np.sin(2*np.pi*d["hour"]/24)
    if "hour_cos" in want_cols and "hour" in d.columns: d["hour_cos"] = np.cos(2*np.pi*d["hour"]/24)
    if "dow_sin" in want_cols and "dayofweek" in d.columns: d["dow_sin"] = np.sin(2*np.pi*d["dayofweek"]/7)
    if "dow_cos" in want_cols and "dayofweek" in d.columns: d["dow_cos"] = np.cos(2*np.pi*d["dayofweek"]/7)

    # distance home->merchant
    if "dist_home_merch" in want_cols and {"lat","long","merch_lat","merch_long"}.issubset(d.columns):
        d["dist_home_merch"] = haversine(d["lat"], d["long"], d["merch_lat"], d["merch_long"])

    # rolling by user
    if ID_COL in d.columns and TIME_COL in d.columns:
        # time since last txn
        if "time_since_last_txn" in want_cols:
            d["time_since_last_txn"] = d.groupby(ID_COL)[TIME_COL].diff().dt.total_seconds().fillna(0)
        # tx count
        if "transaction_count" in want_cols:
            d["transaction_count"] = d.groupby(ID_COL).cumcount()+1
        # amount rolling stats
        if AMT_COL in d.columns:
            g_amt = d.groupby(ID_COL)[AMT_COL]
            if "mean_amt" in want_cols:   d["mean_amt"]   = g_amt.transform(lambda x: x.rolling(50, min_periods=1).mean())
            if "std_amt" in want_cols:    d["std_amt"]    = g_amt.transform(lambda x: x.rolling(50, min_periods=2).std().fillna(0))
            if "median_amt" in want_cols: d["median_amt"] = g_amt.transform(lambda x: x.rolling(50, min_periods=1).median())
            if "max_amt" in want_cols:    d["max_amt"]    = g_amt.transform(lambda x: x.rolling(50, min_periods=1).max())
        # mean distance
        if "mean_distance" in want_cols and "dist_home_merch" in d.columns:
            d["mean_distance"] = d.groupby(ID_COL)["dist_home_merch"].transform(lambda x: x.rolling(50, min_periods=1).mean())
        # window counts/sums
        if AMT_COL in d.columns:
            d["_ts"] = d[TIME_COL].astype("int64") // 10**9
            def window(sub, sec, prefix):
                t = sub["_ts"].to_numpy(); a = sub[AMT_COL].to_numpy(); n=len(sub)
                cnt = np.zeros(n, dtype=np.int32); tot=np.zeros(n, dtype=float); j=0
                for i in range(n):
                    while t[i]-t[j] > sec: j += 1
                    cnt[i] = i-j+1; tot[i]= a[j:i+1].sum()
                return pd.DataFrame({f"txn_count_last_{prefix}":cnt, f"total_amt_last_{prefix}":tot}, index=sub.index)
            if {"txn_count_last_1h","total_amt_last_1h"} & want_cols:
                tmp = d.groupby(ID_COL, group_keys=False).apply(lambda sub: window(sub,3600,"1h"), include_groups=False)
                d[[c for c in ["txn_count_last_1h","total_amt_last_1h"] if c in want_cols]] = tmp[[c for c in ["txn_count_last_1h","total_amt_last_1h"] if c in tmp.columns]]
            if {"txn_count_last_24h","total_amt_last_24h"} & want_cols:
                tmp = d.groupby(ID_COL, group_keys=False).apply(lambda sub: window(sub,86400,"24h"), include_groups=False)
                d[[c for c in ["txn_count_last_24h","total_amt_last_24h"] if c in want_cols]] = tmp[[c for c in ["txn_count_last_24h","total_amt_last_24h"] if c in tmp.columns]]
            d.drop(columns=["_ts"], errors="ignore", inplace=True)

    # simple encodings (safe without labels)
    if "gender_bin" in want_cols and "gender" in d.columns:
        d["gender_bin"] = d["gender"].map({"M":1,"F":0}).fillna(0).astype(int)

    # distance bucket index (no label needed)
    if "dist_category_bucket_idx" in want_cols and "dist_home_merch" in d.columns:
        bins = [-np.inf, 1, 10, 50, 100, np.inf]
        d["dist_category_bucket_idx"] = pd.cut(d["dist_home_merch"].fillna(-1), bins=bins, labels=False)

    # drop obvious labels if present
    d.drop(columns=[c for c in ["is_fraud","label","target","Class"] if c in d.columns], errors="ignore", inplace=True)
    return d

def get_model_expected_names(model) -> list:
    for accessor in [
        "calibrated_classifiers_[0].estimator.booster_.feature_name",
        "booster_.feature_name",
    ]:
        try:
            names = eval(f"model.{accessor}()")
            if names: return list(names)
        except Exception: continue
    return []

def align_numeric(X: pd.DataFrame, expected: list) -> pd.DataFrame:
    X2 = X.copy()
    for c in expected:
        if c not in X2.columns: X2[c] = 0.0
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
    ap.add_argument("--id_col", default=None)
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    policy = load_policy(art / "operating_policy.json")

    pipe_path  = art / "fraud_pipeline.joblib"
    calib_path = art / "fraud_lgbm_calibrated.pkl"
    base_path  = art / "fraud_lgbm_model.pkl"

    df_raw = pd.read_csv(args.input)

    used = None
    if pipe_path.exists():
        # Try pipeline first; if it complains about missing columns, we’ll build them.
        pipe = joblib.load(pipe_path)
        try:
            X_try = df_raw.drop(columns=[c for c in ["is_fraud","label","target","Class"] if c in df_raw.columns], errors="ignore")
            y_prob = pipe.predict_proba(X_try)[:,1]
            used = "pipeline"
        except Exception as e:
            # build features the pipeline likely expects, then retry
            # we don't know the pipeline's exact list, so build a superset commonly used
            common_cols = {
                "amt","city_pop","dayofweek","dayofyear","dist_category_bucket_idx","dist_home_merch",
                "dow_cos","dow_sin","hour","hour_cos","hour_sin","is_business_hours","is_night","is_weekend",
                "max_amt","mean_amt","median_amt","month","std_amt","time_since_last_txn",
                "total_amt_last_1h","total_amt_last_24h","transaction_count","txn_count_last_1h","txn_count_last_24h",
                # if your pipeline used the 38-col schema, many of these will get built too:
                "mean_distance","txn_count_last_1h_category","total_amt_last_1h_category",
                "txn_count_last_24h_category","total_amt_last_24h_category","gender_bin"
            }
            Xb = build_features(df_raw, common_cols)
            y_prob = pipe.predict_proba(Xb)[:,1]
            used = "pipeline(features)"
    else:
        # No pipeline → use calibrated/base model; compute only the features it expects
        model = joblib.load(calib_path) if calib_path.exists() else joblib.load(base_path)
        expected = get_model_expected_names(model)
        if not expected:
            cols_pkl = art / "feature_columns.pkl"
            if cols_pkl.exists():
                expected = list(joblib.load(cols_pkl))
        if not expected:
            raise RuntimeError("Cannot determine expected feature names. Provide feature_columns.pkl or a model with stored names.")
        Xb = build_features(df_raw, set(expected))
        Xb = align_numeric(Xb, expected)
        y_prob = model.predict_proba(Xb)[:,1]
        used = "calibrated" if calib_path.exists() else "base"

    def tier(p): return "block" if p >= policy["block"] else ("review" if p >= policy["review"] else "allow")
    out = pd.DataFrame({"fraud_probability": y_prob, "decision": [tier(float(p)) for p in y_prob]})
    if args.id_col and args.id_col in df_raw.columns:
        out.insert(0, args.id_col, df_raw[args.id_col].values)

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    vc = out["decision"].value_counts().to_dict()
    rate = (out["decision"]!="allow").mean()
    print(f"Scored {len(out)} rows using {used}. Policy: review>={policy['review']}, block>={policy['block']}")
    print(f"Tier counts: {vc} | % flagged: {rate:.2%}")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
