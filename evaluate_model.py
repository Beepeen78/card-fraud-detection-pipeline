# evaluate_model.py — production friendlier
# - Builds features on FULL DF (then splits) to prevent leakage
# - Restores original row order (label/pred alignment)
# - O(n) sliding windows
# - Argparse + portable paths + default policy fallback

import json
import warnings
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------
# Helpers
# --------------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _window_counts_and_totals(sub: pd.DataFrame, window_seconds: int, prefix: str):
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
        tot[i] = a[j : i + 1].sum()
    return pd.DataFrame(
        {f"txn_count_last_{prefix}": cnt, f"total_amt_last_{prefix}": tot},
        index=sub.index,
    )


def build_features(df: pd.DataFrame, want_cols: set, time_col: str, id_col: str) -> pd.DataFrame:
    d = df.copy()
    d["__orig_idx__"] = np.arange(len(d))

    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    sort_cols = [c for c in (id_col, time_col) if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols + ["__orig_idx__"]).reset_index(drop=True)

    # time parts
    if "hour" in want_cols:
        d["hour"] = d[time_col].dt.hour
    if "dayofweek" in want_cols:
        d["dayofweek"] = d[time_col].dt.dayofweek
    if "dayofyear" in want_cols:
        d["dayofyear"] = d[time_col].dt.dayofyear
    if "month" in want_cols:
        d["month"] = d[time_col].dt.month
    if "is_weekend" in want_cols:
        d["is_weekend"] = d["dayofweek"].isin([5, 6]).astype(int)
    if "is_night" in want_cols:
        d["is_night"] = d["hour"].isin(range(0, 6)).astype(int)
    if "is_business_hours" in want_cols:
        d["is_business_hours"] = d["hour"].between(9, 17, inclusive="both").astype(int)
    if "hour_sin" in want_cols:
        d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    if "hour_cos" in want_cols:
        d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    if "dow_sin" in want_cols:
        d["dow_sin"] = np.sin(2 * np.pi * d["dayofweek"] / 7)
    if "dow_cos" in want_cols:
        d["dow_cos"] = np.cos(2 * np.pi * d["dayofweek"] / 7)

    # distances
    if (
        {"lat", "long", "merch_lat", "merch_long"}.issubset(d.columns)
        and "dist_home_merch" in want_cols
    ):
        d["dist_home_merch"] = haversine(
            d["lat"], d["long"], d["merch_lat"], d["merch_long"]
        )

    if "dist_category_bucket_idx" in want_cols and "dist_home_merch" in d.columns:
        bins = [-np.inf, 1, 10, 50, 100, np.inf]
        d["dist_category_bucket_idx"] = pd.cut(
            d["dist_home_merch"].fillna(-1), bins=bins, labels=False
        )

    # per-user rolling
    if id_col in d.columns and time_col in d.columns:
        if "time_since_last_txn" in want_cols:
            d["time_since_last_txn"] = (
                d.groupby(id_col)[time_col].diff().dt.total_seconds().fillna(0)
            )
        if "transaction_count" in want_cols:
            d["transaction_count"] = d.groupby(id_col).cumcount() + 1

        if "amt" in d.columns:
            g_amt = d.groupby(id_col)["amt"]
            if "mean_amt" in want_cols:
                d["mean_amt"] = g_amt.transform(lambda x: x.rolling(50, min_periods=1).mean())
            if "std_amt" in want_cols:
                d["std_amt"] = g_amt.transform(lambda x: x.rolling(50, min_periods=2).std().fillna(0))
            if "median_amt" in want_cols:
                d["median_amt"] = g_amt.transform(lambda x: x.rolling(50, min_periods=1).median())
            if "max_amt" in want_cols:
                d["max_amt"] = g_amt.transform(lambda x: x.rolling(50, min_periods=1).max())

        if "mean_distance" in want_cols and "dist_home_merch" in d.columns:
            d["mean_distance"] = d.groupby(id_col)["dist_home_merch"].transform(
                lambda x: x.rolling(50, min_periods=1).mean()
            )

        if "amt" in d.columns:
            d["_ts"] = d[time_col].astype("int64") // 10**9

            need_1h = {"txn_count_last_1h", "total_amt_last_1h"} & want_cols
            need_24h = {"txn_count_last_24h", "total_amt_last_24h"} & want_cols
            if need_1h:
                tmp = d.groupby(id_col, group_keys=False).apply(
                    lambda sub: _window_counts_and_totals(sub, 3600, "1h"),
                    include_groups=False,
                )
                for c in need_1h:
                    d[c] = tmp[c]
            if need_24h:
                tmp = d.groupby(id_col, group_keys=False).apply(
                    lambda sub: _window_counts_and_totals(sub, 86400, "24h"),
                    include_groups=False,
                )
                for c in need_24h:
                    d[c] = tmp[c]

            d.drop(columns=["_ts"], errors="ignore", inplace=True)

    # simple encodings
    if "gender_bin" in want_cols and "gender" in d.columns:
        d["gender_bin"] = d["gender"].map({"M": 1, "F": 0}).fillna(0).astype(int)

    # restore order
    d = d.sort_values("__orig_idx__").drop(columns="__orig_idx__").reset_index(drop=True)

    d.drop(
        columns=[c for c in ["Class", "label", "target"] if c in d.columns],
        errors="ignore",
        inplace=True,
    )
    return d


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


def get_expected_features(artifacts_dir: Path):
    cols_pkl = artifacts_dir / "feature_columns.pkl"
    cols_from_pkl = None
    if cols_pkl.exists():
        try:
            cols_from_pkl = list(joblib.load(cols_pkl))
        except Exception:
            pass

    model_path = artifacts_dir / "fraud_lgbm_calibrated.pkl"
    if not model_path.exists():
        model_path = artifacts_dir / "fraud_lgbm_model.pkl"
    model = joblib.load(model_path)

    cols_from_model = []
    for accessor in [
        "calibrated_classifiers_[0].estimator.booster_.feature_name",
        "booster_.feature_name",
    ]:
        try:
            names = eval(f"model.{accessor}()")
            if names:
                cols_from_model = list(names)
                break
        except Exception:
            continue

    if cols_from_pkl is not None and cols_from_model:
        if len(cols_from_pkl) != len(cols_from_model):
            warnings.warn(
                f"feature_columns.pkl ({len(cols_from_pkl)}) differs from model ({len(cols_from_model)}). "
                f"Using model feature set to avoid LightGBM shape errors."
            )
            return cols_from_model, "model"
        return cols_from_pkl, "feature_columns.pkl"
    if cols_from_pkl is not None:
        return cols_from_pkl, "feature_columns.pkl"
    if cols_from_model:
        return cols_from_model, "model"

    raise RuntimeError("Cannot determine expected feature names.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path("dataset") / "credit_card_transactions.csv"))
    ap.add_argument("--artifacts_dir", default="notebooks")
    ap.add_argument("--time_col", default="trans_date_trans_time")
    ap.add_argument("--label_col", default="is_fraud")
    ap.add_argument("--id_col", default="cc_num")
    ap.add_argument("--test_start", default="2020-06-01")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parent
    ART = ROOT / args.artifacts_dir
    OUT = ROOT / "eval_out"
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROOT / args.csv) if not Path(args.csv).is_absolute() else pd.read_csv(args.csv)
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")

    # model
    calib = ART / "fraud_lgbm_calibrated.pkl"
    base = ART / "fraud_lgbm_model.pkl"
    model_path = calib if calib.exists() else base
    model = joblib.load(model_path)

    expected, source = get_expected_features(ART)
    print(f"Using expected features from: {source} ({len(expected)})")
    if len(expected) > 0:
        print("First 10 features:", expected[:10])

    # features on full df
    X_all = build_features(df, set(expected), args.time_col, args.id_col)
    y_all = df[args.label_col].astype(int).values

    mask_test = df[args.time_col] >= pd.Timestamp(args.test_start)
    mask_train = ~mask_test

    X_train, y_train = X_all[mask_train], y_all[mask_train]
    X_test, y_test = X_all[mask_test], y_all[mask_test]

    X_test = align_numeric(X_test, expected)

    print(f"Train={mask_train.sum():,}  Test={mask_test.sum():,}")

    # predict & metrics
    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")

    # thresholds
    policy_path = ART / "operating_policy.json"
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())
        thr_review = float(policy.get("review", 0.60))
        thr_block = float(policy.get("block", 0.90))
    else:
        thr_review = 0.60
        thr_block = 0.90
        print("⚠️ operating_policy.json not found; using defaults review=0.60, block=0.90")

    def tier(p):
        return "block" if p >= thr_block else ("review" if p >= thr_review else "allow")

    pred_tier = np.array([tier(p) for p in y_prob])
    y_pred_pos = (pred_tier != "allow").astype(int)

    prec = precision_score(y_test, y_pred_pos, zero_division=0)
    rec = recall_score(y_test, y_pred_pos, zero_division=0)
    print(f"At policy thresholds -> Precision={prec:.3f} | Recall={rec:.3f} | Flags={(y_pred_pos.mean()*100):.2f}%")

    # suggest thresholds
    target_block_prec = 0.98
    target_review_prec = 0.75

    prec_arr, rec_arr, thr = precision_recall_curve(y_test, y_prob)
    thr = np.asarray(thr)
    prec_arr = np.asarray(prec_arr)

    cand_block = [t for p, t in zip(prec_arr[:-1], thr) if p >= target_block_prec]
    best_block = max(cand_block) if cand_block else (float(thr[-1]) if len(thr) else 0.90)

    cand_review = [t for p, t in zip(prec_arr[:-1], thr) if (p >= target_review_prec) and (t < best_block)]
    best_review = max(cand_review) if cand_review else min(best_block, 0.50)

    suggest = {"block": float(best_block), "review": float(best_review), "allow": 0.0}
    (OUT / "threshold_suggestion.json").write_text(json.dumps(suggest, indent=2))
    print("Wrote:", OUT / "threshold_suggestion.json")


if __name__ == "__main__":
    main()
