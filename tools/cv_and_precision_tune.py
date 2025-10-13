"""Run 5-fold evaluation using the pre-trained pipeline and do precision-focused tuning.
Saves results to tuning_results_precision.json and prints a summary.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / 'dummy_transactions.csv'
MODEL_PATH = ROOT / 'fraud_pipeline.joblib'
OUT_JSON = ROOT / 'tuning_results_precision.json'

# Copy simplified feature builder (must match app.build_features behavior)
from datetime import datetime, timezone
import numpy as np

def first(df, cols):
    for c in cols:
        if c in df.columns: return c
    return None

ID_CANDS   = ["transaction_id","trans_num","id"]
CC_CANDS   = ["cc_num","customer_id","cust_id","user_id"]
AMT_CANDS  = ["amount","amt","transaction_amount"]
TS_CANDS   = ["trans_date_trans_time","timestamp","datetime","transaction_time"]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    id_col  = first(df, ID_CANDS)  or "transaction_id"
    cc_col  = first(df, CC_CANDS)  or "cc_num"
    amt_col = first(df, AMT_CANDS) or "amount"
    ts_col  = first(df, TS_CANDS)  or "trans_date_trans_time"

    if id_col not in df.columns:
        df[id_col] = np.arange(len(df)).astype(str)

    if amt_col in df.columns:
        df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
    else:
        df[amt_col] = 0.0

    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    else:
        ts = pd.to_datetime(pd.Series([datetime.now(timezone.utc)] * len(df)), errors="coerce", utc=True)
    df["unix_time"] = (ts.astype("int64") // 10**9).astype("int64")
    df["hour"]      = ts.dt.hour.fillna(0).astype(int)
    df["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    df["dayofyear"] = ts.dt.dayofyear.fillna(1).astype(int)

    def _num_series(col: str, default: float = 0.0) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
        return pd.Series([default] * len(df), dtype=float)

    lat = _num_series("lat", 0.0)
    lon = _num_series("long", 0.0)
    mlat = _num_series("merch_lat", 0.0)
    mlon = _num_series("merch_long", 0.0)

    df["mean_distance"]   = haversine_km(lat, lon, mlat, mlon)
    df["dist_home_merch"] = df["mean_distance"]

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dayofweek"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dayofweek"]/7)

    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    df["is_night"]   = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_business_hours"] = df["hour"].between(9,17).astype(int)

    df["max_amt"]    = df[amt_col]
    df["median_amt"] = df[amt_col]
    df["std_amt"]    = 0.0
    df["mean_amt"]   = df[amt_col]

    for c in [
        "txn_count_last_1h","txn_count_last_24h","txn_count_last_1h_category",
        "txn_count_last_24h_category","total_amt_last_1h","total_amt_last_24h",
        "total_amt_last_1h_category","total_amt_last_24h_category",
        "time_since_last_txn","transaction_count"
    ]:
        df[c] = 0.0

    if "gender" in df.columns:
        df["gender_bin"] = df["gender"].astype(str).str.lower().map({"m":1,"male":1,"f":0,"female":0}).fillna(0).astype(int)
    else:
        df["gender_bin"] = 0

    df["te_dist_category"] = 0.0
    df["te_job"] = 0.0

    for passthru in ["city_pop","merch_zipcode","month"]:
        if passthru not in df.columns: df[passthru] = 0

    if "amt" not in df.columns: df["amt"] = df[amt_col]

    return df


def compute_heuristic_score(feat: pd.DataFrame) -> pd.Series:
    amt = feat.get("amt", feat.get("amount", 0.0)).astype(float)
    amt_score = np.tanh(amt / (amt.std() + 1e-9) / 2.0).clip(0, 1)
    dist = feat.get("mean_distance", pd.Series([0.0] * len(feat))).astype(float)
    dist_score = (dist / 2000.0).clip(0, 1)
    night = feat.get("is_night", pd.Series([0] * len(feat))).astype(int)
    cat = feat.get("merchant_category", pd.Series([""] * len(feat))).astype(str).str.lower()
    risk_map = {"online": 0.7, "travel": 0.6, "entertainment": 0.5}
    cat_score = cat.map(risk_map).fillna(0.0).astype(float)
    score = 0.45 * amt_score + 0.35 * dist_score + 0.15 * night + 0.05 * cat_score
    return score.clip(0, 1)


def get_expected_features(pipe):
    exp = None
    try:
        if hasattr(pipe, 'named_steps') and 'prep' in pipe.named_steps:
            exp = list(pipe.named_steps['prep'].get_feature_names_out())
    except Exception:
        pass
    if exp is None and hasattr(pipe, 'feature_names_in_'):
        exp = list(pipe.feature_names_in_)
    return exp


def main():
    print('Loading data and model...')
    df = pd.read_csv(DATA_CSV)
    pipe = joblib.load(MODEL_PATH)
    expected = get_expected_features(pipe)
    feat = build_features(df)
    X = feat.reindex(columns=expected, fill_value=0.0) if expected is not None else feat
    model_proba = pipe.predict_proba(X)[:, 1]
    heuristic = compute_heuristic_score(feat).to_numpy()
    y = pd.to_numeric(df.get('is_fraud', 0), errors='coerce').fillna(0).astype(int).to_numpy()

    # Use tuned defaults if available
    tuned = {'heuristic_alpha': 0.0, 'threshold': 0.5}
    tpath = ROOT / 'tuning_results.json'
    if tpath.exists():
        with open(tpath, 'r') as f:
            tuned.update({k: float(v) for k, v in json.load(f).items() if k in tuned})
    alpha_default = float(tuned.get('heuristic_alpha', 0.0))
    thr_default = float(tuned.get('threshold', 0.5))

    # 5-fold stratified evaluation (note: model not retrained; this assesses stability across splits)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        probs = model_proba[test_idx]
        heur = heuristic[test_idx]
        combined = (1.0 - alpha_default) * probs + alpha_default * heur
        preds = (combined >= thr_default).astype(int)
        prec = precision_score(y[test_idx], preds, zero_division=0)
        rec = recall_score(y[test_idx], preds, zero_division=0)
        f1 = f1_score(y[test_idx], preds, zero_division=0)
        try:
            roc = roc_auc_score(y[test_idx], combined)
        except Exception:
            roc = float('nan')
        # PR AUC
        try:
            prec_curve, rec_curve, _ = precision_recall_curve(y[test_idx], combined)
            pr_auc = auc(rec_curve, prec_curve)
        except Exception:
            pr_auc = float('nan')
        metrics.append({'fold': fold, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc, 'pr_auc': pr_auc, 'n': len(test_idx)})

    # summarize
    def summarize(key):
        vals = [m[key] for m in metrics if not (isinstance(m[key], float) and np.isnan(m[key]))]
        return float(np.mean(vals)), float(np.std(vals)) if len(vals)>1 else 0.0

    print('\n5-fold evaluation (using tuned defaults alpha={:.2f}, thr={:.2f}):'.format(alpha_default, thr_default))
    for m in metrics:
        print('fold {fold}: n={n}, prec={precision:.3f}, rec={recall:.3f}, f1={f1:.3f}, roc={roc_auc:.3f}, pr_auc={pr_auc:.3f}'.format(**m))
    mean_prec, std_prec = summarize('precision')
    mean_rec, std_rec = summarize('recall')
    mean_f1, std_f1 = summarize('f1')
    mean_roc, std_roc = summarize('roc_auc')
    mean_pr, std_pr = summarize('pr_auc')
    print('\nMean ± std: precision={:.3f}±{:.3f}, recall={:.3f}±{:.3f}, f1={:.3f}±{:.3f}, roc={:.3f}±{:.3f}, pr_auc={:.3f}±{:.3f}'.format(
        mean_prec, std_prec, mean_rec, std_rec, mean_f1, std_f1, mean_roc, std_roc, mean_pr, std_pr
    ))

    # Grid search for precision (constrained: recall >= min_recall)
    alphas = np.linspace(0,1,21)
    thresholds = np.linspace(0.0, 0.9, 19)
    best_unconstrained = None
    best_constrained = None
    min_recall = 0.20
    for a in alphas:
        combined_all = (1.0 - a) * model_proba + a * heuristic
        for t in thresholds:
            preds_all = (combined_all >= t).astype(int)
            prec = precision_score(y, preds_all, zero_division=0)
            rec = recall_score(y, preds_all, zero_division=0)
            if best_unconstrained is None or prec > best_unconstrained[0]:
                best_unconstrained = (prec, rec, a, t)
            if rec >= min_recall:
                if best_constrained is None or prec > best_constrained[0]:
                    best_constrained = (prec, rec, a, t)

    out = {'5fold_summary': {'mean_precision': mean_prec, 'std_precision': std_prec, 'mean_recall': mean_rec, 'std_recall': std_rec, 'mean_f1': mean_f1, 'std_f1': std_f1, 'mean_roc': mean_roc, 'std_roc': std_roc, 'mean_pr': mean_pr, 'std_pr': std_pr}, 'best_unconstrained': None, 'best_constrained': None}
    if best_unconstrained:
        out['best_unconstrained'] = {'precision': float(best_unconstrained[0]), 'recall': float(best_unconstrained[1]), 'alpha': float(best_unconstrained[2]), 'threshold': float(best_unconstrained[3])}
    if best_constrained:
        out['best_constrained'] = {'precision': float(best_constrained[0]), 'recall': float(best_constrained[1]), 'alpha': float(best_constrained[2]), 'threshold': float(best_constrained[3])}

    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)

    print('\nGrid-search results saved to:', OUT_JSON)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
