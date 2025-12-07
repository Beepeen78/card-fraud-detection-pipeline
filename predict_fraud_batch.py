# predict_fraud_batch.py  ←  FINAL VERSION THAT ACTUALLY GIVES REAL PROBABILITIES
import os
import joblib
import numpy as np
import pandas as pd

# USE THE CALIBRATED ONE — THIS IS THE ONLY ONE THAT GIVES REAL PROBABILITIES
MODEL_PATH = "fraud_lgbm_calibrated.pkl"   # ←←← THIS ONE, NOT fraud_lgbm_model.pkl !!

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"CRITICAL: {MODEL_PATH} missing! Use fraud_lgbm_calibrated.pkl (the one with calibration)")

model = joblib.load(MODEL_PATH)
print("CALIBRATED model loaded — you will now see real probabilities!")

def score_batch(raw_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        raise ValueError("No data")

    df = raw_df.copy()

    # Same 25 features your model expects
    features = ['amt', 'city_pop', 'dist_home_merch', 'hour', 'dayofweek', 'month', 'dayofyear',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'txn_count_last_1h', 'total_amt_last_1h',
                'txn_count_last_24h', 'total_amt_last_24h',
                'txn_count_last_1h_category', 'total_amt_last_1h_category',
                'txn_count_last_24h_category', 'total_amt_last_24h_category',
                'mean_distance', 'time_since_last_txn', 'mean_amt', 'std_amt',
                'te_job', 'te_dist_category']

    # Time features
    if "unix_time" in df.columns:
        ts = pd.to_datetime(pd.to_numeric(df["unix_time"], errors="coerce"), unit="s", utc=True)
        df["hour"] = ts.dt.hour
        df["dayofweek"] = ts.dt.dayofweek
        df["month"] = ts.dt.month
        df["dayofyear"] = ts.dt.dayofyear

    # Cyclic
    for col, p, s, c in [("hour",24,"hour_sin","hour_cos"), ("dayofweek",7,"dow_sin","dow_cos")]:
        if col in df.columns:
            angle = 2 * np.pi * df[col] / p
            df[s] = np.sin(angle)
            df[c] = np.cos(angle)

    # Fill missing
    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    X = df[sorted(features)]
    prob = model.predict_proba(X)[:, 1]

    df["fraud_probability"] = prob
    df["fraud_prediction"] = (prob >= threshold).astype(int)
    return df