from __future__ import annotations
import joblib

def load_feature_names(model_path: str) -> list[str]:
    pipe = joblib.load(model_path)
    names = None
    if hasattr(pipe, "get_feature_names_out"):
        try:
            names = list(pipe.get_feature_names_out())
        except Exception:
            names = None
    if names is None and hasattr(pipe, "feature_names_in_"):
        names = list(pipe.feature_names_in_)
    if names is None:
        names = [
            "amt","lat","long","city_pop","unix_time","merch_lat","merch_long","merch_zipcode","hour","dayofweek",
            "mean_distance","time_since_last_txn","std_amt","median_amt","max_amt","mean_amt","transaction_count",
            "dist_home_merch","is_weekend","is_business_hours","is_night","dayofyear","month","gender_bin",
            "hour_sin","hour_cos","dow_sin","dow_cos",
            "txn_count_last_1h","total_amt_last_1h","txn_count_last_24h","total_amt_last_24h",
            "txn_count_last_1h_category","total_amt_last_1h_category",
            "txn_count_last_24h_category","total_amt_last_24h_category",
            "te_job","te_dist_category"
        ]
    return [c.split("__")[-1] for c in names]
