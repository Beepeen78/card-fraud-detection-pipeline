from __future__ import annotations
import os
import pandas as pd

POWERBI_OUT = os.getenv("POWERBI_OUT", "powerbi/out")

def export_powerbi_csvs(scored: pd.DataFrame):
    os.makedirs(POWERBI_OUT, exist_ok=True)
    tx_path = os.path.join(POWERBI_OUT, "transactions_scored.csv")
    m_path  = os.path.join(POWERBI_OUT, "metrics_daily.csv")

    tx_cols = ["transaction_id","customer_id","amount","fraud_probability","fraud_prediction","is_fraud","score_time"]
    avail = [c for c in tx_cols if c in scored.columns]
    scored[avail].to_csv(tx_path, index=False)

    if "score_time" in scored.columns:
        daily = (scored
                 .assign(date=scored["score_time"].dt.floor("D"))
                 .groupby("date")
                 .agg(mean_prob=("fraud_probability","mean"),
                      predicted_fraud=("fraud_prediction","sum"),
                      rows=("fraud_prediction","size"))
                 .reset_index())
        daily.to_csv(m_path, index=False)
    else:
        pd.DataFrame().to_csv(m_path, index=False)

    return tx_path, m_path
