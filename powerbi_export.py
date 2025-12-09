from __future__ import annotations
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

POWERBI_OUT = os.getenv("POWERBI_OUT", "powerbi/out")

def export_powerbi_csvs(scored: pd.DataFrame):
    """
    Export enhanced data for Power BI with additional calculated fields.
    """
    os.makedirs(POWERBI_OUT, exist_ok=True)
    tx_path = os.path.join(POWERBI_OUT, "transactions_scored.csv")
    m_path = os.path.join(POWERBI_OUT, "metrics_daily.csv")
    
    # Prepare enhanced transaction data
    df_pbi = scored.copy()
    
    # Add transaction ID if missing
    if 'transaction_id' not in df_pbi.columns:
        df_pbi['transaction_id'] = range(1, len(df_pbi) + 1)
    
    # Convert unix_time to datetime if present
    if 'unix_time' in df_pbi.columns:
        df_pbi['transaction_date'] = pd.to_datetime(
            pd.to_numeric(df_pbi['unix_time'], errors='coerce'),
            unit='s',
            utc=True
        )
        df_pbi['transaction_date'] = df_pbi['transaction_date'].dt.tz_localize(None)
        df_pbi['date'] = df_pbi['transaction_date'].dt.date
        df_pbi['hour'] = df_pbi['transaction_date'].dt.hour
        df_pbi['day_of_week'] = df_pbi['transaction_date'].dt.dayofweek
        df_pbi['day_name'] = df_pbi['transaction_date'].dt.day_name()
        df_pbi['month'] = df_pbi['transaction_date'].dt.month
        df_pbi['month_name'] = df_pbi['transaction_date'].dt.month_name()
        df_pbi['year'] = df_pbi['transaction_date'].dt.year
    
    # Ensure risk_level exists
    if 'risk_level' not in df_pbi.columns:
        conditions = [
            df_pbi['fraud_probability'] < 0.1,
            df_pbi['fraud_probability'] < 0.5,
            df_pbi['fraud_probability'] < 0.9,
        ]
        choices = ['Low', 'Medium', 'High']
        df_pbi['risk_level'] = np.select(conditions, choices, default='Critical')
    
    # Add amount categories
    if 'amt' in df_pbi.columns:
        df_pbi['amount_category'] = pd.cut(
            df_pbi['amt'],
            bins=[0, 50, 200, 500, 1000, float('inf')],
            labels=['<$50', '$50-$200', '$200-$500', '$500-$1000', '>$1000']
        )
        df_pbi['is_high_value'] = (df_pbi['amt'] > 500).astype(int)
    
    # Add score time if not present
    if 'score_time' not in df_pbi.columns:
        df_pbi['score_time'] = datetime.now()
    
    # Select columns for Power BI (prioritize important ones)
    priority_cols = [
        'transaction_id', 'transaction_date', 'date', 'hour', 'day_of_week', 'day_name',
        'month', 'month_name', 'year', 'amt', 'amount_category', 'is_high_value',
        'fraud_probability', 'fraud_prediction', 'risk_level',
        'city_pop', 'dist_home_merch', 'category', 'score_time'
    ]
    
    # Add ground truth if available
    if 'is_fraud' in df_pbi.columns:
        priority_cols.insert(-1, 'is_fraud')
    
    # Keep available columns + any other columns from original
    available_priority = [col for col in priority_cols if col in df_pbi.columns]
    other_cols = [col for col in df_pbi.columns if col not in priority_cols]
    df_pbi = df_pbi[available_priority + other_cols]
    
    # Save transactions
    df_pbi.to_csv(tx_path, index=False)
    
    # Create daily metrics
    if 'date' in df_pbi.columns or 'transaction_date' in df_pbi.columns:
        date_col = 'date' if 'date' in df_pbi.columns else 'transaction_date'
        if date_col == 'transaction_date':
            df_pbi['date'] = df_pbi['transaction_date'].dt.date
        
        daily = df_pbi.groupby('date').agg({
            'fraud_prediction': ['count', 'sum', 'mean'],
            'fraud_probability': 'mean',
            'amt': ['sum', lambda x: x[df_pbi.loc[x.index, 'fraud_prediction'] == 1].sum() if 'fraud_prediction' in df_pbi.columns else 0]
        }).reset_index()
        
        # Flatten column names
        daily.columns = [
            'date',
            'total_transactions',
            'fraud_count',
            'fraud_rate',
            'avg_probability',
            'total_amount',
            'amount_at_risk'
        ]
        
        daily['amount_at_risk'] = daily['amount_at_risk'].fillna(0)
        daily.to_csv(m_path, index=False)
    else:
        # Fallback to old method
        if "score_time" in df_pbi.columns:
            daily = (df_pbi
                     .assign(date=df_pbi["score_time"].dt.floor("D"))
                     .groupby("date")
                     .agg(mean_prob=("fraud_probability", "mean"),
                          predicted_fraud=("fraud_prediction", "sum"),
                          rows=("fraud_prediction", "size"))
                     .reset_index())
            daily.to_csv(m_path, index=False)
        else:
            pd.DataFrame().to_csv(m_path, index=False)
    
    print(f"✅ Exported {len(df_pbi)} transactions to {tx_path}")
    print(f"✅ Exported daily metrics to {m_path}")
    
    return tx_path, m_path
