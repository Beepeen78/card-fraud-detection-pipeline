#!/usr/bin/env python
"""
Prepare data specifically for Power BI consumption.
Enhances the exported CSV files with additional calculated fields.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import os

POWERBI_OUT = Path("powerbi/out")
POWERBI_OUT.mkdir(parents=True, exist_ok=True)

def prepare_transactions_for_powerbi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance transaction data with Power BI-friendly columns.
    """
    df_pbi = df.copy()
    
    # Ensure we have a transaction ID
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
        
        # Add time components
        df_pbi['hour'] = df_pbi['transaction_date'].dt.hour
        df_pbi['day_of_week'] = df_pbi['transaction_date'].dt.dayofweek
        df_pbi['day_name'] = df_pbi['transaction_date'].dt.day_name()
        df_pbi['month'] = df_pbi['transaction_date'].dt.month
        df_pbi['month_name'] = df_pbi['transaction_date'].dt.month_name()
        df_pbi['year'] = df_pbi['transaction_date'].dt.year
        df_pbi['date'] = df_pbi['transaction_date'].dt.date
    
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
    
    # Select columns for Power BI (in order)
    pbi_columns = [
        'transaction_id',
        'transaction_date',
        'date',
        'hour',
        'day_of_week',
        'day_name',
        'month',
        'month_name',
        'year',
        'amt',
        'amount_category',
        'is_high_value',
        'fraud_probability',
        'fraud_prediction',
        'risk_level',
        'city_pop',
        'dist_home_merch',
        'category',
        'score_time'
    ]
    
    # Add ground truth if available
    if 'is_fraud' in df_pbi.columns:
        pbi_columns.insert(-1, 'is_fraud')
    
    # Keep only available columns
    available_columns = [col for col in pbi_columns if col in df_pbi.columns]
    df_pbi = df_pbi[available_columns]
    
    return df_pbi

def prepare_daily_metrics_for_powerbi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create daily aggregated metrics for Power BI.
    """
    if df.empty:
        return pd.DataFrame(columns=['date', 'total_transactions', 'fraud_count', 'fraud_rate', 
                                    'avg_probability', 'total_amount', 'amount_at_risk'])
    
    df_metrics = df.copy()
    
    # Ensure we have date column
    if 'transaction_date' in df_metrics.columns:
        df_metrics['date'] = df_metrics['transaction_date'].dt.date
    elif 'date' not in df_metrics.columns and 'unix_time' in df_metrics.columns:
        df_metrics['date'] = pd.to_datetime(
            pd.to_numeric(df_metrics['unix_time'], errors='coerce'),
            unit='s',
            utc=True
        ).dt.date
    
    # Aggregate by date
    daily = df_metrics.groupby('date').agg({
        'fraud_prediction': ['count', 'sum', 'mean'],
        'fraud_probability': 'mean',
        'amt': ['sum', lambda x: x[df_metrics.loc[x.index, 'fraud_prediction'] == 1].sum()]
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
    
    # Fill missing amount_at_risk with 0
    daily['amount_at_risk'] = daily['amount_at_risk'].fillna(0)
    
    return daily

def export_for_powerbi(transactions_df: pd.DataFrame, output_dir: Path = POWERBI_OUT):
    """
    Export enhanced data for Power BI.
    """
    # Prepare transactions
    tx_pbi = prepare_transactions_for_powerbi(transactions_df)
    tx_path = output_dir / "transactions_scored.csv"
    tx_pbi.to_csv(tx_path, index=False)
    print(f"✅ Exported {len(tx_pbi)} transactions to {tx_path}")
    
    # Prepare daily metrics
    metrics_pbi = prepare_daily_metrics_for_powerbi(transactions_df)
    metrics_path = output_dir / "metrics_daily.csv"
    metrics_pbi.to_csv(metrics_path, index=False)
    print(f"✅ Exported {len(metrics_pbi)} daily metrics to {metrics_path}")
    
    # Create summary stats
    summary = {
        'export_date': datetime.now().isoformat(),
        'total_transactions': len(tx_pbi),
        'fraud_detected': int(tx_pbi['fraud_prediction'].sum()) if 'fraud_prediction' in tx_pbi.columns else 0,
        'fraud_rate': float(tx_pbi['fraud_prediction'].mean()) if 'fraud_prediction' in tx_pbi.columns else 0.0,
        'date_range_start': str(tx_pbi['date'].min()) if 'date' in tx_pbi.columns else None,
        'date_range_end': str(tx_pbi['date'].max()) if 'date' in tx_pbi.columns else None,
    }
    
    summary_path = output_dir / "export_summary.txt"
    with open(summary_path, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    print(f"✅ Exported summary to {summary_path}")
    
    return tx_path, metrics_path

if __name__ == "__main__":
    # Example usage: Load from app output and enhance
    tx_file = POWERBI_OUT / "transactions_scored.csv"
    
    if tx_file.exists():
        print(f"Loading transactions from {tx_file}...")
        df = pd.read_csv(tx_file)
        export_for_powerbi(df)
        print("\n✅ Power BI data preparation complete!")
    else:
        print(f"❌ File not found: {tx_file}")
        print("   Run the app first to generate transaction data.")
