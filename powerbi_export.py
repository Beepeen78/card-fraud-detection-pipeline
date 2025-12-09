"""
Power BI export module for fraud detection pipeline.
Exports transaction data and metrics in Power BI-friendly format.
"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add powerbi directory to path to import prepare_powerbi_data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from powerbi.prepare_powerbi_data import export_for_powerbi, POWERBI_OUT
except ImportError:
    # Fallback if powerbi module structure is different
    POWERBI_OUT = Path("powerbi/out")
    POWERBI_OUT.mkdir(parents=True, exist_ok=True)


def export_powerbi_csvs(result_df: pd.DataFrame, output_dir: Path = None):
    """
    Export fraud detection results to Power BI format.
    
    This function is called by the Gradio app to automatically export
    scored transactions and metrics for Power BI consumption.
    
    Args:
        result_df: DataFrame with fraud detection results (must include
                  'fraud_probability', 'fraud_prediction', 'risk_level')
        output_dir: Optional output directory (defaults to powerbi/out)
    
    Returns:
        tuple: (transactions_path, metrics_path)
    """
    if output_dir is None:
        output_dir = POWERBI_OUT
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use the existing export function from powerbi module
        tx_path, metrics_path = export_for_powerbi(result_df, output_dir)
        return tx_path, metrics_path
    except Exception as e:
        # If the powerbi module isn't available, do a simple export
        print(f"⚠️ Warning: Could not use full Power BI export: {e}")
        print("   Falling back to simple CSV export...")
        
        # Simple fallback export
        tx_path = output_dir / "transactions_scored.csv"
        result_df.to_csv(tx_path, index=False)
        print(f"✅ Exported {len(result_df)} transactions to {tx_path}")
        
        return tx_path, None

