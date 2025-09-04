
# upload_to_bigquery.py — Append the app's CSV exports into BigQuery tables
from google.cloud import bigquery
import pandas as pd
from pathlib import Path
import os

PROJECT = os.getenv("BQ_PROJECT", "credit-card-fraud-pipeline")
DATASET = os.getenv("BQ_DATASET", "fraud_prod")
TABLE_TX  = f"{PROJECT}.{DATASET}.transactions_scored"
TABLE_MET = f"{PROJECT}.{DATASET}.metrics_daily"

TX_FILE = Path("powerbi/out/transactions_scored.csv")
MET_FILE = Path("powerbi/out/metrics_daily.csv")

def upload_csv_to_bq(file_path: Path, table: str):
    client = bigquery.Client(project=PROJECT)
    df = pd.read_csv(file_path)
    job = client.load_table_from_dataframe(df, table,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"))
    job.result()
    print(f"✅ Uploaded {len(df)} rows to {table}")

def main():
    if TX_FILE.exists(): upload_csv_to_bq(TX_FILE, TABLE_TX)
    else: print("⚠️ Missing:", TX_FILE)
    if MET_FILE.exists(): upload_csv_to_bq(MET_FILE, TABLE_MET)
    else: print("⚠️ Missing:", MET_FILE)

if __name__ == "__main__":
    main()
