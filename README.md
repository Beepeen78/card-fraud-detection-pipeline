# Credit Card Fraud – Streamlit + BigQuery + Snowflake Export

[![CI](https://github.com/Beepeen78/card-fraud-detection-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Beepeen78/card-fraud-detection-pipeline/actions/workflows/ci.yml)

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Data Warehouses

#### BigQuery Setup
```bash
# Windows: ensure ADC is set up
gcloud auth application-default login
set GOOGLE_APPLICATION_CREDENTIALS=%APPDATA%\gcloud\application_default_credentials.json
```

#### Snowflake Setup
Set the following environment variables:
```bash
# Required
set SNOWFLAKE_ACCOUNT=your_account_identifier
set SNOWFLAKE_USER=your_username
set SNOWFLAKE_PASSWORD=your_password

# Optional (defaults provided)
set SNOWFLAKE_WAREHOUSE=COMPUTE_WH
set SNOWFLAKE_DATABASE=FRAUD_DETECTION
set SNOWFLAKE_SCHEMA=PRODUCTION
set SNOWFLAKE_ROLE=PUBLIC
```

### 3. Initialize Snowflake Database
```bash
# Test connection first
python setup_snowflake.py test

# Set up database, tables, and views
python setup_snowflake.py
```

## Run
```bash
streamlit run app.py
```

## Features
- Upload a raw CSV (id, time, amount, coords…)
- Real-time fraud scoring with configurable thresholds
- Export to multiple destinations:
  - **CSV files** under `powerbi/out/`
  - **BigQuery** tables:
    - `fraud_prod.transactions_scored`
    - `fraud_prod.metrics_daily`
  - **Snowflake** tables:
    - `FRAUD_DETECTION.PRODUCTION.TRANSACTIONS_SCORED`
    - `FRAUD_DETECTION.PRODUCTION.METRICS_DAILY`

## Snowflake Analytics Views
The setup creates useful views for analysis:
- `HIGH_RISK_TRANSACTIONS`: Filtered view of high-risk transactions with risk levels
- `DAILY_FRAUD_SUMMARY`: Daily aggregated metrics with flag rates and recall

## Notes
> If your model requires engineered features and your joblib doesn't include preprocessing, this app will align missing columns to zeros as a **fallback**. For best results, retrain and save a Pipeline with preprocessing (preferred).

## Quick CLI usage

Score a CSV with the pipeline (CLI mode):

```powershell
python app.py --csv dummy_transactions.csv --out-dir powerbi/out --threshold 0.37 --heuristic-alpha 0.46
```

Generate evaluation artifacts (PR/ROC, metrics):

```powershell
python run_eval.py
```
