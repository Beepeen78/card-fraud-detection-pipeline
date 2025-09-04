# Credit Card Fraud – Streamlit + BigQuery Export (Drop-in)

## Setup
```bash
pip install -r requirements.txt
# Windows: ensure ADC is set up
# gcloud auth application-default login
set GOOGLE_APPLICATION_CREDENTIALS=%APPDATA%\gcloud\application_default_credentials.json
```

## Run
```bash
streamlit run app.py
```

- Upload a raw CSV (id, time, amount, coords…).
- Click **Export** to save CSVs under `powerbi/out/`.
- Click **Export to BigQuery (append)** to push to:
  - `fraud_prod.transactions_scored`
  - `fraud_prod.metrics_daily`

> If your model requires engineered features and your joblib doesn’t include preprocessing, this app will align missing columns to zeros as a **fallback**. For best results, retrain and save a Pipeline with preprocessing (preferred).