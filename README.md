# Credit Card Fraud – Final Updated Bundle

## How to run
```powershell
# In the folder where app.py is located
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Put your trained model file here:
#   fraud_pipeline.joblib   (next to app.py)

streamlit run app.py
```

### Power BI Export
CSV outputs (row-level + daily) are written to `powerbi/out/`.

### Optional: BigQuery Export
Set environment variables and make sure ADC or a service account is configured.
```powershell
$env:BQ_PROJECT="your-project-id"
$env:BQ_DATASET="fraud_prod"
$env:BQ_TABLE_SCORED="transactions_scored"
$env:BQ_TABLE_METRICS="metrics_daily"
```
Then click **Export to BigQuery (append)** in the app.
