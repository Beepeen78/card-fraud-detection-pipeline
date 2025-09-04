
# test_bq.py — quick connectivity test
from google.cloud import bigquery

client = bigquery.Client(project="credit-card-fraud-pipeline")
datasets = list(client.list_datasets())
print("Datasets in project:", [d.dataset_id for d in datasets])

tables = list(client.list_tables("fraud_prod"))
print("Tables in fraud_prod:", [t.table_id for t in tables])
