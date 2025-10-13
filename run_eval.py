import sys
# Ensure app is imported in CLI mode (so it won't build the Streamlit UI on import)
if "--csv" not in sys.argv:
	sys.argv.insert(1, "--csv")

from app import generate_evaluation_report
import pandas as pd

raw = pd.read_csv(r'd:\\cursor\\card-fraud-detection-pipeline\\dummy_transactions.csv')
scored = pd.read_csv(r'd:\\cursor\\card-fraud-detection-pipeline\\powerbi\\out\\transactions_scored.csv')
metrics = generate_evaluation_report(raw, scored, out_dir=r'd:\\cursor\\card-fraud-detection-pipeline\\eval_out')
print(metrics)
