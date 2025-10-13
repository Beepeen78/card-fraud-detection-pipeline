import sys
# Ensure app is imported in CLI mode (so it won't build the Streamlit UI on import)
if "--csv" not in sys.argv:
    sys.argv.insert(1, "--csv")

from app import tune_on_csv

best, results = tune_on_csv(r'd:\\cursor\\card-fraud-detection-pipeline\\dummy_transactions.csv')
print('BEST:', best)
res_sorted = sorted(results, key=lambda r: r[2], reverse=True)
for r in res_sorted[:5]:
    print(r)
