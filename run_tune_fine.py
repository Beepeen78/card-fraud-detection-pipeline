import sys
# prevent Streamlit UI from running on import
if "--csv" not in sys.argv:
    sys.argv.insert(1, "--csv")

from app import tune_on_csv
import numpy as np

ths = list(np.linspace(0.3, 0.5, 21))
alphas = list(np.linspace(0.4, 0.6, 21))
best, results = tune_on_csv(r'd:\\cursor\\card-fraud-detection-pipeline\\dummy_transactions.csv', search_thresholds=ths, search_alphas=alphas)
print('BEST:', best)
res_sorted = sorted(results, key=lambda r: r[2], reverse=True)
for r in res_sorted[:10]:
    print(r)
