import subprocess
import sys
import os
import pandas as pd

# A tiny smoke test: load model and run scorer on 10 rows
def test_load_model_and_score(tmp_path):
    # ensure we can import joblib and load the pipeline
    import joblib
    p = os.path.join(os.getcwd(), 'fraud_pipeline.joblib')
    pipe = joblib.load(p)
    assert pipe is not None

    # Prepare a small CSV (use first 10 rows of dummy if available)
    df = pd.read_csv('dummy_transactions.csv').head(10)
    small = tmp_path / 'small.csv'
    df.to_csv(small, index=False)

    # Run CLI scorer
    ret = subprocess.run([sys.executable, 'app.py', '--csv', str(small), '--out-dir', str(tmp_path), '--threshold', '0.5'], capture_output=True, text=True)
    print(ret.stdout)
    assert ret.returncode == 0
    scored = pd.read_csv(tmp_path / 'transactions_scored.csv')
    assert 'fraud_probability' in scored.columns
