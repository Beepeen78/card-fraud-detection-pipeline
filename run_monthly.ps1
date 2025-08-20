# run_monthly.ps1
# Scores the full dataset, builds queues, then runs monthly evaluation.

& ".\.venv\Scripts\python.exe" ".\predict_fraud_batch.py" `
  --input ".\dataset\credit_card_transactions.csv" `
  --output ".\eval_out\predictions_calibrated.csv" `
  --artifacts_dir ".\notebooks" `
  --id_col trans_num

& ".\.venv\Scripts\python.exe" ".\make_review_queues.py"

& ".\.venv\Scripts\python.exe" ".\monitoring\monthly_eval.py" `
  --input ".\dataset\credit_card_transactions.csv" `
  --preds ".\eval_out\predictions_calibrated.csv" `
  --policy ".\notebooks\operating_policy.json" `
  --id_col trans_num
