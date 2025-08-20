# monitoring/monthly_eval.py
"""
Monthly evaluation that uses the predictions CSV you already produced.
- Joins predictions back to the original dataset via an ID column
- Computes ROC-AUC / PR-AUC (if labels exist), precision/recall at policy
- Suggests new thresholds for target precisions
- Writes results into eval_out/monthly/YYYY-MM/

Usage:
  python monitoring/monthly_eval.py \
    --input dataset/credit_card_transactions.csv \
    --preds eval_out/predictions_calibrated.csv \
    --policy notebooks/operating_policy.json \
    --id_col trans_num
"""

import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, precision_recall_curve
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to original CSV with labels")
    ap.add_argument("--preds", required=True, help="Path to predictions_calibrated.csv")
    ap.add_argument("--policy", default="notebooks/operating_policy.json")
    ap.add_argument("--id_col", default="trans_num")
    ap.add_argument("--time_col", default="trans_date_trans_time")
    ap.add_argument("--label_col", default="is_fraud")
    ap.add_argument("--target_block_precision", type=float, default=0.98)
    ap.add_argument("--target_review_precision", type=float, default=0.75)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    OUTROOT = ROOT / "eval_out" / "monthly"
    OUTROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    preds = pd.read_csv(args.preds)

    # Determine the month folder from max timestamp present in data
    if args.time_col in df.columns:
        df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
        month_tag = df[args.time_col].max().strftime("%Y-%m")
    else:
        month_tag = "unknown-month"

    OUT = OUTROOT / month_tag
    OUT.mkdir(parents=True, exist_ok=True)

    # Join on ID if present in predictions (best practice: pass --id_col when scoring)
    if args.id_col in preds.columns and args.id_col in df.columns:
        keep = [c for c in [args.id_col, args.label_col, args.time_col] if c in df.columns]
        df_join = df[keep].merge(
            preds[[args.id_col, "fraud_probability", "decision"]],
            on=args.id_col, how="inner"
        )
    else:
        # Fallback: align by row order (not ideal, but works if no ID was output)
        n = min(len(df), len(preds))
        df_join = pd.DataFrame({
            "fraud_probability": preds["fraud_probability"].values[:n],
            "decision": preds["decision"].values[:n],
        })
        if args.label_col in df.columns:
            df_join[args.label_col] = df[args.label_col].values[:n]
        if args.time_col in df.columns:
            df_join[args.time_col] = pd.to_datetime(df[args.time_col].values[:n], errors="coerce")

    # Load current policy
    policy_path = Path(args.policy)
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {"block":0.90,"review":0.60}

    # Always compute flag stats (labels not required)
    flag_mask = df_join["decision"].ne("allow")
    flag_rate = float(flag_mask.mean())
    counts = df_join["decision"].value_counts().to_dict()

    metrics = {
        "rows": int(len(df_join)),
        "flags_total": int(flag_mask.sum()),
        "flag_rate": float(flag_rate),
        "tier_counts": {k: int(v) for k, v in counts.items()},
        "policy": policy,
        "month": month_tag,
    }

    # If labels exist, compute quality metrics
    if args.label_col in df_join.columns:
        y_true = df_join[args.label_col].astype(int).values
        y_prob = df_join["fraud_probability"].astype(float).values

        try:
            roc = roc_auc_score(y_true, y_prob)
            pr  = average_precision_score(y_true, y_prob)
        except Exception:
            roc, pr = float("nan"), float("nan")

        # Precision/recall at current policy (treat review+block as positive)
        y_pred_pos = df_join["decision"].ne("allow").astype(int).values
        prec = precision_score(y_true, y_pred_pos, zero_division=0)
        rec  = recall_score(y_true, y_pred_pos, zero_division=0)

        metrics.update({
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "precision_at_policy": float(prec),
            "recall_at_policy": float(rec),
        })

        # Suggest thresholds for target precisions
        prec_arr, rec_arr, thr = precision_recall_curve(y_true, y_prob)

        # Highest threshold achieving desired block precision
        cand_block = [t for p, t in zip(prec_arr[:-1], thr) if p >= args.target_block_precision]
        best_block = max(cand_block) if cand_block else float(thr[-1])

        # Highest threshold for review precision, below block threshold
        cand_review = [t for p, t in zip(prec_arr[:-1], thr)
                       if p >= args.target_review_precision and t < best_block]
        best_review = max(cand_review) if cand_review else min(best_block, 0.5)

        suggest = {"block": float(best_block), "review": float(best_review), "allow": 0.0}
        (OUT / "threshold_suggestion.json").write_text(json.dumps(suggest, indent=2))
        metrics["threshold_suggestion"] = suggest

    # Save metrics JSON
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Monthly eval ->", OUT)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
