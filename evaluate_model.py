from __future__ import annotations
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def evaluate(scored: pd.DataFrame) -> dict:
    y_true = scored.get("is_fraud")
    y_prob = scored["fraud_probability"]
    y_pred = scored["fraud_prediction"]
    metrics = {}
    if y_true is not None and y_true.notna().any():
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        metrics.update(precision=float(p), recall=float(r), f1=float(f))
    return metrics
