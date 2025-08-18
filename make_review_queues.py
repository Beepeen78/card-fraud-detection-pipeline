import pandas as pd
from pathlib import Path

ROOT = Path(r"D:\mini_projects\creadit card transaction dataset")
raw  = (pd.read_csv(ROOT / r"dataset\credit_card_transactions.csv")
          .drop(columns=["Unnamed: 0"], errors="ignore"))
pred = pd.read_csv(ROOT / r"eval_out\predictions_calibrated.csv")

# If predictions include trans_num (from --id_col), join on it; else fall back to row order
if "trans_num" in pred.columns and "trans_num" in raw.columns:
    df = raw.merge(pred[["trans_num","fraud_probability","decision"]],
                   on="trans_num", how="left")
else:
    df = raw.copy()
    df["fraud_probability"] = pred["fraud_probability"].values
    df["decision"] = pred["decision"].values

out_dir = ROOT / "eval_out"
out_dir.mkdir(exist_ok=True)

# CSV queues
df.query("decision == 'block'").sort_values("fraud_probability", ascending=False)\
  .to_csv(out_dir / "queue_block.csv", index=False)
df.query("decision == 'review'").sort_values("fraud_probability", ascending=False)\
  .to_csv(out_dir / "queue_review.csv", index=False)

# Summary tables
by_cat  = (df[df.decision != "allow"]
           .groupby("category")["fraud_probability"]
           .agg(count="size", avg_prob="mean")
           .sort_values("avg_prob", ascending=False))
top_merch = (df[df.decision != "allow"]
             .groupby("merchant")["fraud_probability"]
             .agg(count="size", avg_prob="mean")
             .sort_values("avg_prob", ascending=False)
             .head(50))
by_hour = (df.assign(hour=pd.to_datetime(df["trans_date_trans_time"],
                                        errors="coerce").dt.hour)
           .groupby("hour")["fraud_probability"]
           .agg(flagged="count", avg_prob="mean"))

# Excel analyst pack
with pd.ExcelWriter(out_dir / "analyst_pack.xlsx", engine="xlsxwriter") as xl:
    df.query("decision == 'block'").sort_values("fraud_probability", ascending=False)\
      .to_excel(xl, sheet_name="block", index=False)
    df.query("decision == 'review'").sort_values("fraud_probability", ascending=False)\
      .to_excel(xl, sheet_name="review", index=False)
    by_cat.reset_index().to_excel(xl, sheet_name="summary_by_category", index=False)
    top_merch.reset_index().to_excel(xl, sheet_name="top_merchants", index=False)
    by_hour.reset_index().to_excel(xl, sheet_name="summary_by_hour", index=False)

print("Saved:",
      out_dir / "queue_block.csv",
      out_dir / "queue_review.csv",
      out_dir / "analyst_pack.xlsx")
