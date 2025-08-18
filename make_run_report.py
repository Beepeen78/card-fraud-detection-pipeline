import json
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\mini_projects\creadit card transaction dataset")
pred = pd.read_csv(ROOT / r"eval_out\predictions_calibrated.csv")

summary = {
    "rows": int(len(pred)),
    "flag_rate": float((pred["decision"]!="allow").mean()),
    "block": int((pred["decision"]=="block").sum()),
    "review": int((pred["decision"]=="review").sum())
}

out = ROOT / r"eval_out\run_report.json"
out.write_text(json.dumps(summary, indent=2))
print("Saved:", out)
