#!/usr/bin/env python3
import argparse, joblib, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="fraud_pipeline.joblib", help="Path to pipeline .joblib")
    args = ap.parse_args()

    if not os.path.exists(args.model_path):
        print("❌ Model file not found:", args.model_path, file=sys.stderr)
        sys.exit(2)

    p = joblib.load(args.model_path)
    print("✅ Loaded pipeline:", type(p))

    # Try multiple ways to get features
    shown = False
    if hasattr(p, "feature_names_in_"):
        print("feature_names_in_ (len={}):".format(len(p.feature_names_in_)))
        print(p.feature_names_in_[:50])
        shown = True

    try:
        prep = getattr(p, "named_steps", {}).get("prep")
        if prep is not None and hasattr(prep, "get_feature_names_out"):
            cols = prep.get_feature_names_out()
            print("prep.get_feature_names_out (len={}):".format(len(cols)))
            print(cols[:50])
            shown = True
    except Exception:
        pass

    if not shown:
        print("⚠️ No feature names accessible. Check training notebook to export feature_columns.pkl")

if __name__ == "__main__":
    main()
