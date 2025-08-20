import joblib, os

model_path = r"D:\mini_projects\creadit card transaction dataset\fraud_pipeline.joblib"

if not os.path.exists(model_path):
    print("❌ Model file not found:", model_path)
else:
    p = joblib.load(model_path)
    print("✅ Loaded pipeline:", type(p))

    if hasattr(p, "feature_names_in_"):
        print("Features:", p.feature_names_in_)
    else:
        print("⚠️ No feature_names_in_ attribute. Check training notebook for features.")
