# 💳 Credit Card Fraud Detection Pipeline

End-to-end machine learning pipeline + Streamlit app to detect fraudulent credit card transactions.

## Features
- Feature engineering (EDA, velocity, time-based features)
- Trained LightGBM model with calibrated probabilities
- Batch scoring (`predict_fraud_batch.py`)
- Streamlit app for manual & CSV scoring
- Configurable fraud threshold (`inference_threshold.json`)

## Quickstart
```bash
git clone https://github.com/Beepeen78/card-fraud-detection-pipeline.git
cd card-fraud-detection-pipeline
pip install -r requirements.txt
streamlit run app.py
