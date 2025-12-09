---
title: Credit Card Fraud Detection
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🛡️ Credit Card Fraud Detection System

A machine learning pipeline for detecting fraudulent credit card transactions in real-time using an ensemble of LightGBM, XGBoost, and Random Forest models.

## 🚀 Quick Start

1. **Upload a CSV file** with transaction data
2. **Adjust the threshold** (recommended: 0.01-0.05 for imbalanced data)
3. **Click "Detect Fraud"** to analyze transactions
4. **View results** across 15 interactive visualizations

## 📊 Features

- **Real-time fraud scoring** with calibrated probabilities
- **15 interactive visualizations** including:
  - Fraud probability distributions
  - Risk level breakdowns
  - Time series analysis
  - Model performance metrics (when ground truth available)
  - ROC and Precision-Recall curves
  - Feature correlation heatmaps
  - Threshold sensitivity analysis

## 📁 Required CSV Format

Your CSV should include at minimum:
- `unix_time` or timestamp column
- `amt` or amount column
- `city_pop` (city population)
- `dist_home_merch` (distance from home to merchant)
- `category` (transaction category)

**Note:** If you're missing velocity features (like `txn_count_last_1h`), the system will fill them with sensible defaults.

## 🎯 Model Details

- **25 engineered features** including time-based, velocity, and aggregated features
- **Ensemble approach**: LightGBM + XGBoost + Random Forest
- **Calibrated probabilities** for reliable threshold tuning
- **Handles imbalanced data** (typical fraud rate: 0.2%)

## 📈 Expected Performance

- **ROC-AUC**: ~0.81 (good discrimination)
- **PR-AUC**: 0.01-0.10 (typical for imbalanced data)
- **Precision**: 0.01-0.20 (depends on threshold)
- **Recall**: 0.50-0.95 (depends on threshold)

## 💡 Usage Tips

1. **Threshold Selection**: For imbalanced fraud data, use **0.01-0.05** instead of 0.5. The default 0.5 is too high and will miss most fraud.

2. **File Size**: Processing is limited to 10,000 rows for optimal performance.

3. **Ground Truth**: If your CSV includes a fraud label column (`is_fraud`, `fraud`, `target`, etc.), the app will automatically calculate model performance metrics.

## 🔧 Technical Details

- **Framework**: Gradio Blocks API
- **Visualizations**: Plotly (interactive charts)
- **Model**: Calibrated LightGBM ensemble
- **Features**: 25 engineered features with automatic feature engineering

## 📝 Model File

**Important**: This Space requires the model file `fraud_lgbm_calibrated.pkl` to be present. 

If deploying this Space:
1. Train the model using `train_improved_model.py` (if you have the training script)
2. Upload the model file to the Space repository
3. Or use Git LFS for large model files

## 🔗 Related Resources

- Full project documentation: See `README.md` in the repository
- Model training: `train_improved_model.py`
- Sample data generator: `generate_sample_dataset.py`
- Power BI integration: `powerbi_export.py`

## 📄 License

This project is for educational and portfolio purposes. Ensure you have proper data usage rights before processing real transaction data.

---

Built with Python, LightGBM, XGBoost, Gradio, and Plotly.
