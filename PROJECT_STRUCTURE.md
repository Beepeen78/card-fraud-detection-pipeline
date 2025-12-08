# Project Structure

## Cleaned Project Organization

This document outlines the cleaned and organized project structure after removing redundant and unused files.

## Core Files (Essential)

```
├── app.py                          # Main Gradio web application
├── train_improved_model.py         # Model training script
├── generate_sample_dataset.py      # Sample data generator
├── evaluate_model.py               # Model evaluation utilities
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── fraud_lgbm_calibrated.pkl      # Trained model file (used by app.py)
└── sample_transactions.csv         # Sample data for testing
```

## Integration & Export

```
├── powerbi_export.py               # Power BI CSV export
├── upload_to_bigquery.py           # BigQuery integration
└── powerbi/
    └── out/                        # Power BI output directory
        ├── metrics_daily.csv
        └── transactions_scored.csv
```

## Deployment & Automation

```
├── Dockerfile                      # Docker container configuration
└── run_monthly.ps1                # Monthly batch processing script
```

## Development & Analysis

```
└── notebooks/
    ├── eda_and_feature_engineering.ipynb  # EDA and feature engineering
    ├── feature_columns.pkl                 # Feature reference
    └── operating_policy.json               # Operating parameters
```

## Version Control

```
├── .gitignore                      # Git ignore rules
└── .gitattributes                  # Git attributes
```

## Removed Files (Cleaned Up)

The following files were removed as they were redundant, unused, or had hardcoded paths:

- `batch_project.py` - Alternative simple app (not used)
- `predict_fraud_batch.py` - Used different feature set (V1-V28), conflicted with app.py
- `fraud_lgbm_model.pkl` - Old model file (replaced by fraud_lgbm_calibrated.pkl)
- `fraud_pipeline.joblib` - Old pipeline from notebook (not used)
- `inference_threshold.json` - Old threshold file (not used)
- `behavior feature.txt` - Notes file (not needed)
- `show_pipe_features.py` - Utility for old pipeline (not needed)
- `test_speed.py` - Test file (not needed in production)
- `test_bq.py` - Test file (not needed in production)
- `make_review_queues.py` - Had hardcoded paths (not generic)
- `make_run_report.py` - Had hardcoded paths (not generic)

## File Purposes

### app.py
Main application with Gradio interface. Handles:
- CSV file uploads
- Feature engineering (25 features)
- Model predictions
- 15 interactive visualizations
- Results display

### train_improved_model.py
Trains the fraud detection model:
- Ensemble of LightGBM, XGBoost, Random Forest
- SMOTE/ADASYN oversampling
- Cost-sensitive learning
- Probability calibration
- Saves: `fraud_model_improved_calibrated.pkl` and related files

### generate_sample_dataset.py
Creates sample transaction data for testing:
- 100 realistic transactions
- Base features: unix_time, amt, city_pop, dist_home_merch, category
- Mix of normal and suspicious patterns

### evaluate_model.py
Model evaluation utilities:
- Performance metrics calculation
- Holdout set testing
- Confusion matrix generation

### powerbi_export.py
Exports scored transactions to CSV format for Power BI:
- Row-level transaction data
- Daily aggregated metrics
- Outputs to `powerbi/out/`

### upload_to_bigquery.py
BigQuery integration:
- Uploads scored transactions
- Uploads daily metrics
- Requires environment variables for configuration

## Model Files

- **fraud_lgbm_calibrated.pkl** - Main production model (used by app.py)
- **fraud_model_improved_calibrated.pkl** - Output from train_improved_model.py (if you train a new model)

Note: If you train a new model, update `MODEL_PATH` in `app.py` to point to the new model file.
