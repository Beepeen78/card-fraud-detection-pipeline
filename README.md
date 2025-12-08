# Credit Card Fraud Detection System

A machine learning pipeline for detecting fraudulent credit card transactions in real-time. This project uses an ensemble of LightGBM, XGBoost, and Random Forest models trained on 25 engineered features to identify suspicious transactions with high accuracy.

## What This Does

You upload a CSV file with transaction data, and the system scores each transaction with a fraud probability. It then shows you which ones look suspicious, along with a bunch of charts and stats to help you understand what's going on. The whole thing runs in a web interface built with Gradio, so you don't need to mess with command lines.

## Quick Start

### Setup

```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate it
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Then open your browser to `http://127.0.0.1:7860` and you're good to go.

### Try the Sample Data

If you want to test it out without your own data, there's a sample dataset generator:

```bash
python generate_sample_dataset.py
```

This creates `sample_transactions.csv` with 100 realistic transactions. Then in the app, just click the "📊 Use Sample Dataset" button.

## Project Structure

Here's what each file does:

### Core Application Files

**`app.py`** - The main web interface
- Gradio-based UI for uploading CSVs and viewing results
- Handles file uploads, feature engineering, and prediction
- Generates 15 different visualizations (more on that below)
- Shows summary stats, top suspicious transactions, and model performance metrics

**`score_batch()` function** - The scoring engine
- Takes raw transaction data and engineers 25 features
- Handles missing data with sensible defaults
- Runs predictions through the calibrated model
- Returns fraud probabilities and risk levels

**`create_visualizations()` function** - Chart generator
- Creates interactive Plotly charts
- 15 different visualizations covering distributions, correlations, time series, etc.
- Automatically adapts based on available data

### Model Training

**`train_improved_model.py`** - Model training script
- Trains ensemble of LightGBM, XGBoost, and Random Forest
- Uses SMOTE/ADASYN for handling imbalanced data
- Implements cost-sensitive learning (fraud is weighted 20x more than normal)
- Calibrates probabilities for better threshold tuning
- Saves multiple model files:
  - `fraud_model_improved_calibrated.pkl` - Main production model
  - `fraud_model_improved_ensemble.pkl` - All ensemble components
  - `fraud_model_improved_info.json` - Model metadata
  - `fraud_model_improved_threshold.json` - Optimal thresholds

**`evaluate_model.py`** - Model evaluation
- Tests model performance on holdout sets
- Calculates ROC-AUC, PR-AUC, precision, recall, F1
- Generates confusion matrices

### Data Generation

**`generate_sample_dataset.py`** - Sample data generator
- Creates realistic transaction data for testing
- Includes base features: unix_time, amt, city_pop, dist_home_merch, category
- Generates 100 transactions with mix of normal and suspicious patterns
- Run this to create `sample_transactions.csv` for testing

### Data Export

**`powerbi_export.py`** - Power BI integration
- Exports scored transactions to CSV for Power BI
- Creates daily metrics files
- Outputs to `powerbi/out/` directory

**`upload_to_bigquery.py`** - BigQuery integration
- Uploads results to Google BigQuery
- Requires environment variables:
  - `BQ_PROJECT` - Your GCP project ID
  - `BQ_DATASET` - Dataset name (e.g., "fraud_prod")
  - `BQ_TABLE_SCORED` - Table for scored transactions
  - `BQ_TABLE_METRICS` - Table for daily metrics


### Notebooks

**`notebooks/eda_and_feature_engineering.ipynb`** - Exploratory analysis
- Jupyter notebook with full EDA
- Feature engineering experiments
- Model development and testing
- Target encoding implementations

### Configuration

**`requirements.txt`** - Python dependencies
- All required packages with version constraints
- Core: pandas, numpy, scikit-learn, joblib
- ML: lightgbm, xgboost, imbalanced-learn
- UI: gradio, plotly
- Optional: google-cloud-bigquery (for BigQuery export)

**`Dockerfile`** - Container configuration
- Docker setup for deployment
- Includes all dependencies

**`run_monthly.ps1`** - Monthly batch script
- PowerShell script for scheduled runs
- Useful for Windows task scheduler

## The 25 Features

The model expects exactly these 25 features (in sorted order):

**Base Features:**
- `amt` - Transaction amount
- `city_pop` - City population
- `dist_home_merch` - Distance from home to merchant

**Time Features:**
- `hour`, `dayofweek`, `month`, `dayofyear` - Extracted from timestamp
- `hour_sin`, `hour_cos` - Cyclic encoding of hour (24-hour cycle)
- `dow_sin`, `dow_cos` - Cyclic encoding of day of week (7-day cycle)

**Velocity Features (last 1 hour):**
- `txn_count_last_1h` - Number of transactions in last hour
- `total_amt_last_1h` - Total amount spent in last hour
- `txn_count_last_1h_category` - Transactions in same category (last hour)
- `total_amt_last_1h_category` - Amount in same category (last hour)

**Velocity Features (last 24 hours):**
- `txn_count_last_24h` - Number of transactions in last 24 hours
- `total_amt_last_24h` - Total amount spent in last 24 hours
- `txn_count_last_24h_category` - Transactions in same category (last 24h)
- `total_amt_last_24h_category` - Amount in same category (last 24h)

**Aggregated Features:**
- `mean_distance` - Average distance from home
- `time_since_last_txn` - Seconds since last transaction
- `mean_amt` - Average transaction amount
- `std_amt` - Standard deviation of transaction amounts

**Target Encodings:**
- `te_job` - Target-encoded job category fraud rate
- `te_dist_category` - Target-encoded distance category fraud rate

If your data doesn't have all these features, the system fills missing ones with sensible defaults (not zeros - that was causing issues earlier).

## Visualizations

The app generates 15 interactive charts:

### Basic Charts
1. **Fraud Probability Distribution** - Histogram showing distribution of fraud scores
2. **Risk Level Breakdown** - Pie chart of Low/Medium/High/Critical risk levels
3. **Amount vs Fraud Probability** - Scatter plot showing relationship
4. **Transaction Summary** - Bar chart with total, fraud, and normal counts
5. **Fraud Probability Over Time** - Time series with rolling average
6. **Top 20 Most Suspicious** - Bar chart of highest-risk transactions

### Model Performance (when ground truth available)
7. **Model Performance Metrics** - Bar chart of ROC-AUC, PR-AUC, Precision, Recall, F1
8. **Confusion Matrix** - Heatmap of true/false positives/negatives

### Advanced Analytics
9. **ROC Curve** - Receiver Operating Characteristic curve with AUC score
10. **Precision-Recall Curve** - PR curve (better for imbalanced data) with PR-AUC
11. **Amount Distribution Comparison** - Box plots comparing normal vs fraud amounts
12. **Probability Distribution (Violin Plot)** - Distribution shapes for fraud vs normal
13. **Feature Correlation Heatmap** - Shows relationships between top features
14. **Cumulative Fraud Detection** - Cumulative count and rate over time
15. **Threshold Sensitivity Analysis** - How precision/recall/F1 change with threshold

All charts are interactive - you can zoom, pan, hover for details, etc.

## Model Performance

The model uses an ensemble approach:

- **LightGBM** (40-60% weight) - Fast gradient boosting, handles large datasets well
- **XGBoost** (40% weight if available) - Another gradient booster, catches different patterns
- **Random Forest** (20-40% weight) - Ensemble of decision trees, good for non-linear patterns

**Training Techniques:**
- SMOTE oversampling to balance the 0.2% fraud rate
- Cost-sensitive learning (fraud weighted 20x more than normal)
- Probability calibration for reliable thresholds
- Threshold optimization using precision-recall curves

**Expected Performance:**
- ROC-AUC: ~0.81 (good discrimination)
- PR-AUC: Varies, but typically 0.01-0.10 for imbalanced data
- Precision: 0.01-0.20 depending on threshold
- Recall: 0.50-0.95 depending on threshold
- F1-Score: 0.02-0.10 (target is >0.05 for production)

The model is calibrated, meaning the probabilities are reliable for threshold selection.

## Results Example

When you run the app on sample data, you'll see something like:

```
📊 Fraud Detection Results

Processing Summary:
- ✅ Processed 100 transactions in 0.15s
- 🎯 Detection threshold: 0.05

Fraud Statistics:
- 🚨 Fraud flagged: 12 transactions (12.00%)
- ✅ Normal transactions: 88 (88.00%)

Probability Statistics:
- 📈 Average probability: 0.0823
- 📊 Median probability: 0.0345
- ⬆️ Maximum probability: 0.8765
- ⬇️ Minimum probability: 0.0012

Risk Level Breakdown:
Low        65
Medium     23
High        8
Critical    4
```

If you have ground truth labels (like `is_fraud` column), you'll also get:

```
🎯 Model Performance Metrics

Current Threshold (0.05) Performance:
- 📊 ROC-AUC: 0.8123 ✅ Good discrimination
- 📈 PR-AUC: 0.0234 ⚠️ Very low - model struggles with imbalance
- 🎯 Precision: 0.0156 (98.44% false positives)
- 🔍 Recall: 0.9234 (7.66% fraud missed)
- ⚖️ F1-Score: 0.0307 ⚠️ Poor balance

Confusion Matrix:
- ✅ True Negatives: 8,234
- ❌ False Positives: 1,456
- ❌ False Negatives: 12
- ✅ True Positives: 298

Accuracy: 84.56%
```

## Usage Tips

1. **Threshold Selection**: For imbalanced fraud data, use 0.01-0.05 instead of 0.5. The default 0.5 is way too high and will miss most fraud.

2. **File Format**: Your CSV needs at minimum:
   - `unix_time` or timestamp column
   - `amt` or amount column
   - `city_pop` (city population)
   - `dist_home_merch` (distance from home)
   - `category` (transaction category)

3. **Large Files**: The app limits processing to 10,000 rows for performance. For larger files, you can modify the `nrows` parameter in `app.py` or process in batches.

4. **Missing Features**: Don't worry if you're missing velocity features (like `txn_count_last_1h`). The system fills them with reasonable defaults based on your transaction data.

5. **Sample Data**: Use the sample dataset button to test the system without uploading your own data first.

## Troubleshooting

**Model gives all zeros:**
- This was a bug where missing features were set to 0. It's fixed now - the system uses sensible defaults instead.

**Permission errors on Windows:**
- Gradio sometimes has issues with Windows temp files. Use the "Use dataset/fraudTest.csv" button or copy your file to the project folder first.

**Model file not found:**
- Make sure `fraud_lgbm_calibrated.pkl` is in the same directory as `app.py`
- If you trained a new model with `train_improved_model.py`, you might need to update the MODEL_PATH in `app.py`

**Feature mismatch errors:**
- The model expects exactly 25 features in sorted order. Check the console output for which features are missing.

## Next Steps

- Train your own model: Run `train_improved_model.py` with your dataset
- Export to Power BI: Use the export functionality in the app
- Set up BigQuery: Configure environment variables and use BigQuery export
- Schedule runs: Use `run_monthly.ps1` with Windows Task Scheduler
- Customize features: Modify `score_batch()` in `app.py` to add your own features

## License

This project is for educational and portfolio purposes. Make sure you have proper data usage rights before processing real transaction data.

---

Built with Python, LightGBM, XGBoost, Gradio, and Plotly. The model training uses scikit-learn, imbalanced-learn, and various other ML libraries. Check `requirements.txt` for the full list.
