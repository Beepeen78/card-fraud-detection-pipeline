# Credit Card Fraud Detection System - Project Report

**Project Name:** Credit Card Fraud Detection Pipeline  
**Version:** 1.0  
**Date:** January 2025  
**Status:** Production Ready

---

## Executive Summary

This project implements a comprehensive machine learning-based fraud detection system for credit card transactions. The system uses an ensemble of calibrated LightGBM models to predict fraudulent transactions in real-time, with a focus on handling highly imbalanced datasets (typical fraud rate: 0.2%). The solution includes a web-based interface, comprehensive visualizations, Power BI integration, and automated monitoring capabilities.

### Key Achievements
- ✅ Real-time fraud detection with calibrated probability scores
- ✅ 15 interactive visualizations for analysis
- ✅ Power BI dashboard integration
- ✅ Automated monthly evaluation and threshold optimization
- ✅ GPU acceleration support for large-scale processing
- ✅ Production-ready Gradio web interface

---

## 1. Project Overview

### 1.1 Problem Statement
Credit card fraud is a significant challenge in financial services, with fraudsters constantly evolving their tactics. Traditional rule-based systems are insufficient, requiring a machine learning approach that can:
- Detect fraud in real-time
- Handle highly imbalanced data (fraud rate ~0.2%)
- Provide interpretable risk scores
- Adapt to changing fraud patterns

### 1.2 Solution Approach
The system employs a multi-layered approach:
1. **Feature Engineering**: 25 engineered features from raw transaction data
2. **Ensemble Modeling**: Calibrated LightGBM model for probability estimation
3. **Real-time Scoring**: Batch processing with GPU acceleration support
4. **Visualization**: 15 interactive charts for analysis
5. **Integration**: Power BI dashboard for business users
6. **Monitoring**: Automated monthly evaluation and threshold optimization

---

## 2. System Architecture

### 2.1 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (Gradio)                    │
│  - File Upload & Processing                                  │
│  - Interactive Visualizations (15 charts)                    │
│  - Real-time Fraud Scoring                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Feature Engineering Pipeline                    │
│  - Time-based features (hour, day, month, cyclic)           │
│  - Velocity features (txn_count_last_1h, total_amt_last_24h)│
│  - Aggregated features (mean, std, median, max)              │
│  - Distance and location features                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│            Calibrated LightGBM Model                         │
│  - 25 engineered features                                   │
│  - Probability calibration for threshold tuning              │
│  - Handles imbalanced data                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Output & Integration                            │
│  - Fraud probabilities & predictions                         │
│  - Risk level classification                                │
│  - Power BI export (automatic)                               │
│  - Monthly evaluation reports                                │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

**Core ML & Data Processing:**
- Python 3.8+
- pandas 2.0+ (data manipulation)
- numpy 1.26+ (numerical operations)
- scikit-learn 1.7.1 (model calibration, metrics)
- LightGBM 4.0+ (gradient boosting)
- XGBoost 2.0+ (ensemble support)
- joblib (model serialization)

**Web Interface:**
- Gradio 4.44+ (web UI framework)
- Plotly 5.22+ (interactive visualizations)

**Optional GPU Acceleration:**
- CuPy (GPU-accelerated NumPy operations)

**Business Intelligence:**
- Power BI (dashboard and analytics)
- Google Cloud BigQuery (optional data warehouse)

**Monitoring & Evaluation:**
- Custom Python scripts for monthly evaluation
- JSON-based operating policy configuration

---

## 3. Model Details

### 3.1 Model Architecture

**Model Type:** Calibrated LightGBM Classifier  
**Features:** 25 engineered features  
**Calibration:** Probability calibration for reliable threshold tuning  
**Output:** Fraud probability (0-1) and binary prediction

### 3.2 Feature Engineering

The model uses exactly **25 features** organized into categories:

#### Time-Based Features (9 features)
- `hour`, `dayofweek`, `month`, `dayofyear` (raw time components)
- `hour_sin`, `hour_cos` (cyclic encoding for hour)
- `dow_sin`, `dow_cos` (cyclic encoding for day of week)
- `is_weekend`, `is_night`, `is_business_hours` (time-based flags)

#### Transaction Amount Features (6 features)
- `amt` (transaction amount)
- `mean_amt`, `std_amt`, `median_amt`, `max_amt` (aggregated statistics)
- `total_amt_last_1h`, `total_amt_last_24h` (velocity features)

#### Transaction Count Features (3 features)
- `transaction_count` (total transactions)
- `txn_count_last_1h`, `txn_count_last_24h` (velocity features)

#### Location & Distance Features (3 features)
- `city_pop` (city population)
- `dist_home_merch` (distance from home to merchant)
- `dist_category_bucket_idx` (categorical distance bucket)

#### Temporal Features (1 feature)
- `time_since_last_txn` (seconds since last transaction)

### 3.3 Model Training

- **Training Data:** Historical transaction data with fraud labels
- **Validation:** Time-based split to prevent data leakage
- **Calibration:** Platt scaling or isotonic regression for probability calibration
- **Hyperparameters:** Optimized for imbalanced data (class weights, focal loss)

### 3.4 Expected Performance

Based on typical fraud detection datasets:
- **ROC-AUC**: ~0.81 (good discrimination between fraud and normal)
- **PR-AUC**: 0.01-0.10 (typical for highly imbalanced data)
- **Precision**: 0.01-0.20 (depends on threshold - lower threshold = lower precision)
- **Recall**: 0.50-0.95 (depends on threshold - lower threshold = higher recall)

**Note:** For imbalanced fraud data, use threshold **0.01-0.05** instead of 0.5. The default 0.5 is too high and will miss most fraud cases.

---

## 4. Features & Capabilities

### 4.1 Web Interface Features

**File Upload & Processing:**
- CSV file upload with automatic validation
- Support for up to 10,000 rows per batch
- Automatic feature engineering from raw data
- Sample dataset for quick testing
- Direct dataset file access (bypasses upload issues)

**Interactive Visualizations (15 charts):**
1. **Fraud Probability Distribution** - Histogram showing probability distribution
2. **Risk Level Breakdown** - Pie chart of risk levels (Low/Medium/High/Critical)
3. **Amount vs Fraud Probability** - Scatter plot showing relationship
4. **Transaction Summary** - Bar chart with fraud statistics
5. **Fraud Probability Over Time** - Time series with rolling average
6. **Top 20 Most Suspicious Transactions** - Bar chart of highest risk transactions
7. **Model Performance Metrics** - ROC-AUC, PR-AUC, Precision, Recall, F1-Score
8. **Confusion Matrix** - Heatmap of true/false positives/negatives
9. **ROC Curve** - Receiver Operating Characteristic curve
10. **Precision-Recall Curve** - Better metric for imbalanced data
11. **Amount Distribution Comparison** - Box plots comparing fraud vs normal
12. **Probability Distribution Comparison** - Violin plots by actual/predicted labels
13. **Feature Correlation Heatmap** - Top 15 features correlation matrix
14. **Cumulative Fraud Detection** - Time series of cumulative fraud detection
15. **Threshold Sensitivity Analysis** - Precision/Recall/F1 across thresholds

**Real-time Analysis:**
- Adjustable fraud detection threshold (0.01-0.99)
- Instant fraud scoring with calibrated probabilities
- Risk level classification (Low/Medium/High/Critical)
- Top suspicious transactions table

### 4.2 Power BI Integration

**Automatic Data Export:**
- Transactions scored with fraud probabilities
- Daily aggregated metrics
- Ready-to-use CSV format

**Dashboard Components:**
- 5 dashboard pages (Executive Summary, Transaction Analysis, Model Performance, Time Series, Geographic Analysis)
- 30+ pre-built DAX measures
- Custom color theme matching risk levels
- Alert configuration and business rules
- Complete setup documentation

### 4.3 Monitoring & Evaluation

**Monthly Evaluation Script:**
- Automatic performance metrics calculation
- Threshold optimization for target precision/recall
- Operating policy configuration (JSON-based)
- Monthly reports in `eval_out/monthly/YYYY-MM/`

**Output Files:**
- `metrics.json` - Performance metrics
- `threshold_suggestion.json` - Recommended thresholds
- `analyst_pack.xlsx` - Comprehensive analysis workbook

---

## 5. Project Structure

```
card-fraud-detection-pipeline/
│
├── app.py                          # Main Gradio web application
├── fraud_lgbm_calibrated.pkl       # Trained model file
├── requirements.txt                # Python dependencies
├── README_SPACE.md                 # Hugging Face Spaces documentation
├── sample_transactions.csv         # Sample dataset for testing
│
├── dataset/
│   └── fraudTest.csv               # Test dataset
│
├── notebooks/
│   ├── eda_and_feature_engineering.ipynb  # EDA and model training
│   ├── feature_columns.pkl         # Feature definitions
│   └── operating_policy.json        # Operating policy configuration
│
├── powerbi/
│   ├── README.md                   # Power BI documentation
│   ├── QUICK_START.md              # Quick start guide
│   ├── setup_instructions.md       # Detailed setup guide
│   ├── dashboard_layout.md         # Dashboard design
│   ├── dax_measures.txt            # Pre-built DAX measures
│   ├── powerbi_theme.json          # Custom theme
│   ├── alerts_and_rules.md         # Alert configuration
│   ├── fraud_detection_queries.m   # Power Query scripts
│   ├── prepare_powerbi_data.py     # Data preparation script
│   └── out/                        # Exported data for Power BI
│       ├── transactions_scored.csv
│       └── metrics_daily.csv
│
├── monitoring/
│   └── monthly_eval.py             # Monthly evaluation script
│
├── eval_out/                       # Evaluation outputs
│   ├── predictions_calibrated.csv
│   ├── queue_block.csv             # High-risk transactions
│   ├── queue_review.csv            # Medium-risk transactions
│   ├── run_report.json
│   ├── threshold_suggestion.json
│   ├── analyst_pack.xlsx
│   └── monthly/
│       └── YYYY-MM/
│           ├── metrics.json
│           └── threshold_suggestion.json
│
└── Card-Fraud-detection/           # Alternative Streamlit app
    ├── app.py
    ├── fraud_lgbm_calibrated.pkl
    ├── powerbi_export.py
    └── README.md
```

---

## 6. Installation & Setup

### 6.1 Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for acceleration

### 6.2 Installation Steps

1. **Clone or download the project:**
   ```bash
   cd card-fraud-detection-pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists:**
   - The model file `fraud_lgbm_calibrated.pkl` should be in the project root
   - If missing, train the model using the notebook: `notebooks/eda_and_feature_engineering.ipynb`

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the web interface:**
   - Open browser to `http://127.0.0.1:7860`
   - Or use the provided URL if running on Hugging Face Spaces

### 6.3 Optional GPU Setup

For GPU acceleration (optional):
```bash
pip install cupy-cuda12x  # For CUDA 12.x
# Or
pip install cupy-cuda11x  # For CUDA 11.x
```

---

## 7. Usage Guide

### 7.1 Basic Usage

1. **Start the application:**
   ```bash
   python app.py
   ```

2. **Upload a CSV file or use sample data:**
   - Click "📊 Use Sample Dataset" for quick testing
   - Or upload your own CSV file
   - Or click "📂 Use dataset/fraudTest.csv" to use the test dataset

3. **Adjust the threshold:**
   - Recommended: 0.01-0.05 for imbalanced data
   - Lower threshold = higher recall (catches more fraud, more false positives)
   - Higher threshold = higher precision (fewer false positives, may miss fraud)

4. **Click "🔍 Detect Fraud"** to process transactions

5. **View results:**
   - Summary statistics in the main panel
   - Top suspicious transactions table
   - 15 interactive visualizations in tabs

### 7.2 CSV File Format

Your CSV should include at minimum:
- `unix_time` or timestamp column (for time-based features)
- `amt` or amount column (transaction amount)
- `city_pop` (city population)
- `dist_home_merch` (distance from home to merchant)
- `category` (transaction category, optional)

**Note:** If velocity features (like `txn_count_last_1h`) are missing, the system will fill them with sensible defaults based on the transaction data.

### 7.3 Power BI Integration

1. **Process transactions** in the web app (data exports automatically)

2. **Open Power BI Desktop**

3. **Load data:**
   - Get Data → Text/CSV
   - Select `powerbi/out/transactions_scored.csv`

4. **Follow the guides:**
   - `powerbi/QUICK_START.md` for basic dashboard
   - `powerbi/setup_instructions.md` for complete setup

### 7.4 Monthly Evaluation

Run monthly evaluation to assess model performance:
```bash
python monitoring/monthly_eval.py \
  --input dataset/fraudTest.csv \
  --preds eval_out/predictions_calibrated.csv \
  --policy notebooks/operating_policy.json \
  --id_col trans_num
```

---

## 8. Performance Metrics & Evaluation

### 8.1 Model Performance

The model is designed for highly imbalanced fraud detection:

**Typical Performance:**
- **ROC-AUC**: 0.80-0.85 (good discrimination)
- **PR-AUC**: 0.01-0.10 (expected for 0.2% fraud rate)
- **Precision at threshold 0.05**: 0.05-0.15
- **Recall at threshold 0.05**: 0.60-0.85

**Threshold Guidelines:**
- **0.01-0.05**: High recall, catches most fraud (recommended for imbalanced data)
- **0.05-0.10**: Balanced precision/recall
- **0.10-0.50**: Higher precision, fewer false positives
- **0.50+**: Very high precision, but will miss most fraud (not recommended)

### 8.2 Evaluation Metrics

The system calculates comprehensive metrics when ground truth is available:

**Classification Metrics:**
- Precision, Recall, F1-Score
- Confusion Matrix (True/False Positives/Negatives)
- Accuracy

**Probability Metrics:**
- ROC-AUC (Area Under ROC Curve)
- PR-AUC (Area Under Precision-Recall Curve) - better for imbalanced data
- Threshold sensitivity analysis

**Business Metrics:**
- Fraud detection rate
- False positive rate
- Cost analysis (if cost matrix provided)

### 8.3 Monitoring

**Monthly Evaluation:**
- Automatic performance tracking
- Threshold optimization
- Trend analysis
- Alert generation for performance degradation

---

## 9. Technical Implementation Details

### 9.1 Feature Engineering Pipeline

The system automatically engineers 25 features from raw transaction data:

1. **Time Feature Extraction:**
   - Parses `unix_time` to extract hour, day, month, day of year
   - Creates cyclic encodings (sin/cos) for hour and day of week
   - Generates time-based flags (weekend, night, business hours)

2. **Velocity Features:**
   - Transaction counts in last 1 hour and 24 hours
   - Total amounts in last 1 hour and 24 hours
   - Time since last transaction

3. **Aggregated Features:**
   - Mean, median, standard deviation, max of transaction amounts
   - Transaction count

4. **Location Features:**
   - City population
   - Distance from home to merchant
   - Distance category bucket index

5. **Missing Value Handling:**
   - Sensible defaults for missing velocity features
   - Median/mean imputation for other missing values
   - Handles NaN and infinite values

### 9.2 GPU Acceleration

The system supports optional GPU acceleration for:
- Cyclic encoding calculations (sin/cos)
- Large batch processing (>100 rows)
- ZeroGPU compatibility (Hugging Face Spaces)

**Fallback:** Automatically falls back to CPU if GPU is unavailable.

### 9.3 Error Handling

Comprehensive error handling for:
- File upload issues (Windows temp file permissions)
- Missing model file
- Invalid CSV format
- Missing required columns
- GPU operation failures

### 9.4 Data Export

**Automatic Power BI Export:**
- Exports scored transactions to `powerbi/out/transactions_scored.csv`
- Exports daily metrics to `powerbi/out/metrics_daily.csv`
- Includes all original columns plus fraud predictions

---

## 10. Deployment Options

### 10.1 Local Deployment

```bash
python app.py
```
- Runs on `http://127.0.0.1:7860`
- Suitable for development and testing
- Requires local Python environment

### 10.2 Hugging Face Spaces

The project is configured for Hugging Face Spaces deployment:
- GPU support via `@spaces.GPU` decorator
- Automatic model loading
- Public or private deployment options

**Configuration:**
- `README_SPACE.md` contains Spaces configuration
- `app_file: app.py` in metadata
- SDK: Gradio 4.44+

### 10.3 Production Deployment

For production deployment:
1. Use a production web server (Gunicorn, uWSGI)
2. Set up reverse proxy (Nginx)
3. Configure SSL/TLS
4. Set up monitoring and logging
5. Use containerization (Docker) for consistency

---

## 11. Future Enhancements

### 11.1 Planned Features

1. **Real-time Streaming:**
   - Kafka/RabbitMQ integration for real-time transaction processing
   - WebSocket support for live updates

2. **Model Improvements:**
   - Online learning for model updates
   - Ensemble of multiple models
   - Deep learning models (LSTM, Transformer)

3. **Enhanced Features:**
   - Graph-based features (transaction networks)
   - Behavioral biometrics
   - Device fingerprinting

4. **Explainability:**
   - SHAP values for feature importance
   - LIME explanations for individual predictions
   - Model interpretability dashboard

5. **Advanced Monitoring:**
   - Drift detection
   - Automated retraining pipeline
   - A/B testing framework

### 11.2 Scalability Improvements

- Distributed processing (Spark, Dask)
- Model serving API (FastAPI, Flask)
- Database integration (PostgreSQL, MongoDB)
- Caching layer (Redis)

---

## 12. Limitations & Considerations

### 12.1 Current Limitations

1. **Batch Processing:** Limited to 10,000 rows per batch for performance
2. **Feature Engineering:** Requires specific input columns (with defaults)
3. **Model Updates:** Manual retraining required (no online learning)
4. **Real-time:** Not optimized for sub-second latency requirements

### 12.2 Data Requirements

- Requires transaction data with time, amount, and location information
- Works best with historical transaction patterns
- May need domain-specific feature engineering for different use cases

### 12.3 Performance Considerations

- GPU acceleration recommended for large batches (>1000 rows)
- Model file size: ~34 MB (LightGBM model)
- Memory usage: ~500 MB - 2 GB depending on batch size

---

## 13. Conclusion

This Credit Card Fraud Detection System provides a comprehensive, production-ready solution for detecting fraudulent transactions. With its interactive web interface, comprehensive visualizations, Power BI integration, and automated monitoring, it offers both technical depth and business usability.

The system's focus on handling imbalanced data, providing calibrated probabilities, and offering extensive visualization capabilities makes it suitable for both technical teams and business stakeholders.

### Key Strengths

✅ **Comprehensive:** 15 visualizations, Power BI integration, monitoring  
✅ **User-Friendly:** Intuitive web interface with sample data  
✅ **Production-Ready:** Error handling, GPU support, automated evaluation  
✅ **Flexible:** Handles missing features, multiple input formats  
✅ **Well-Documented:** Extensive documentation and guides

### Use Cases

- Financial institutions detecting credit card fraud
- E-commerce platforms identifying fraudulent transactions
- Payment processors screening transactions
- Fraud analytics teams analyzing patterns
- Data science teams learning fraud detection techniques

---

## 14. Contact & Support

For questions, issues, or contributions:
- Review the documentation in `README_SPACE.md` and `powerbi/README.md`
- Check the Power BI guides for dashboard setup
- Review the monitoring scripts for evaluation setup

---

**Report Generated:** January 2025  
**Project Version:** 1.0  
**Status:** Production Ready

