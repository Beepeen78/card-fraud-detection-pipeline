Credit Card Fraud Detection Pipeline
Version: 1.0
Date: January 2025
Status: Production Ready

Executive Summary
This project delivers a machine learning–based credit card fraud detection system capable of real-time fraud scoring and business analytics.
The solution uses a calibrated LightGBM ensemble model with automated feature engineering, GPU-accelerated processing, and Power BI integration for effective monitoring and decision-making.

Highlights

Real-time fraud detection with calibrated probability scores

15 interactive visualizations for detailed analytics

Power BI dashboard with automated data exports

Monthly evaluation with threshold tuning

GPU acceleration for large-scale inference

User-friendly Gradio web interface

1. Project Overview
1.1 Problem Statement
Credit card fraud poses a persistent challenge, with an average fraud rate of just 0.2%. Traditional rule-based systems fail to keep up with evolving fraud tactics. The main objectives were to:

Detect fraud in real time.

Handle extreme class imbalance.

Generate interpretable risk probabilities.

Adapt dynamically to new fraud behaviors.

1.2 Solution Overview
The proposed solution uses a multi-layered data and modeling pipeline:

Feature Engineering: 25 features generated from raw transaction logs.

Modeling: Calibrated LightGBM with probability tuning and class-weighting.

Visualization: 15 interactive charts and Power BI dashboards.

Integration: Automated exports and monthly evaluation modules.

Deployment: Gradio-based production web app, GPU optional.

2. System Architecture
2.1 Core Components
text
┌──────────────────────┐
│  Web Interface (Gradio) │
│  • File upload, scoring │
│  • 15 interactive charts │
└─────────────┬────────────┘
              │
┌─────────────▼─────────────┐
│ Feature Engineering Module │
│  • Time, location, velocity │
│  • Aggregation & cleaning   │
└─────────────┬────────────┘
              │
┌─────────────▼────────────┐
│ Calibrated LightGBM Model │
│  • 25 engineered features  │
│  • Probability calibration  │
└─────────────┬────────────┘
              │
┌─────────────▼────────────┐
│ Output & BI Integration   │
│  • Predictions, risk tags │
│  • Power BI export & eval │
└───────────────────────────┘
2.2 Technology Stack
Layer	Tools & Libraries
ML & Processing	Python 3.8+, pandas, numpy, scikit-learn, LightGBM, XGBoost, joblib
Web Interface	Gradio, Plotly
Acceleration (Optional)	CuPy (CUDA 11/12)
Business Intelligence	Power BI, BigQuery
Monitoring & Evaluation	Python scripts + JSON config
3. Model Details
3.1 Model Overview
Type: Calibrated LightGBM Classifier

Features: 25 engineered features across 5 categories

Output: Fraud probability (0–1) + binary classification

Calibration: Isotonic or Platt scaling

3.2 Key Feature Groups
Category	Description	Example Features
Time-based (9)	Periodic transaction patterns	hour, dayofweek, hour_sin, is_weekend
Amount-based (6)	Statistical + velocity metrics	amt, mean_amt, total_amt_last_24h
Transaction Count (3)	Volume behavior	txn_count_last_1h, txn_count_last_24h
Location (3)	Geo-based risk	city_pop, dist_home_merch
Temporal (1)	Time gap	time_since_last_txn
3.3 Training Details
Dataset: Historical transactions labeled for fraud

Validation: Time-based split (to avoid leakage)

Loss Adjustment: Class weights & focal loss for imbalance

Performance:

ROC-AUC ≈ 0.81

PR-AUC ≈ 0.05 (typical for 0.2% fraud rate)

Recall: 0.60–0.95 @ threshold 0.01–0.05

4. Features & Capabilities
Web Interface
Upload CSV (up to 10k rows) or use demo dataset

Automatic feature extraction and cleaning

Adjustable fraud threshold

Real-time scoring with risk classification

Interactive analytics with 15 visualizations (ROC, PR, correlation heatmap, etc.)

Power BI Integration
Auto-export of predictions to powerbi/out/transactions_scored.csv

Ready-to-use five-page dashboard

30+ DAX measures pre-configured

Daily metrics, trend tracking, and alerting

Monitoring
Monthly evaluation script with metric tracking and threshold optimization

Auto-generated reports in eval_out/monthly/YYYY-MM/

5. Installation & Setup
Requirements: Python ≥3.8, pip, optional GPU
Setup:

bash
git clone https://github.com/Beepeen78/card-fraud-detection-pipeline
cd card-fraud-detection-pipeline
pip install -r requirements.txt
python app.py
Access the interface at http://127.0.0.1:7860

For GPU acceleration:

bash
pip install cupy-cuda12x  # or cupy-cuda11x
6. Deployment Options
Environment	Description
Local	Run locally via python app.py
Hugging Face Spaces	Cloud deployment with GPU support
Production	Use Gunicorn + Nginx + Docker for scalability and SSL
7. Monitoring & Evaluation
Command Example:

bash
python monitoring/monthly_eval.py \
  --input dataset/fraudTest.csv \
  --preds eval_out/predictions_calibrated.csv \
  --policy notebooks/operating_policy.json \
  --id_col trans_num
Generates metrics.json, threshold_suggestion.json, and analyst workbooks.

8. Limitations & Future Work
Current Limitations

Batch-only (no streaming yet)

Manual retraining required

Performance limited to ~10,000 rows per batch

Upcoming Enhancements

Real-time streaming via Kafka or RabbitMQ

Model serving API (FastAPI)

Explainability tools (SHAP, LIME)

Drift detection and A/B testing

Deep learning extensions (LSTM, Transformer)

9. Conclusion
The Credit Card Fraud Detection Pipeline provides a production-grade ML system blending technical precision with business usability.
Its modular design, visualization depth, and calibration-aware modeling make it applicable for both real-time fraud screening and analytical research in financial domains.

Key Strengths

✅ Interactive and interpretable

✅ Power BI business dashboard

✅ Automated monitoring and evaluation

✅ GPU-ready and fault-tolerant

✅ Extensively documented

Primary Use Cases

Fraud analytics teams

Financial institutions

Payment gateways

E-commerce risk detection

Contact & Support
For questions, setup issues, or contributions:

Review documentation in README_SPACE.md and powerbi/README.md.

Check Power BI setup in powerbi/setup_instructions.md.

Submit issues or contributions via the project’s GitHub repository.
