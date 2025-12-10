# app.py - Enhanced Gradio Fraud Detection App with Visualizations

import time
import os
import shutil
import tempfile
import uuid
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from io import BytesIO
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, confusion_matrix,
    precision_recall_curve, average_precision_score, f1_score, roc_curve
)

# Import Hugging Face Spaces GPU decorator (optional, only works on Spaces)
try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False
    spaces = None

# Try to import GPU-accelerated libraries for data processing
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy available - GPU acceleration enabled for data processing")
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    print("⚠️ CuPy not available - using CPU for data processing")

# Import Power BI export function
try:
    from powerbi_export import export_powerbi_csvs
    POWERBI_AVAILABLE = True
except ImportError:
    POWERBI_AVAILABLE = False
    print("⚠️ powerbi_export module not found. Power BI export will be skipped.")

# Fix for Gradio API info generation bug
# Monkey-patch to handle the TypeError in gradio_client.utils when schema is a bool
try:
    import gradio_client.utils as client_utils
    
    # Patch get_type to handle bool values
    original_get_type = client_utils.get_type
    
    def patched_get_type(schema):
        """Patched version that handles bool schema values"""
        # Handle case where schema is a bool (True/False) instead of a dict
        if isinstance(schema, bool):
            return "bool"
        if not isinstance(schema, dict):
            return "Any"
        # Check if "const" key exists before using "in" operator
        if isinstance(schema, dict) and "const" in schema:
            return str(schema["const"])
        try:
            return original_get_type(schema)
        except TypeError:
            # Fallback if original function fails
            return "Any"
    
    client_utils.get_type = patched_get_type
    
    # Also patch _json_schema_to_python_type to handle additionalProperties being a bool
    original_json_schema_to_python_type = client_utils._json_schema_to_python_type
    
    def patched_json_schema_to_python_type(schema, defs=None):
        """Patched version that handles bool additionalProperties"""
        if isinstance(schema, bool):
            return "Any"
        if isinstance(schema, dict) and "additionalProperties" in schema:
            additional_props = schema["additionalProperties"]
            if isinstance(additional_props, bool):
                # If additionalProperties is True, allow any properties
                # If False, no additional properties allowed
                schema = schema.copy()
                schema["additionalProperties"] = {"type": "object" if additional_props else "null"}
        try:
            return original_json_schema_to_python_type(schema, defs)
        except (TypeError, KeyError) as e:
            # Fallback to avoid crashes
            return "Any"
    
    client_utils._json_schema_to_python_type = patched_json_schema_to_python_type
    print("✅ Gradio API info bug patched")
except Exception as e:
    print(f"⚠️ Could not patch Gradio client utils: {e}")

MODEL_PATH = "fraud_lgbm_calibrated.pkl"
model = None

def load_model():
    """Load the model file if it exists."""
    global model
    if model is not None:
        return model
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Please train the model first or download it from your repository.\n"
            f"The model file should be in the same directory as app.py"
        )
    
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded! Features: {model.n_features_in_}")
    print(f"   Model type: {type(model)}")
    
    # Test model with dummy data to verify it works
    try:
        test_features = np.zeros((1, model.n_features_in_))
        test_pred = model.predict_proba(test_features)[:, 1]
        print(f"   ✅ Model test prediction: {test_pred[0]:.4f}")
    except Exception as e:
        print(f"   ⚠️ Model test failed: {e}")
    
    return model

# Try to load model at startup (but don't crash if it fails)
try:
    load_model()
except FileNotFoundError as e:
    print(f"⚠️ {e}")
    print("⚠️ App will start but predictions will fail until model is available.")


def score_batch(raw_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Batch scoring with the exact 25 features the calibrated model expects.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("No data provided")

    df = raw_df.copy()

    # === EXACT 25 FEATURES THE MODEL WAS TRAINED ON ===
    exact_features = [
        'amt', 'city_pop', 'dayofweek', 'dayofyear', 'dist_category_bucket_idx',
        'dist_home_merch', 'dow_cos', 'dow_sin', 'hour', 'hour_cos', 'hour_sin',
        'is_business_hours', 'is_night', 'is_weekend', 'max_amt', 'mean_amt',
        'median_amt', 'month', 'std_amt', 'time_since_last_txn',
        'total_amt_last_1h', 'total_amt_last_24h', 'transaction_count',
        'txn_count_last_1h', 'txn_count_last_24h'
    ]

    # Extract time features from unix_time if present
    if "unix_time" in df.columns:
        ts = pd.to_datetime(pd.to_numeric(df["unix_time"], errors="coerce"), unit="s", utc=True)
        df["hour"] = ts.dt.hour.fillna(0).astype(int)
        df["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
        df["month"] = ts.dt.month.fillna(1).astype(int)
        df["dayofyear"] = ts.dt.dayofyear.fillna(1).astype(int)
    else:
        # Default time features if not available
        df["hour"] = 12  # Noon as default
        df["dayofweek"] = 1  # Monday as default
        df["month"] = 6  # June as default
        df["dayofyear"] = 150  # Mid-year as default

    # Cyclic encodings for hour and dayofweek
    # Use GPU acceleration if available for large datasets
    for col, period, sin_col, cos_col in [
        ("hour", 24, "hour_sin", "hour_cos"),
        ("dayofweek", 7, "dow_sin", "dow_cos")
    ]:
        if col in df.columns:
            # Use GPU for large datasets if CuPy is available
            if CUPY_AVAILABLE and len(df) > 1000:
                try:
                    gpu_array = cp.asarray(df[col].values)
                    angle = 2 * cp.pi * gpu_array / period
                    df[sin_col] = cp.asnumpy(cp.sin(angle))
                    df[cos_col] = cp.asnumpy(cp.cos(angle))
                except Exception:
                    # Fallback to CPU if GPU operation fails
                    angle = 2 * np.pi * df[col] / period
                    df[sin_col] = np.sin(angle)
                    df[cos_col] = np.cos(angle)
            else:
                # CPU computation for small datasets or when CuPy unavailable
                angle = 2 * np.pi * df[col] / period
                df[sin_col] = np.sin(angle)
                df[cos_col] = np.cos(angle)
    
    # Time-based flags
    if "hour" in df.columns:
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
        df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
    else:
        df["is_weekend"] = 0
        df["is_night"] = 0
        df["is_business_hours"] = 0

    # Ensure base features exist with sensible defaults
    if "amt" not in df.columns:
        # Try to find amount column
        for col in ["amount", "Amount", "TransactionAmt"]:
            if col in df.columns:
                df["amt"] = df[col]
                break
        if "amt" not in df.columns:
            df["amt"] = 50.0  # Default amount
    
    if "city_pop" not in df.columns:
        df["city_pop"] = 100000.0  # Default city population
    
    if "dist_home_merch" not in df.columns:
        df["dist_home_merch"] = 5.0  # Default distance (5 km)
    
    # Create dist_category_bucket_idx from dist_category if available
    if "dist_category_bucket_idx" not in df.columns:
        if "dist_category" in df.columns:
            # Map distance categories to bucket indices
            dist_order = {"<1km": 0, "1-10km": 1, "10-50km": 2, "50-100km": 3, "100-500km": 4, "500km+": 5, ">=500km": 5}
            df["dist_category_bucket_idx"] = df["dist_category"].map(dist_order).fillna(3).astype(int)
        else:
            # Default to middle bucket (3 = 50-100km)
            df["dist_category_bucket_idx"] = 3

    # Fill historical/velocity features with sensible defaults (not 0!)
    # These features typically have non-zero values in real data
    velocity_defaults = {
        'txn_count_last_1h': 1.0,  # At least this transaction
        'total_amt_last_1h': df["amt"].fillna(50.0),  # Use current amount as estimate
        'txn_count_last_24h': 3.0,  # Typical: 3 transactions per day
        'total_amt_last_24h': df["amt"].fillna(50.0) * 3,  # Estimate based on count
    }
    
    for col, default_val in velocity_defaults.items():
        if col not in df.columns:
            if isinstance(default_val, pd.Series):
                df[col] = default_val
            else:
                df[col] = default_val

    # Fill aggregated features with sensible defaults
    amt_filled = df["amt"].fillna(50.0)
    aggregate_defaults = {
        'time_since_last_txn': 3600.0,  # 1 hour ago (in seconds)
        'mean_amt': amt_filled,  # Use current amount as mean
        'std_amt': amt_filled * 0.3,  # 30% of mean as std
        'median_amt': amt_filled,  # Use current amount as median
        'max_amt': amt_filled,  # Use current amount as max
        'transaction_count': 1.0,  # At least this transaction
    }
    
    for col, default_val in aggregate_defaults.items():
        if col not in df.columns:
            if isinstance(default_val, pd.Series):
                df[col] = default_val
            else:
                df[col] = default_val

    # Use ONLY the 25 features, in sorted order (LightGBM is picky about feature order)
    feature_df = df[sorted(exact_features)].copy()
    
    # Final safety check: replace any NaN/inf with defaults
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    for col in feature_df.columns:
        if feature_df[col].isna().any():
            # Use median or mean for that column
            fill_val = feature_df[col].median() if feature_df[col].notna().any() else 0.0
            if pd.isna(fill_val):
                fill_val = 0.0
            feature_df[col] = feature_df[col].fillna(fill_val)

    # Debug: Print feature statistics
    print(f"📊 Feature Engineering Summary:")
    print(f"   - Features created: {len(exact_features)}")
    print(f"   - Feature ranges:")
    for col in sorted(exact_features)[:5]:  # Show first 5
        if col in feature_df.columns:
            print(f"     {col}: [{feature_df[col].min():.2f}, {feature_df[col].max():.2f}]")
    print(f"   - Any NaN values: {feature_df.isna().sum().sum()}")

    # Load model if not already loaded
    try:
        current_model = load_model()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Cannot perform predictions without the model.\n"
            f"Please ensure fraud_lgbm_calibrated.pkl is in the same directory as app.py"
        )
    
    # Predict calibrated probabilities
    try:
        prob = current_model.predict_proba(feature_df)[:, 1]
        print(f"   - Probability range: [{prob.min():.4f}, {prob.max():.4f}]")
        print(f"   - Mean probability: {prob.mean():.4f}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        print(f"   Model expects {current_model.n_features_in_} features, got {feature_df.shape[1]}")
        print(f"   Feature columns: {list(feature_df.columns)}")
        raise
    
    df["fraud_probability"] = prob
    df["fraud_prediction"] = (prob >= threshold).astype(int)

    # Risk levels
    conditions = [
        df["fraud_probability"] < 0.1,
        df["fraud_probability"] < 0.5,
        df["fraud_probability"] < 0.9,
    ]
    choices = ["Low", "Medium", "High"]
    df["risk_level"] = np.select(conditions, choices, default="Critical")

    return df


def create_visualizations(result_df: pd.DataFrame, threshold: float, model_metrics: dict = None):
    """Create comprehensive visualizations for fraud detection results."""
    visualizations = []
    
    # 1. Fraud Probability Distribution
    fig1 = go.Figure()
    if len(result_df) > 0:
        fig1.add_trace(go.Histogram(
            x=result_df["fraud_probability"],
            nbinsx=min(50, len(result_df) // 2),
            name="All Transactions",
            marker_color="lightblue",
            opacity=0.7
        ))
        fraud_df = result_df[result_df["fraud_prediction"] == 1]
        if len(fraud_df) > 0:
            fig1.add_trace(go.Histogram(
                x=fraud_df["fraud_probability"],
                nbinsx=min(50, len(fraud_df) // 2),
                name="Flagged as Fraud",
                marker_color="red",
                opacity=0.8
            ))
        fig1.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Threshold: {threshold:.2f}"
        )
    fig1.update_layout(
        title="Fraud Probability Distribution",
        xaxis_title="Fraud Probability",
        yaxis_title="Count",
        height=400,
        showlegend=True
    )
    visualizations.append(fig1)

    # 2. Risk Level Breakdown (Pie Chart)
    if len(result_df) > 0:
        risk_counts = result_df["risk_level"].value_counts()
        color_map = {"Low": "green", "Medium": "yellow", "High": "orange", "Critical": "red"}
        colors = [color_map.get(label, "gray") for label in risk_counts.index]
        fig2 = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker_colors=colors
        )])
        fig2.update_layout(
            title="Risk Level Distribution",
            height=400
        )
    else:
        fig2 = go.Figure()
        fig2.update_layout(title="Risk Level Distribution (No Data)", height=400)
    visualizations.append(fig2)

    # 3. Amount vs Fraud Probability (Scatter)
    amount_col = None
    for col in ["amt", "amount", "Amount", "TransactionAmt"]:
        if col in result_df.columns:
            amount_col = col
            break
    
    if amount_col and len(result_df) > 0:
        fig3 = go.Figure()
        fraud_df = result_df[result_df["fraud_prediction"] == 1]
        normal_df = result_df[result_df["fraud_prediction"] == 0]
        
        if len(normal_df) > 0:
            fig3.add_trace(go.Scatter(
                x=normal_df[amount_col],
                y=normal_df["fraud_probability"],
                mode="markers",
                name="Normal",
                marker=dict(color="blue", size=4, opacity=0.5)
            ))
        if len(fraud_df) > 0:
            fig3.add_trace(go.Scatter(
                x=fraud_df[amount_col],
                y=fraud_df["fraud_probability"],
                mode="markers",
                name="Fraud",
                marker=dict(color="red", size=6, opacity=0.8)
            ))
        fig3.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Threshold: {threshold:.2f}"
        )
        fig3.update_layout(
            title="Transaction Amount vs Fraud Probability",
            xaxis_title=f"Amount ({amount_col})",
            yaxis_title="Fraud Probability",
            height=400,
            showlegend=True
        )
        visualizations.append(fig3)
    else:
        fig3 = go.Figure()
        fig3.update_layout(
            title="Transaction Amount vs Fraud Probability (Amount column not found)",
            height=400
        )
        visualizations.append(fig3)

    # 4. Fraud Statistics Bar Chart
    if len(result_df) > 0:
        total = len(result_df)
        fraud_count = int(result_df["fraud_prediction"].sum())
        fraud_rate = (fraud_count / total) * 100 if total > 0 else 0.0
        
        fig4 = go.Figure(data=[
            go.Bar(
                x=["Total Transactions", "Fraud Detected", "Normal"],
                y=[total, fraud_count, total - fraud_count],
                marker_color=["blue", "red", "green"],
                text=[f"{total:,}", f"{fraud_count:,}", f"{total - fraud_count:,}"],
                textposition="auto"
            )
        ])
        fig4.update_layout(
            title="Transaction Summary",
            yaxis_title="Count",
            height=400
        )
    else:
        fig4 = go.Figure()
        fig4.update_layout(title="Transaction Summary (No Data)", height=400)
    visualizations.append(fig4)

    # 5. Fraud Probability Over Time (if time column exists)
    time_col = None
    for col in ["Time", "time", "unix_time", "timestamp"]:
        if col in result_df.columns:
            time_col = col
            break
    
    if time_col and len(result_df) > 0:
        # Sort by time and create rolling average
        sorted_df = result_df.sort_values(time_col).copy()
        window_size = max(1, min(100, len(sorted_df) // 10))
        sorted_df["rolling_avg"] = sorted_df["fraud_probability"].rolling(
            window=window_size, center=True
        ).mean()
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=sorted_df[time_col],
            y=sorted_df["fraud_probability"],
            mode="markers",
            name="Individual",
            marker=dict(size=3, opacity=0.3, color="lightblue")
        ))
        if sorted_df["rolling_avg"].notna().any():
            fig5.add_trace(go.Scatter(
                x=sorted_df[time_col],
                y=sorted_df["rolling_avg"],
                mode="lines",
                name="Rolling Average",
                line=dict(color="blue", width=2)
            ))
        fig5.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Threshold: {threshold:.2f}"
        )
        fig5.update_layout(
            title="Fraud Probability Over Time",
            xaxis_title=time_col,
            yaxis_title="Fraud Probability",
            height=400,
            showlegend=True
        )
        visualizations.append(fig5)
    else:
        fig5 = go.Figure()
        fig5.update_layout(
            title="Fraud Probability Over Time (Time column not found)",
            height=400
        )
        visualizations.append(fig5)

    # 6. Top 20 Most Suspicious Transactions
    if len(result_df) > 0:
        top_suspicious = result_df.nlargest(min(20, len(result_df)), "fraud_probability")
        fig6 = go.Figure(data=[
            go.Bar(
                x=[str(i) for i in top_suspicious.index],
                y=top_suspicious["fraud_probability"],
                marker_color=top_suspicious["fraud_probability"],
                marker_colorscale="Reds",
                text=top_suspicious["fraud_probability"].round(3),
                textposition="auto"
            )
        ])
        fig6.update_layout(
            title="Top 20 Most Suspicious Transactions",
            xaxis_title="Transaction Index",
            yaxis_title="Fraud Probability",
            height=400
        )
        visualizations.append(fig6)
    else:
        fig6 = go.Figure()
        fig6.update_layout(title="Top 20 Most Suspicious Transactions (No Data)", height=400)
        visualizations.append(fig6)

    # 7. Model Performance Metrics (if available)
    if model_metrics and len(model_metrics) > 0:
        fig7 = go.Figure()
        
        # Create metrics bar chart - include PR-AUC for imbalanced data
        metrics_names = ['ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [
            model_metrics.get('roc_auc', 0),
            model_metrics.get('pr_auc', 0),
            model_metrics.get('precision', 0),
            model_metrics.get('recall', 0),
            model_metrics.get('f1', 0)
        ]
        
        colors = ['#1f77b4', '#9467bd', '#ff7f0e', '#2ca02c', '#d62728']
        fig7.add_trace(go.Bar(
            x=metrics_names,
            y=metrics_values,
            marker_color=colors,
            text=[f'{v:.4f}' for v in metrics_values],
            textposition='auto'
        ))
        fig7.update_layout(
            title="Model Performance Metrics (PR-AUC better for imbalanced data)",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400
        )
        visualizations.append(fig7)
        
        # 8. Confusion Matrix Heatmap
        if all(k in model_metrics for k in ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']):
            fig8 = go.Figure(data=go.Heatmap(
                z=[[model_metrics['true_negatives'], model_metrics['false_positives']],
                   [model_metrics['false_negatives'], model_metrics['true_positives']]],
                x=['Predicted Normal', 'Predicted Fraud'],
                y=['Actual Normal', 'Actual Fraud'],
                colorscale='Blues',
                text=[[model_metrics['true_negatives'], model_metrics['false_positives']],
                      [model_metrics['false_negatives'], model_metrics['true_positives']]],
                texttemplate='%{text}',
                textfont={"size": 16},
                colorbar=dict(title="Count")
            ))
            fig8.update_layout(
                title="Confusion Matrix",
                height=400
            )
            visualizations.append(fig8)
        else:
            visualizations.append(None)
    else:
        # Add None placeholders if no metrics
        visualizations.append(None)
        visualizations.append(None)

    # ============================================================================
    # ADVANCED VISUALIZATIONS
    # ============================================================================
    
    # 9. ROC Curve (if ground truth available)
    if model_metrics and 'roc_auc' in model_metrics:
        try:
            # Get ground truth and predictions
            ground_truth_cols = ["is_fraud", "fraud", "target", "label", "isFraud"]
            y_true = None
            for col in ground_truth_cols:
                if col in result_df.columns:
                    y_true = result_df[col]
                    break
            
            if y_true is not None:
                y_prob = result_df["fraud_probability"]
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
                
                fig9 = go.Figure()
                fig9.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {model_metrics["roc_auc"]:.4f})',
                    line=dict(color='blue', width=2)
                ))
                fig9.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='red', dash='dash', width=1)
                ))
                fig9.update_layout(
                    title="ROC Curve (Receiver Operating Characteristic)",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=400,
                    showlegend=True
                )
                visualizations.append(fig9)
            else:
                visualizations.append(None)
        except Exception as e:
            print(f"Warning: Could not create ROC curve: {e}")
            visualizations.append(None)
    else:
        visualizations.append(None)
    
    # 10. Precision-Recall Curve (if ground truth available)
    if model_metrics and 'pr_auc' in model_metrics:
        try:
            ground_truth_cols = ["is_fraud", "fraud", "target", "label", "isFraud"]
            y_true = None
            for col in ground_truth_cols:
                if col in result_df.columns:
                    y_true = result_df[col]
                    break
            
            if y_true is not None:
                y_prob = result_df["fraud_probability"]
                precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
                
                fig10 = go.Figure()
                fig10.add_trace(go.Scatter(
                    x=recalls,
                    y=precisions,
                    mode='lines',
                    name=f'PR Curve (AUC = {model_metrics["pr_auc"]:.4f})',
                    line=dict(color='green', width=2),
                    fill='tonexty'
                ))
                fig10.add_hline(
                    y=y_true.mean(),
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Baseline (Fraud Rate: {y_true.mean():.2%})"
                )
                fig10.update_layout(
                    title="Precision-Recall Curve (Better for Imbalanced Data)",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    height=400,
                    showlegend=True
                )
                visualizations.append(fig10)
            else:
                visualizations.append(None)
        except Exception as e:
            print(f"Warning: Could not create PR curve: {e}")
            visualizations.append(None)
    else:
        visualizations.append(None)
    
    # 11. Amount Distribution Comparison (Box Plot)
    amount_col = None
    for col in ["amt", "amount", "Amount", "TransactionAmt"]:
        if col in result_df.columns:
            amount_col = col
            break
    
    if amount_col and len(result_df) > 0:
        fraud_df = result_df[result_df["fraud_prediction"] == 1]
        normal_df = result_df[result_df["fraud_prediction"] == 0]
        
        fig11 = go.Figure()
        if len(normal_df) > 0:
            fig11.add_trace(go.Box(
                y=normal_df[amount_col],
                name="Normal Transactions",
                marker_color="blue",
                boxmean='sd'
            ))
        if len(fraud_df) > 0:
            fig11.add_trace(go.Box(
                y=fraud_df[amount_col],
                name="Flagged as Fraud",
                marker_color="red",
                boxmean='sd'
            ))
        fig11.update_layout(
            title="Transaction Amount Distribution Comparison",
            yaxis_title=f"Amount ({amount_col})",
            height=400,
            showlegend=True
        )
        visualizations.append(fig11)
    else:
        fig11 = go.Figure()
        fig11.update_layout(
            title="Transaction Amount Distribution (Amount column not found)",
            height=400
        )
        visualizations.append(fig11)
    
    # 12. Fraud Probability Distribution Comparison (Violin Plot)
    if len(result_df) > 0:
        ground_truth_cols = ["is_fraud", "fraud", "target", "label", "isFraud"]
        has_gt = any(col in result_df.columns for col in ground_truth_cols)
        
        if has_gt:
            gt_col = next((col for col in ground_truth_cols if col in result_df.columns), None)
            if gt_col:
                fraud_probs = result_df[result_df[gt_col] == 1]["fraud_probability"]
                normal_probs = result_df[result_df[gt_col] == 0]["fraud_probability"]
                
                fig12 = go.Figure()
                if len(normal_probs) > 0:
                    fig12.add_trace(go.Violin(
                        y=normal_probs,
                        name="Actual Normal",
                        box_visible=True,
                        meanline_visible=True,
                        marker_color="blue"
                    ))
                if len(fraud_probs) > 0:
                    fig12.add_trace(go.Violin(
                        y=fraud_probs,
                        name="Actual Fraud",
                        box_visible=True,
                        meanline_visible=True,
                        marker_color="red"
                    ))
                fig12.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Threshold: {threshold:.2f}"
                )
                fig12.update_layout(
                    title="Fraud Probability Distribution: Actual Fraud vs Normal",
                    yaxis_title="Fraud Probability",
                    height=400,
                    showlegend=True
                )
                visualizations.append(fig12)
            else:
                visualizations.append(None)
        else:
            # Use predicted labels if no ground truth
            fraud_probs = result_df[result_df["fraud_prediction"] == 1]["fraud_probability"]
            normal_probs = result_df[result_df["fraud_prediction"] == 0]["fraud_probability"]
            
            fig12 = go.Figure()
            if len(normal_probs) > 0:
                fig12.add_trace(go.Violin(
                    y=normal_probs,
                    name="Predicted Normal",
                    box_visible=True,
                    meanline_visible=True,
                    marker_color="blue"
                ))
            if len(fraud_probs) > 0:
                fig12.add_trace(go.Violin(
                    y=fraud_probs,
                    name="Predicted Fraud",
                    box_visible=True,
                    meanline_visible=True,
                    marker_color="red"
                ))
            fig12.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Threshold: {threshold:.2f}"
            )
            fig12.update_layout(
                title="Fraud Probability Distribution: Predicted Fraud vs Normal",
                yaxis_title="Fraud Probability",
                height=400,
                showlegend=True
            )
            visualizations.append(fig12)
    else:
        visualizations.append(None)
    
    # 13. Feature Correlation Heatmap (if multiple features available)
    if len(result_df) > 0:
        try:
            # Select numeric features for correlation
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude probability and prediction columns for cleaner visualization
            exclude_cols = ['fraud_probability', 'fraud_prediction', 'is_fraud', 'fraud', 'target', 'label']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Limit to top 15 features for readability
            if len(feature_cols) > 15:
                # Select features with highest variance
                variances = result_df[feature_cols].var().sort_values(ascending=False)
                feature_cols = variances.head(15).index.tolist()
            
            if len(feature_cols) > 1:
                corr_matrix = result_df[feature_cols + ['fraud_probability']].corr()
                
                fig13 = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.round(2).values,
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))
                fig13.update_layout(
                    title="Feature Correlation Heatmap (Top Features)",
                    height=500,
                    width=600
                )
                visualizations.append(fig13)
            else:
                visualizations.append(None)
        except Exception as e:
            print(f"Warning: Could not create correlation heatmap: {e}")
            visualizations.append(None)
    else:
        visualizations.append(None)
    
    # 14. Cumulative Fraud Detection Over Time
    time_col = None
    for col in ["Time", "time", "unix_time", "timestamp"]:
        if col in result_df.columns:
            time_col = col
            break
    
    if time_col and len(result_df) > 0:
        sorted_df = result_df.sort_values(time_col).copy()
        sorted_df['cumulative_fraud'] = sorted_df['fraud_prediction'].cumsum()
        sorted_df['cumulative_total'] = range(1, len(sorted_df) + 1)
        sorted_df['fraud_rate'] = sorted_df['cumulative_fraud'] / sorted_df['cumulative_total']
        
        fig14 = go.Figure()
        fig14.add_trace(go.Scatter(
            x=sorted_df[time_col],
            y=sorted_df['cumulative_fraud'],
            mode='lines',
            name='Cumulative Fraud Detected',
            line=dict(color='red', width=2)
        ))
        fig14.add_trace(go.Scatter(
            x=sorted_df[time_col],
            y=sorted_df['fraud_rate'] * 100,
            mode='lines',
            name='Fraud Rate (%)',
            yaxis='y2',
            line=dict(color='orange', width=2, dash='dash')
        ))
        fig14.update_layout(
            title="Cumulative Fraud Detection Over Time",
            xaxis_title=time_col,
            yaxis_title="Cumulative Fraud Count",
            yaxis2=dict(
                title="Fraud Rate (%)",
                overlaying='y',
                side='right'
            ),
            height=400,
            showlegend=True
        )
        visualizations.append(fig14)
    else:
        visualizations.append(None)
    
    # 15. Threshold Sensitivity Analysis (if ground truth available)
    if model_metrics:
        try:
            ground_truth_cols = ["is_fraud", "fraud", "target", "label", "isFraud"]
            y_true = None
            for col in ground_truth_cols:
                if col in result_df.columns:
                    y_true = result_df[col]
                    break
            
            if y_true is not None and len(y_true) > 0:
                y_prob = result_df["fraud_probability"]
                thresholds = np.linspace(0.01, 0.99, 50)
                precisions_list = []
                recalls_list = []
                f1_scores_list = []
                
                for t in thresholds:
                    y_pred = (y_prob >= t).astype(int)
                    if y_pred.sum() > 0:
                        prec, rec, f1, _ = precision_recall_fscore_support(
                            y_true, y_pred, average='binary', zero_division=0
                        )
                        precisions_list.append(prec)
                        recalls_list.append(rec)
                        f1_scores_list.append(f1)
                    else:
                        precisions_list.append(0)
                        recalls_list.append(0)
                        f1_scores_list.append(0)
                
                fig15 = go.Figure()
                fig15.add_trace(go.Scatter(
                    x=thresholds,
                    y=precisions_list,
                    mode='lines',
                    name='Precision',
                    line=dict(color='blue', width=2)
                ))
                fig15.add_trace(go.Scatter(
                    x=thresholds,
                    y=recalls_list,
                    mode='lines',
                    name='Recall',
                    line=dict(color='green', width=2)
                ))
                fig15.add_trace(go.Scatter(
                    x=thresholds,
                    y=f1_scores_list,
                    mode='lines',
                    name='F1-Score',
                    line=dict(color='red', width=2)
                ))
                fig15.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Current: {threshold:.2f}"
                )
                fig15.update_layout(
                    title="Threshold Sensitivity Analysis",
                    xaxis_title="Threshold",
                    yaxis_title="Score",
                    height=400,
                    showlegend=True
                )
                visualizations.append(fig15)
            else:
                visualizations.append(None)
        except Exception as e:
            print(f"Warning: Could not create threshold sensitivity: {e}")
            visualizations.append(None)
    else:
        visualizations.append(None)

    return visualizations


def predict_fraud_enhanced(file, threshold: float = 0.5):
    """
    Enhanced fraud detection with comprehensive visualizations.
    """
    print("=" * 60)
    print("FUNCTION CALLED: predict_fraud_enhanced")
    print(f"File received: {file}")
    print(f"Threshold: {threshold}")
    print("=" * 60)
    
    if file is None:
        print("ERROR: File is None")
        return (
            "Please upload a CSV file.",
            pd.DataFrame(),
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        )

    try:
        start = time.time()

        # Read CSV (limit to 10000 rows for performance)
        # Handle different Gradio file input formats (Gradio 6.0 compatibility)
        # With type="filepath", Gradio returns a string path directly
        file_path = None
        
        # Handle Gradio 6.0 file input - comprehensive file path extraction
        file_path = None
        
        # Method 1: Direct string path (most common in Gradio 6.0)
        # Also handle if user provides a relative path like "dataset/fraudTest.csv"
        if isinstance(file, str):
            file_path = file
            # If it's a relative path and exists, use it directly
            if not os.path.isabs(file_path) and os.path.exists(file_path):
                print(f"DEBUG: Using relative path: {file_path}")
            else:
                print(f"DEBUG: File is string: {file_path}")
        
        # Method 2: FileData object (check all possible attributes)
        elif file is not None:
            # Try all possible file path attributes
            for attr in ['path', 'name', 'orig_name', 'file_path', 'file_name']:
                if hasattr(file, attr):
                    value = getattr(file, attr)
                    if value and isinstance(value, str) and len(value) > 3:
                        file_path = value
                        print(f"DEBUG: Found file path via {attr}: {file_path}")
                        break
            
            # If still no path, try tuple/list
            if not file_path and isinstance(file, (list, tuple)) and len(file) > 0:
                file_path = file[0] if isinstance(file[0], str) else str(file[0])
                print(f"DEBUG: File is tuple/list: {file_path}")
            
            # Last resort: convert to string
            if not file_path:
                file_str = str(file)
                print(f"DEBUG: File as string: {file_str}")
                # Check if it's a valid path-like string
                if file_str and file_str != 'file' and file_str != 'None' and len(file_str) > 3:
                    if os.path.sep in file_str or file_str.endswith('.csv'):
                        file_path = file_str
                        print(f"DEBUG: Extracted path from string: {file_path}")
        
        print(f"DEBUG: Final file_path: {file_path}")
        
        # Validate file path exists
        if not file_path:
            error_msg = f"Error: Could not determine file path. File type: {type(file)}, File value: {file}"
            print(error_msg)
            return (
                error_msg,
                pd.DataFrame(),
                None, None, None, None, None, None, None, None
            )
        
        if not os.path.exists(file_path):
            error_msg = f"Error: File not found at path: {file_path}. Please try uploading the file again."
            print(error_msg)
            return (
                error_msg,
                pd.DataFrame(),
                None, None, None, None, None, None, None, None
            )
        
        print(f"DEBUG: File exists, reading CSV from: {file_path}")
        
        # Read CSV with error handling for large files and permission issues
        # Windows/Gradio temp files often have permission issues, so we try multiple methods
        df = None
        read_methods_tried = []
        
        # Method 1: Try reading in binary mode first (most reliable for locked files)
        try:
            print("Trying binary read method (most reliable for Windows temp files)...")
            with open(file_path, 'rb') as f:
                content = f.read()
            df = pd.read_csv(BytesIO(content), nrows=10000, low_memory=False)
            print("✓ File read successfully using binary method")
        except (PermissionError, IOError, OSError) as binary_error:
            read_methods_tried.append(f"Binary read: {str(binary_error)}")
            print(f"⚠ Binary read failed, trying direct read...")
            
            # Method 2: Try to read directly
            try:
                df = pd.read_csv(file_path, nrows=10000, low_memory=False)
                print("✓ File read successfully using direct method")
            except (PermissionError, IOError, OSError) as perm_error:
                read_methods_tried.append(f"Direct read: {str(perm_error)}")
                print(f"⚠ Permission denied, trying copy methods...")
                
                # Method 3: Try copying to project directory (more reliable)
                try:
                    print("Trying copy to project directory...")
                    project_temp = os.path.join(os.getcwd(), f"temp_upload_{uuid.uuid4().hex[:8]}.csv")
                    # Use shutil.copy instead of copy2 to avoid metadata issues
                    shutil.copy(file_path, project_temp)
                    # Small delay to ensure file is fully copied
                    time.sleep(0.1)
                    df = pd.read_csv(project_temp, nrows=10000, low_memory=False)
                    # Clean up
                    try:
                        os.remove(project_temp)
                        print("✓ Temporary file cleaned up")
                    except:
                        pass
                    print("✓ File read successfully after copying to project directory")
                except Exception as copy_error:
                    read_methods_tried.append(f"Copy method: {str(copy_error)}")
                    
                    # Method 4: Try system temp directory with different approach
                    try:
                        print("Trying system temp directory with alternative method...")
                        temp_dir = tempfile.gettempdir()
                        temp_filename = f"gradio_upload_{uuid.uuid4().hex[:8]}.csv"
                        temp_file = os.path.join(temp_dir, temp_filename)
                        # Read and write in chunks to avoid locking
                        with open(file_path, 'rb') as src, open(temp_file, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        df = pd.read_csv(temp_file, nrows=10000, low_memory=False)
                        # Clean up
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                        print("✓ File read successfully using system temp directory")
                    except Exception as temp_error:
                        read_methods_tried.append(f"System temp: {str(temp_error)}")
                        
                        # All methods failed - provide helpful error message
                        error_details = "\n".join([f"- {m}" for m in read_methods_tried])
                        return (
                            f"❌ Error: Could not read file due to permission/access issues.\n\n"
                            f"**Quick Fix:**\n"
                            f"1. **Copy the file to the project folder first** (e.g., `dataset/fraudTest.csv`)\n"
                            f"2. Then upload it from there - this avoids Windows temp file permission issues\n\n"
                            f"**Alternative Solutions:**\n"
                            f"- Close any programs that might be using the file (Excel, text editor, etc.)\n"
                            f"- Try uploading the file again (sometimes works on retry)\n"
                            f"- If using Windows, try running the app as Administrator\n"
                            f"- Use the file directly: The app can process `dataset/fraudTest.csv` if it exists\n\n"
                            f"**Why this happens:**\n"
                            f"Gradio stores uploaded files in Windows temp directory which can have permission restrictions.\n"
                            f"Copying the file to your project folder first avoids this issue.\n\n"
                            f"**Technical details:**\n{error_details}",
                            pd.DataFrame(),
                            None, None, None, None, None, None, None, None
                        )
        except Exception as csv_error:
            # Handle CSV parsing errors separately
            if "Permission denied" in str(csv_error) or "Errno 13" in str(csv_error):
                return (
                    f"❌ Permission Error: Could not access the uploaded file.\n\n"
                    f"**Solution:** Copy your CSV file to the project folder (e.g., `dataset/fraudTest.csv`) "
                    f"and upload it from there. This avoids Windows temp file permission issues.\n\n"
                    f"**Error:** {str(csv_error)}",
                    pd.DataFrame(),
                    None, None, None, None, None, None, None, None
                )
            return (
                f"Error reading CSV file: {str(csv_error)}\n\nPlease ensure the file is a valid CSV format.",
                pd.DataFrame(),
                None, None, None, None, None, None, None, None
            )
        
        if df is None:
            return (
                "Error: Could not read the CSV file. Please try uploading again.",
                pd.DataFrame(),
                None, None, None, None, None, None, None, None
            )

        if df.empty:
            return (
                "CSV is empty.",
                pd.DataFrame(),
                None, None, None, None, None, None, None, None
            )

        print(f"Processing {len(df)} rows...")
        print(f"CSV columns: {list(df.columns)[:10]}...")  # Print first 10 columns for debugging
        print(f"Total columns in CSV: {len(df.columns)}")

        # Score the batch
        try:
            print("Starting score_batch...")
            result_df = score_batch(df, threshold=threshold)
            print(f"Score_batch completed. Result shape: {result_df.shape}")
            print(f"Fraud probabilities range: {result_df['fraud_probability'].min():.4f} to {result_df['fraud_probability'].max():.4f}")
        except FileNotFoundError as model_error:
            # Handle missing model file with user-friendly message
            error_msg = (
                f"❌ **Model File Not Found**\n\n"
                f"The fraud detection model (`fraud_lgbm_calibrated.pkl`) is required to make predictions.\n\n"
                f"**To fix this:**\n"
                f"1. Download the model file from your GitHub repository\n"
                f"2. Place it in the same directory as `app.py`\n"
                f"3. Or train a new model using the notebook: `notebooks/eda_and_feature_engineering.ipynb`\n\n"
                f"**Error details:** {str(model_error)}"
            )
            print(f"MODEL ERROR: {error_msg}")
            return (
                error_msg,
                pd.DataFrame(),
                None, None, None, None, None, None, None, None, None, None, None, None, None, None
            )
        except Exception as score_error:
            import traceback
            error_details = f"Error during scoring: {str(score_error)}\n\n{traceback.format_exc()}"
            print(f"SCORING ERROR: {error_details}")
            return (
                f"❌ {error_details}",
                pd.DataFrame(),
                None, None, None, None, None, None, None, None, None, None, None, None, None, None
            )

        # Calculate statistics
        total = len(result_df)
        fraud_count = int(result_df["fraud_prediction"].sum())
        fraud_rate = (fraud_count / total) * 100 if total > 0 else 0.0
        avg_prob = result_df["fraud_probability"].mean()
        max_prob = result_df["fraud_probability"].max()
        min_prob = result_df["fraud_probability"].min()
        median_prob = result_df["fraud_probability"].median()

        elapsed = time.time() - start

        # Calculate model performance metrics if ground truth is available
        model_metrics = {}
        has_ground_truth = False
        
        # Check for ground truth labels (common column names)
        ground_truth_cols = ["is_fraud", "fraud", "target", "label", "isFraud"]
        y_true = None
        for col in ground_truth_cols:
            if col in result_df.columns:
                y_true = result_df[col]
                has_ground_truth = True
                print(f"Found ground truth column: {col}")
                break
        
        if has_ground_truth and y_true is not None:
            try:
                y_prob = result_df["fraud_probability"]
                y_pred = result_df["fraud_prediction"]
                
                # Calculate metrics at current threshold
                model_metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0
                )
                model_metrics["precision"] = float(precision)
                model_metrics["recall"] = float(recall)
                model_metrics["f1"] = float(f1)
                
                # PR-AUC (better for imbalanced datasets)
                model_metrics["pr_auc"] = average_precision_score(y_true, y_prob)
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                model_metrics["true_negatives"] = int(cm[0, 0])
                model_metrics["false_positives"] = int(cm[0, 1])
                model_metrics["false_negatives"] = int(cm[1, 0])
                model_metrics["true_positives"] = int(cm[1, 1])
                
                print(f"Model metrics calculated: AUC={model_metrics['roc_auc']:.4f}, PR-AUC={model_metrics['pr_auc']:.4f}")
            except Exception as e:
                print(f"Warning: Could not calculate metrics: {str(e)}")
                import traceback
                print(traceback.format_exc())
                has_ground_truth = False

        # Create summary text with model metrics
        summary = f"""
## 📊 Fraud Detection Results

**Processing Summary:**
- ✅ Processed **{total:,}** transactions in **{elapsed:.2f}s**
- 🎯 Detection threshold: **{threshold:.2f}**

**Fraud Statistics:**
- 🚨 **Fraud flagged:** {fraud_count:,} transactions ({fraud_rate:.2f}%)
- ✅ **Normal transactions:** {total - fraud_count:,} ({(100 - fraud_rate):.2f}%)

**Probability Statistics:**
- 📈 **Average probability:** {avg_prob:.4f}
- 📊 **Median probability:** {median_prob:.4f}
- ⬆️ **Maximum probability:** {max_prob:.4f}
- ⬇️ **Minimum probability:** {min_prob:.4f}

**Risk Level Breakdown:**
{result_df['risk_level'].value_counts().to_string()}
        """
        
        # Add model performance metrics if available
        if has_ground_truth and model_metrics:
            # Calculate additional metrics
            tp = model_metrics.get('true_positives', 0)
            fp = model_metrics.get('false_positives', 0)
            fn = model_metrics.get('false_negatives', 0)
            tn = model_metrics.get('true_negatives', 0)
            
            summary += f"""

---

## 🎯 Model Performance Metrics

**Current Threshold ({threshold:.2f}) Performance:**
- 📊 **ROC-AUC:** {model_metrics['roc_auc']:.4f} {'✅ Good discrimination' if model_metrics['roc_auc'] > 0.8 else '⚠️ Needs improvement'}
- 📈 **PR-AUC (Better for imbalanced data):** {model_metrics.get('pr_auc', 0):.4f} {'✅ Good for imbalanced data' if model_metrics.get('pr_auc', 0) > 0.1 else '⚠️ Very low - model struggles with imbalance'}
- 🎯 **Precision:** {model_metrics['precision']:.4f} ({int((1-model_metrics['precision'])*100)}% false positives)
- 🔍 **Recall:** {model_metrics['recall']:.4f} ({int((1-model_metrics['recall'])*100)}% fraud missed)
- ⚖️ **F1-Score:** {model_metrics['f1']:.4f} {'✅ Balanced' if model_metrics['f1'] > 0.2 else '⚠️ Poor balance'}

**Confusion Matrix:**
- ✅ **True Negatives (Correctly identified normal):** {tn:,}
- ❌ **False Positives (Normal flagged as fraud):** {fp:,}
- ❌ **False Negatives (Fraud missed):** {fn:,}
- ✅ **True Positives (Correctly identified fraud):** {tp:,}

**Accuracy:** {(tp + tn) / total * 100:.2f}%
        """

        # Export to Power BI (automatic)
        if POWERBI_AVAILABLE:
            try:
                export_powerbi_csvs(result_df)
                print("✅ Data exported to Power BI format")
            except Exception as e:
                print(f"⚠️ Power BI export failed: {e}")
        
        # Get top suspicious transactions
        top_df = result_df.nlargest(20, "fraud_probability")
        
        # Select columns to display
        display_cols = ["fraud_probability", "fraud_prediction", "risk_level"]
        
        # Add amount column if available
        for amt_col in ["amt", "amount", "Amount", "TransactionAmt"]:
            if amt_col in top_df.columns:
                display_cols.insert(1, amt_col)
                break

        # Add first few original columns
        for col in result_df.columns:
            if col not in ["fraud_probability", "fraud_prediction", "risk_level"] and col not in display_cols:
                if len(display_cols) < 8:  # Limit display columns
                    display_cols.append(col)
                else:
                    break

        display_cols = [c for c in display_cols if c in top_df.columns]
        top_display = top_df[display_cols].copy()
        
        # Format probability column for display
        if "fraud_probability" in top_display.columns:
            top_display["fraud_probability"] = top_display["fraud_probability"].round(4)

        # Create visualizations (pass model_metrics if available)
        try:
            viz_list = create_visualizations(result_df, threshold, model_metrics if has_ground_truth else None)
        except Exception as viz_error:
            import traceback
            print(f"Warning: Error creating some visualizations: {str(viz_error)}")
            print(traceback.format_exc())
            # Create empty visualizations list if there's an error
            viz_list = []
        
        # Ensure we always return 15 visualizations
        while len(viz_list) < 15:
            viz_list.append(None)
        
        # Return all outputs (summary, table, and 15 visualizations)
        print(f"Returning results: summary length={len(summary)}, top_display shape={top_display.shape}, viz_count={len([v for v in viz_list if v is not None])}")
        return (
            summary,
            top_display,
            viz_list[0],  # Probability Distribution
            viz_list[1],  # Risk Level Breakdown
            viz_list[2],  # Amount Analysis
            viz_list[3],  # Statistics
            viz_list[4],  # Time Series
            viz_list[5],  # Top 20 Suspicious
            viz_list[6] if len(viz_list) > 6 else None,  # Model Performance Metrics
            viz_list[7] if len(viz_list) > 7 else None,  # Confusion Matrix
            viz_list[8] if len(viz_list) > 8 else None,  # ROC Curve
            viz_list[9] if len(viz_list) > 9 else None,  # Precision-Recall Curve
            viz_list[10] if len(viz_list) > 10 else None,  # Amount Distribution (Box Plot)
            viz_list[11] if len(viz_list) > 11 else None,  # Probability Distribution (Violin)
            viz_list[12] if len(viz_list) > 12 else None,  # Correlation Heatmap
            viz_list[13] if len(viz_list) > 13 else None,  # Cumulative Fraud Detection
            viz_list[14] if len(viz_list) > 14 else None,  # Threshold Sensitivity
        )

    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(f"FATAL ERROR: {error_msg}")
        return (
            error_msg,
            pd.DataFrame(),
            None, None, None, None, None, None, None, None
        )


# Create Gradio Interface
with gr.Blocks(
    title="Credit Card Fraud Detection",
    analytics_enabled=False
) as demo:
    
    gr.Markdown("""
    # 🛡️ Credit Card Fraud Detection System
    
    Upload a CSV file with transaction data to detect potential fraud. The system uses a calibrated 
    LightGBM model to predict fraud probabilities and provides comprehensive visualizations.
    
    **Supported formats:** CSV files with numeric features (V1-V28, Amount, Time, etc.)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📁 Upload Transaction CSV",
                file_count="single"
            )
            # Add button to use dataset file directly (avoids permission issues)
            use_dataset_btn = gr.Button(
                "📂 Use dataset/fraudTest.csv (Skip Upload)",
                variant="secondary",
                size="sm"
            )
            # Add Sample button to use sample dataset
            sample_btn = gr.Button(
                "📊 Use Sample Dataset",
                variant="secondary",
                size="sm"
            )
            gr.Markdown("""
            **💡 Tip:** If you get permission errors, click the button above to use the dataset file directly!
            **🎯 Try Sample:** Click "Use Sample Dataset" to test the model with sample data!
            """)
        with gr.Column(scale=1):
            threshold = gr.Slider(
                minimum=0.01,
                maximum=0.99,
                value=0.05,
                step=0.01,
                label="🎯 Fraud Detection Threshold",
                info="⚠️ For imbalanced data, use 0.01-0.05 for better recall. Default 0.5 is too high!"
            )
    
    # Function to use dataset file directly
    def use_dataset_file():
        dataset_path = "dataset/fraudTest.csv"
        if os.path.exists(dataset_path):
            # Return the file path as a string - the predict function will handle it
            return dataset_path
        else:
            return None
    
    # When button is clicked, trigger prediction with dataset file
    def use_dataset_and_predict(threshold_val):
        dataset_path = "dataset/fraudTest.csv"
        if os.path.exists(dataset_path):
            # Call predict function directly with dataset path
            return predict_fraud_enhanced(dataset_path, threshold_val)
        else:
            return (
                "❌ Error: dataset/fraudTest.csv not found. Please upload a file instead.",
                pd.DataFrame(),
                None, None, None, None, None, None, None, None
            )
    
    # When Sample button is clicked, use sample dataset
    def use_sample_and_predict(threshold_val):
        sample_path = "sample_transactions.csv"
        if os.path.exists(sample_path):
            # Call predict function directly with sample path
            return predict_fraud_enhanced(sample_path, threshold_val)
        else:
            return (
                "❌ Error: sample_transactions.csv not found. Please run generate_sample_dataset.py first.",
                pd.DataFrame(),
                None, None, None, None, None, None, None, None
            )
    
    btn = gr.Button("🔍 Detect Fraud", variant="primary", size="lg")
    
    with gr.Row():
        summary_output = gr.Markdown(label="Summary")
    
    with gr.Tabs():
        with gr.Tab("📋 Top Suspicious Transactions"):
            output_table = gr.Dataframe(
                label="Top 20 Most Suspicious Transactions",
                wrap=True
            )
        
        with gr.Tab("📊 Probability Distribution"):
            viz1 = gr.Plot(label="Fraud Probability Distribution")
        
        with gr.Tab("🥧 Risk Level Breakdown"):
            viz2 = gr.Plot(label="Risk Level Distribution")
        
        with gr.Tab("💰 Amount Analysis"):
            viz3 = gr.Plot(label="Amount vs Fraud Probability")
        
        with gr.Tab("📈 Statistics"):
            viz4 = gr.Plot(label="Transaction Summary")
        
        with gr.Tab("⏰ Time Series"):
            viz5 = gr.Plot(label="Fraud Probability Over Time")
        
        with gr.Tab("🔝 Top 20 Suspicious"):
            viz6 = gr.Plot(label="Top 20 Most Suspicious Transactions")
        
        with gr.Tab("🎯 Model Performance"):
            viz7 = gr.Plot(label="Model Performance Metrics")
        
        with gr.Tab("📊 Confusion Matrix"):
            viz8 = gr.Plot(label="Confusion Matrix")
        
        with gr.Tab("📈 ROC Curve"):
            viz9 = gr.Plot(label="ROC Curve")
        
        with gr.Tab("📉 Precision-Recall Curve"):
            viz10 = gr.Plot(label="Precision-Recall Curve")
        
        with gr.Tab("📦 Amount Distribution"):
            viz11 = gr.Plot(label="Amount Distribution Comparison")
        
        with gr.Tab("🎻 Probability Distribution"):
            viz12 = gr.Plot(label="Fraud Probability Distribution Comparison")
        
        with gr.Tab("🔥 Correlation Heatmap"):
            viz13 = gr.Plot(label="Feature Correlation Heatmap")
        
        with gr.Tab("📊 Cumulative Detection"):
            viz14 = gr.Plot(label="Cumulative Fraud Detection Over Time")
        
        with gr.Tab("⚙️ Threshold Analysis"):
            viz15 = gr.Plot(label="Threshold Sensitivity Analysis")
    
    # Connect button to use dataset file (after outputs are defined)
    use_dataset_btn.click(
        fn=use_dataset_and_predict,
        inputs=[threshold],
        outputs=[
            summary_output,
            output_table,
            viz1, viz2, viz3, viz4, viz5, viz6, viz7, viz8,
            viz9, viz10, viz11, viz12, viz13, viz14, viz15
        ]
    )
    
    # Connect Sample button (after outputs are defined)
    sample_btn.click(
        fn=use_sample_and_predict,
        inputs=[threshold],
        outputs=[
            summary_output,
            output_table,
            viz1, viz2, viz3, viz4, viz5, viz6, viz7, viz8,
            viz9, viz10, viz11, viz12, viz13, viz14, viz15
        ]
    )

    btn.click(
        fn=predict_fraud_enhanced,
        inputs=[file_input, threshold],
        outputs=[
            summary_output,
            output_table,
            viz1, viz2, viz3, viz4, viz5, viz6, viz7, viz8,
            viz9, viz10, viz11, viz12, viz13, viz14, viz15
        ]
        # Removed api_name to avoid Gradio API info generation bug
    )
    
    gr.Markdown("""
    ---
    ### 📝 Notes:
    - The model expects numeric features (V1-V28, Amount, optionally Time)
    - If exact feature names are not found, the system will use available numeric columns
    - Processing is limited to 10,000 rows for optimal performance
    - Adjust the threshold to balance between false positives and false negatives
    """)


if __name__ == "__main__":
    import os
    
    print("Starting Gradio app...")
    
    # Check if running on Hugging Face Spaces
    is_spaces = os.getenv("SPACE_ID") is not None
    
    if is_spaces:
        print("Detected Hugging Face Spaces environment")
        # On Spaces, use default settings - Gradio handles everything automatically
        # Add GPU decorator if available (for ZeroGPU or GPU hardware)
        if SPACES_AVAILABLE:
            print("GPU decorator available - using @spaces.GPU for GPU acceleration")
            
            @spaces.GPU
            def launch_with_gpu():
                # Perform GPU operations inside the decorated function to satisfy ZeroGPU
                if CUPY_AVAILABLE:
                    try:
                        # Initialize GPU with a small operation
                        test_gpu = cp.array([1.0, 2.0, 3.0])
                        _ = cp.sum(test_gpu)
                        print("✅ GPU initialized and ready for data processing")
                    except Exception as e:
                        print(f"⚠️ GPU initialization failed, using CPU: {e}")
                
                # Launch Gradio app
                demo.launch()
            
            launch_with_gpu()
        else:
            demo.launch()
    else:
        print("Running locally - server will be available at http://127.0.0.1:7860")
        print("Watch this console for debug output when you upload files!")
        demo.queue(api_open=False)
        try:
            demo.launch(
                server_name="0.0.0.0",  # Use 0.0.0.0 instead of 127.0.0.1 for better compatibility
                server_port=7860,
                share=False,
                show_error=True,
                quiet=False,
                inbrowser=False,
            )
        except ValueError as e:
            # Fallback if 0.0.0.0 doesn't work, try with share enabled
            if "localhost is not accessible" in str(e):
                print("⚠️ Localhost not accessible, trying with share=True...")
                demo.launch(
                    server_name="0.0.0.0",
                    server_port=7860,
                    share=True,  # Enable share as fallback
                    show_error=True,
                    quiet=False,
                    inbrowser=False,
                )
            else:
                raise
