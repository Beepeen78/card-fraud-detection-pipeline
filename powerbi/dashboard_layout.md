# Power BI Dashboard Layout Guide

## Dashboard Structure

### Page 1: Executive Summary

**Top Row:**
- **Card 1:** Total Transactions (KPI)
- **Card 2:** Fraud Detected (KPI with red indicator)
- **Card 3:** Fraud Rate % (KPI)
- **Card 4:** Total Amount at Risk (KPI with currency format)

**Middle Row:**
- **Visual 1:** Fraud Rate Trend (Line Chart)
  - X-axis: TransactionDate
  - Y-axis: Fraud Rate
  - Show last 30 days
  
- **Visual 2:** Risk Level Breakdown (Donut Chart)
  - Values: Low, Medium, High, Critical counts
  - Colors: Green, Yellow, Orange, Red

**Bottom Row:**
- **Visual 3:** Fraud by Hour (Bar Chart)
  - X-axis: Hour (0-23)
  - Y-axis: Total Fraud Detected
  - Color: Red gradient
  
- **Visual 4:** Fraud by Day of Week (Bar Chart)
  - X-axis: DayOfWeek (Mon-Sun)
  - Y-axis: Total Fraud Detected

### Page 2: Transaction Analysis

**Top Row:**
- **Card 1:** Average Fraud Probability
- **Card 2:** Average Transaction Amount
- **Card 3:** High Value Fraud Count (>$500)
- **Card 4:** Average Fraud Amount

**Middle Row:**
- **Visual 1:** Fraud Probability Distribution (Histogram)
  - X-axis: Fraud Probability (bins)
  - Y-axis: Count of Transactions
  - Color: Risk Level
  
- **Visual 2:** Amount vs Fraud Probability (Scatter Plot)
  - X-axis: Amount
  - Y-axis: Fraud Probability
  - Color: Risk Level
  - Size: Amount

**Bottom Row:**
- **Visual 3:** Top 20 Most Suspicious Transactions (Table)
  - Columns: TransactionDate, Amount, Fraud Probability, Risk Level
  - Sort by: Fraud Probability (Descending)
  
- **Visual 4:** Fraud by Category (if available) (Bar Chart)
  - X-axis: Category
  - Y-axis: Total Fraud Detected

### Page 3: Model Performance (if ground truth available)

**Top Row:**
- **Card 1:** Precision
- **Card 2:** Recall
- **Card 3:** F1 Score
- **Card 4:** Accuracy

**Middle Row:**
- **Visual 1:** Confusion Matrix (Matrix Visual)
  - Rows: Actual (Fraud, Normal)
  - Columns: Predicted (Fraud, Normal)
  - Values: Count
  - Color scale: Red to Green
  
- **Visual 2:** Precision/Recall Over Time (Line Chart)
  - X-axis: TransactionDate
  - Y-axis: Precision, Recall (dual axis)

**Bottom Row:**
- **Visual 3:** Performance Metrics Comparison (Bar Chart)
  - Categories: Precision, Recall, F1, Accuracy
  - Values: Measure values
  
- **Visual 4:** Error Analysis (Table)
  - False Positives and False Negatives
  - Show transaction details

### Page 4: Time Series Analysis

**Top Row:**
- **Card 1:** Fraud Today
- **Card 2:** Fraud This Week
- **Card 3:** Fraud This Month
- **Card 4:** Fraud Rate Trend (change from yesterday)

**Middle Row:**
- **Visual 1:** Fraud Detection Over Time (Line Chart)
  - X-axis: TransactionDate
  - Y-axis: Total Fraud Detected (cumulative)
  - Secondary Y-axis: Fraud Rate
  
- **Visual 2:** Daily Fraud Pattern (Area Chart)
  - X-axis: TransactionDate
  - Y-axis: Total Fraud Detected
  - Color: Risk Level

**Bottom Row:**
- **Visual 3:** Hourly Fraud Pattern (Heatmap)
  - Rows: Day of Week
  - Columns: Hour
  - Values: Total Fraud Detected
  - Color scale: Light to Dark Red
  
- **Visual 4:** Monthly Trend (Line Chart)
  - X-axis: Month
  - Y-axis: Total Fraud Detected
  - Show trend line

### Page 5: Geographic Analysis (if location data available)

**Top Row:**
- **Card 1:** Cities with Most Fraud
- **Card 2:** Average Distance for Fraud
- **Card 3:** High Risk Locations

**Middle Row:**
- **Visual 1:** Fraud by City Population (Scatter Plot)
  - X-axis: City Population
  - Y-axis: Fraud Count
  - Size: Total Amount
  
- **Visual 2:** Fraud by Distance (Bar Chart)
  - X-axis: Distance buckets (<1km, 1-10km, etc.)
  - Y-axis: Total Fraud Detected

**Bottom Row:**
- **Visual 3:** Top Fraud Locations (Table)
  - City, Fraud Count, Average Amount
  
- **Visual 4:** Distance Distribution (Histogram)
  - X-axis: Distance
  - Y-axis: Count
  - Color: Fraud vs Normal

## Color Scheme

- **Low Risk:** Green (#00B050)
- **Medium Risk:** Yellow (#FFC000)
- **High Risk:** Orange (#FF6600)
- **Critical Risk:** Red (#C00000)
- **Background:** Light Gray (#F2F2F2)
- **Cards:** White with subtle shadow

## Filters (Apply to All Pages)

- Date Range Slicer
- Risk Level Slicer
- Amount Range Slicer
- Category Slicer (if available)

## Tips

1. Use conditional formatting for KPI cards (red if fraud rate > threshold)
2. Add tooltips with additional context
3. Use bookmarks for different views
4. Add drill-through pages for transaction details
5. Use themes for consistent styling
6. Enable cross-filtering between visuals
