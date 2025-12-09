# Power BI Dashboard Setup Instructions

## Step 1: Export Data from the App

First, run the fraud detection app and process your transactions:

```bash
python app.py
```

1. Upload your transaction CSV or use the sample dataset
2. Process the transactions
3. The app will automatically export data to `powerbi/out/`:
   - `transactions_scored.csv`
   - `metrics_daily.csv`

## Step 2: Open Power BI Desktop

1. Launch Power BI Desktop
2. Create a new report

## Step 3: Load Data

### Option A: Load from CSV Files

1. Click **Get Data** → **Text/CSV**
2. Navigate to `powerbi/out/transactions_scored.csv`
3. Click **Load** or **Transform Data**
4. Repeat for `metrics_daily.csv`

### Option B: Use Power Query Script

1. Click **Get Data** → **Blank Query**
2. Click **Advanced Editor**
3. Copy contents from `fraud_detection_queries.m`
4. Paste into the editor
5. Click **Done**
6. Rename the query to "Transactions"
7. Repeat for Daily Metrics (create separate query)

## Step 4: Transform Data

### Add Calculated Columns

In Power Query Editor, add these columns to the Transactions table:

1. **TransactionDate:**
   ```
   = DateTime.From(#datetime(1970, 1, 1, 0, 0, 0) + #duration(0, 0, 0, [unix_time]))
   ```

2. **RiskLevel:**
   ```
   = if [fraud_probability] < 0.1 then "Low"
     else if [fraud_probability] < 0.5 then "Medium"
     else if [fraud_probability] < 0.9 then "High"
     else "Critical"
   ```

3. **Hour:**
   ```
   = Time.Hour([TransactionDate])
   ```

4. **DayOfWeek:**
   ```
   = Date.DayOfWeek([TransactionDate], Day.Monday)
   ```

5. **Month:**
   ```
   = Date.Month([TransactionDate])
   ```

### Set Data Types

- `fraud_probability`: Decimal Number
- `fraud_prediction`: Whole Number
- `amt`: Decimal Number
- `TransactionDate`: Date/Time
- `RiskLevel`: Text

Click **Close & Apply**

## Step 5: Create Relationships

1. Go to **Model** view
2. If you have Daily Metrics table, create relationship:
   - Transactions[TransactionDate] → DailyMetrics[date]
   - Relationship type: Many-to-One

## Step 6: Add DAX Measures

1. Go to **Data** view
2. Select the Transactions table
3. Click **New Measure**
4. Copy measures from `dax_measures.txt` one by one
5. Name each measure exactly as shown

Key measures to start with:
- Total Transactions
- Total Fraud Detected
- Fraud Rate
- Average Fraud Probability

## Step 7: Build Visualizations

Follow the layout guide in `dashboard_layout.md`:

1. **Create Page 1: Executive Summary**
   - Add KPI cards
   - Add line chart for trends
   - Add donut chart for risk breakdown

2. **Create Page 2: Transaction Analysis**
   - Add scatter plots
   - Add tables
   - Add histograms

3. **Create Page 3: Model Performance** (if you have ground truth)
   - Add confusion matrix
   - Add performance metrics

4. **Create Page 4: Time Series**
   - Add time-based charts
   - Add heatmaps

5. **Create Page 5: Geographic** (if location data available)
   - Add maps or location-based visuals

## Step 8: Add Filters

1. Insert → Slicer
2. Add slicers for:
   - Date Range (use TransactionDate)
   - Risk Level
   - Amount Range
3. Format slicers to match your theme

## Step 9: Formatting

1. Apply color scheme:
   - Low Risk: Green
   - Medium Risk: Yellow
   - High Risk: Orange
   - Critical Risk: Red

2. Format numbers:
   - Percentages: Show as %
   - Currency: Format amounts as currency
   - Decimals: Set appropriate decimal places

3. Add titles and labels to all visuals

## Step 10: Set Up Auto-Refresh (Optional)

If you want the dashboard to refresh automatically:

1. File → Options and Settings → Data Source Settings
2. Configure refresh schedule
3. Or use Power BI Gateway for scheduled refreshes

## Troubleshooting

**Data not loading:**
- Check file paths in Power Query
- Ensure CSV files exist in `powerbi/out/`
- Check file encoding (should be UTF-8)

**Measures showing errors:**
- Check table and column names match exactly
- Ensure relationships are set up correctly
- Verify data types are correct

**Visuals not showing data:**
- Check filters aren't excluding all data
- Verify measures are calculating correctly
- Check date ranges in slicers

## Next Steps

- Publish to Power BI Service for sharing
- Set up scheduled data refresh
- Create alerts for high fraud rates
- Add drill-through pages for detailed analysis
- Create mobile-optimized views
