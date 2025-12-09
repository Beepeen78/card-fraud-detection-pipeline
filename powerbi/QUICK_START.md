# Power BI Quick Start Guide

## 5-Minute Setup

### Step 1: Generate Data
```bash
python app.py
# Click "📊 Use Sample Dataset" button
# Or upload your own CSV
```

### Step 2: Open Power BI Desktop
1. Launch Power BI Desktop
2. Click **Get Data** → **Text/CSV**
3. Select `powerbi/out/transactions_scored.csv`
4. Click **Load**

### Step 3: Add Basic Visuals

**Create a simple dashboard:**

1. **KPI Cards:**
   - Drag `fraud_prediction` to a Card visual
   - Change aggregation to "Sum"
   - Title: "Fraud Detected"

2. **Risk Breakdown:**
   - Drag `risk_level` to a Pie Chart
   - Drag `transaction_id` to Values (Count)

3. **Trend:**
   - Drag `transaction_date` to X-axis
   - Drag `fraud_prediction` to Y-axis (Sum)
   - Visual type: Line Chart

### Step 4: Add DAX Measures (Optional)

1. Right-click Transactions table → **New Measure**
2. Copy from `dax_measures.txt`:
   ```
   Fraud Rate = DIVIDE(SUM(Transactions[fraud_prediction]), COUNTROWS(Transactions), 0)
   ```
3. Use this measure in your visuals

## That's It!

You now have a basic fraud detection dashboard. For more advanced features, see `setup_instructions.md`.

## Next Steps

- Add more visuals (see `dashboard_layout.md`)
- Import DAX measures from `dax_measures.txt`
- Apply theme from `powerbi_theme.json`
- Set up alerts (see `alerts_and_rules.md`)
