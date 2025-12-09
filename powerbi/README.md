# Power BI Fraud Detection Dashboard

This directory contains everything you need to build a comprehensive Power BI dashboard for fraud detection analytics.

## Files

- **QUICK_START.md** - Get started in 5 minutes ⚡
- **setup_instructions.md** - Detailed step-by-step setup guide
- **dashboard_layout.md** - Complete dashboard design with 5 pages
- **fraud_detection_queries.m** - Power Query (M) scripts for data transformation
- **dax_measures.txt** - 30+ pre-built DAX measures for analytics
- **powerbi_theme.json** - Custom color theme (import in Power BI)
- **alerts_and_rules.md** - Alert configuration and business rules
- **prepare_powerbi_data.py** - Enhanced data preparation script
- **PROJECT_SUMMARY.md** - Complete project overview

## Quick Start

**For fastest setup, see `QUICK_START.md`**

1. **Run the app** (data exports automatically):
   ```bash
   python app.py
   # Click "📊 Use Sample Dataset" or upload your CSV
   # Data automatically exports to powerbi/out/
   ```

2. **Open Power BI Desktop**

3. **Load data:**
   - Get Data → Text/CSV
   - Select `powerbi/out/transactions_scored.csv`
   - Click Load

4. **Add basic visuals:**
   - See `QUICK_START.md` for 3 simple visuals

5. **For full dashboard:**
   - Follow `setup_instructions.md` for complete setup
   - Use `dashboard_layout.md` for design
   - Import DAX measures from `dax_measures.txt`

## Data Sources

The dashboard expects CSV files in `powerbi/out/`:
- `transactions_scored.csv` - Individual transaction scores
- `metrics_daily.csv` - Daily aggregated metrics

## Features

- Real-time fraud detection monitoring
- Risk level breakdowns
- Transaction trend analysis
- Model performance metrics
- Geographic fraud patterns (if location data available)
- Time-based fraud patterns
- Amount distribution analysis

## Important: What I Can and Cannot Do

**I CAN create:**
- ✅ All Power Query scripts, DAX measures, and documentation
- ✅ Data export scripts (automatic)
- ✅ Power BI Project (.pbip) structure
- ✅ Complete setup instructions

**I CANNOT:**
- ❌ Open Power BI Desktop directly
- ❌ Create ready-to-use .pbix files from scratch
- ❌ Access Power BI Service

**See `ACCESS_POWERBI.md` for details on what I can help with and what you need to do.**
