#!/usr/bin/env python
"""
Helper script to prepare everything for Power BI import.
This creates a structured template that can be easily imported into Power BI Desktop.
"""

import json
import os
from pathlib import Path

def create_powerbi_template():
    """
    Creates a template structure and instructions for Power BI.
    Since .pbix files are binary and require Power BI Desktop,
    this script prepares all the components needed.
    """
    
    template_dir = Path("powerbi/template")
    template_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a template configuration file
    template_config = {
        "name": "Fraud Detection Dashboard",
        "version": "1.0",
        "description": "Complete fraud detection analytics dashboard",
        "data_sources": [
            {
                "name": "Transactions",
                "type": "CSV",
                "path": "powerbi/out/transactions_scored.csv",
                "query_file": "fraud_detection_queries.m"
            },
            {
                "name": "Daily Metrics",
                "type": "CSV",
                "path": "powerbi/out/metrics_daily.csv",
                "query_file": "fraud_detection_queries.m"
            }
        ],
        "pages": [
            {
                "name": "Executive Summary",
                "description": "High-level KPIs and trends"
            },
            {
                "name": "Transaction Analysis",
                "description": "Detailed transaction views"
            },
            {
                "name": "Model Performance",
                "description": "Precision, recall, confusion matrix"
            },
            {
                "name": "Time Series Analysis",
                "description": "Hourly, daily, monthly patterns"
            },
            {
                "name": "Geographic Analysis",
                "description": "Location-based fraud patterns"
            }
        ],
        "measures_file": "dax_measures.txt",
        "theme_file": "powerbi_theme.json",
        "setup_guide": "setup_instructions.md"
    }
    
    # Save template config
    config_path = template_dir / "template_config.json"
    with open(config_path, 'w') as f:
        json.dump(template_config, f, indent=2)
    
    # Create import script for Power BI
    import_script = """
# Power BI Import Script
# This script helps you import all components into Power BI Desktop

# STEP 1: Open Power BI Desktop
# STEP 2: Get Data → Text/CSV
# STEP 3: Select: powerbi/out/transactions_scored.csv
# STEP 4: Click "Transform Data" (not Load)
# STEP 5: In Power Query Editor → Advanced Editor
# STEP 6: Copy and paste contents from: fraud_detection_queries.m
# STEP 7: Click "Done"
# STEP 8: Rename query to "Transactions"
# STEP 9: Repeat for metrics_daily.csv (rename to "DailyMetrics")
# STEP 10: Click "Close & Apply"

# STEP 11: Go to Data view
# STEP 12: Select Transactions table
# STEP 13: Click "New Measure"
# STEP 14: Copy measures from dax_measures.txt one by one

# STEP 15: Go to Report view
# STEP 16: Follow dashboard_layout.md to create visuals

# STEP 17: View → Themes → Browse for themes
# STEP 18: Select powerbi_theme.json

# STEP 19: File → Save As → fraud_detection_dashboard.pbix
"""
    
    import_script_path = template_dir / "IMPORT_INSTRUCTIONS.txt"
    with open(import_script_path, 'w') as f:
        f.write(import_script)
    
    # Create a checklist
    checklist = """
# Power BI Setup Checklist

## Data Preparation
- [ ] Run app.py and process transactions
- [ ] Verify powerbi/out/transactions_scored.csv exists
- [ ] Verify powerbi/out/metrics_daily.csv exists

## Power BI Desktop Setup
- [ ] Open Power BI Desktop
- [ ] Load transactions_scored.csv
- [ ] Load metrics_daily.csv
- [ ] Apply Power Query transformations (fraud_detection_queries.m)
- [ ] Create relationships between tables
- [ ] Set correct data types

## DAX Measures
- [ ] Import Total Transactions measure
- [ ] Import Total Fraud Detected measure
- [ ] Import Fraud Rate measure
- [ ] Import all risk level measures
- [ ] Import time-based measures
- [ ] Import performance measures (if ground truth available)

## Visualizations
- [ ] Create Executive Summary page
- [ ] Create Transaction Analysis page
- [ ] Create Model Performance page (if applicable)
- [ ] Create Time Series page
- [ ] Create Geographic page (if data available)
- [ ] Add slicers/filters
- [ ] Format visuals consistently

## Styling
- [ ] Import powerbi_theme.json
- [ ] Apply color scheme
- [ ] Format numbers (currency, percentages)
- [ ] Add titles and labels

## Final Steps
- [ ] Test all visuals
- [ ] Verify calculations
- [ ] Save as .pbix file
- [ ] (Optional) Publish to Power BI Service
"""
    
    checklist_path = template_dir / "SETUP_CHECKLIST.md"
    with open(checklist_path, 'w') as f:
        f.write(checklist)
    
    print("✅ Power BI template structure created!")
    print(f"   Location: {template_dir}")
    print("\n📋 Files created:")
    print(f"   - template_config.json - Template configuration")
    print(f"   - IMPORT_INSTRUCTIONS.txt - Step-by-step import guide")
    print(f"   - SETUP_CHECKLIST.md - Setup checklist")
    print("\n💡 Next steps:")
    print("   1. Follow IMPORT_INSTRUCTIONS.txt in Power BI Desktop")
    print("   2. Use SETUP_CHECKLIST.md to track progress")
    print("   3. All queries and measures are in the parent directory")

if __name__ == "__main__":
    create_powerbi_template()
