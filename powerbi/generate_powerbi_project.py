#!/usr/bin/env python
"""
Generate a Power BI Project (.pbip) file programmatically.
This creates a text-based Power BI project that can be opened in Power BI Desktop.
"""

import json
import os
from pathlib import Path
from datetime import datetime

def create_pbip_structure():
    """
    Creates a Power BI Project (.pbip) structure.
    Power BI Projects are folders with text files that Power BI Desktop can open.
    """
    
    pbip_dir = Path("powerbi/fraud_detection_dashboard.pbip")
    pbip_dir.mkdir(parents=True, exist_ok=True)
    
    # Create .pbip metadata file
    pbip_metadata = {
        "version": "1.0",
        "name": "fraud_detection_dashboard",
        "type": "Report"
    }
    
    with open(pbip_dir / ".pbip", 'w') as f:
        json.dump(pbip_metadata, f, indent=2)
    
    # Create report folder structure
    report_dir = pbip_dir / "Report"
    report_dir.mkdir(exist_ok=True)
    
    semantic_model_dir = pbip_dir / "SemanticModel"
    semantic_model_dir.mkdir(exist_ok=True)
    
    # Create basic report layout
    report_layout = {
        "version": "1.0",
        "sections": [
            {
                "name": "Executive Summary",
                "displayName": "Executive Summary",
                "visualContainers": []
            },
            {
                "name": "Transaction Analysis",
                "displayName": "Transaction Analysis",
                "visualContainers": []
            },
            {
                "name": "Time Series",
                "displayName": "Time Series Analysis",
                "visualContainers": []
            }
        ],
        "config": {
            "name": "Fraud Detection Dashboard",
            "settings": {}
        }
    }
    
    layout_path = report_dir / "Layout"
    with open(layout_path, 'w', encoding='utf-16-le') as f:
        json.dump(report_layout, f, ensure_ascii=False, indent=2)
    
    # Create semantic model configuration
    semantic_model = {
        "version": "1.0",
        "name": "FraudDetectionModel",
        "tables": [
            {
                "name": "Transactions",
                "source": {
                    "type": "structured",
                    "protocol": "file",
                    "path": "powerbi/out/transactions_scored.csv"
                }
            },
            {
                "name": "DailyMetrics",
                "source": {
                    "type": "structured",
                    "protocol": "file",
                    "path": "powerbi/out/metrics_daily.csv"
                }
            }
        ]
    }
    
    model_path = semantic_model_dir / "Model.bim"
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(semantic_model, f, indent=2)
    
    # Create README for the project
    readme = f"""# Power BI Project: Fraud Detection Dashboard

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## How to Use

1. **Ensure data files exist:**
   - `powerbi/out/transactions_scored.csv`
   - `powerbi/out/metrics_daily.csv`

2. **Open in Power BI Desktop:**
   - File → Open → Browse
   - Select this folder: `fraud_detection_dashboard.pbip`
   - Power BI Desktop will recognize it as a project

3. **If Power BI Desktop doesn't recognize it:**
   - Enable Power BI Projects feature:
     - File → Options → Preview features
     - Enable "Power BI Project (.pbip) save option"
   - Or manually create a new report and import the data

## Next Steps

- Import DAX measures from `../dax_measures.txt`
- Follow `../dashboard_layout.md` for visual design
- Apply theme from `../powerbi_theme.json`

## Note

This is a basic project structure. You'll need to:
- Connect to the data sources in Power BI Desktop
- Apply Power Query transformations from `../fraud_detection_queries.m`
- Create relationships between tables
- Build visualizations

See `../setup_instructions.md` for detailed steps.
"""
    
    with open(pbip_dir / "README.txt", 'w') as f:
        f.write(readme)
    
    print(f"✅ Power BI Project created at: {pbip_dir}")
    print("\n📋 Project Structure:")
    print(f"   - {pbip_dir}/.pbip (metadata)")
    print(f"   - {pbip_dir}/Report/Layout (report structure)")
    print(f"   - {pbip_dir}/SemanticModel/Model.bim (data model)")
    print(f"   - {pbip_dir}/README.txt (instructions)")
    print("\n💡 To open:")
    print(f"   1. Open Power BI Desktop")
    print(f"   2. File → Open → Browse")
    print(f"   3. Select folder: {pbip_dir}")
    print("\n⚠️  Note: You may need to enable Power BI Projects feature first:")
    print("   File → Options → Preview features → Enable 'Power BI Project (.pbip) save option'")

def create_pbix_alternative():
    """
    Alternative: Create instructions for using pypbireport library
    to programmatically create a .pbix file.
    """
    
    script_content = '''#!/usr/bin/env python
"""
Alternative approach: Use pypbireport to create .pbix programmatically.
Install: pip install pypbireport
"""

try:
    import pypbireport as ppr
    print("✅ pypbireport is available")
    print("\\nYou can use this to programmatically create .pbix files.")
    print("\\nExample usage:")
    print("""
    # Create a new report
    report = ppr.PBIReport()
    
    # Add pages
    report.add_page('Executive Summary')
    report.add_page('Transaction Analysis')
    
    # Add visuals (requires existing data model)
    # ... add your visuals here ...
    
    # Save as .pbix
    report.save('fraud_detection_dashboard.pbix')
    """)
except ImportError:
    print("⚠️  pypbireport not installed")
    print("\\nTo install:")
    print("  pip install pypbireport")
    print("\\nNote: This library requires an existing .pbix file as a template")
    print("      or you need to create the report structure manually.")
'''
    
    script_path = Path("powerbi/create_pbix_with_pypbireport.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\n✅ Alternative script created: {script_path}")

if __name__ == "__main__":
    print("Creating Power BI Project structure...")
    create_pbip_structure()
    create_pbix_alternative()
    print("\n✅ Done! See powerbi/ directory for all files.")
