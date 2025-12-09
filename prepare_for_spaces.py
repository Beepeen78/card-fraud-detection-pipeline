#!/usr/bin/env python3
"""
Script to prepare files for Hugging Face Spaces deployment.
This script validates and prepares all necessary files for deployment.
"""

import os
import shutil
import sys
from pathlib import Path

# Required files for Spaces deployment
REQUIRED_FILES = {
    "app.py": "Main Gradio application",
    "requirements.txt": "Python dependencies",
    "README_SPACE.md": "Space README with YAML frontmatter",
}

OPTIONAL_FILES = {
    "fraud_lgbm_calibrated.pkl": "Trained model file (REQUIRED for app to work)",
    "generate_sample_dataset.py": "Sample data generator",
    "sample_transactions.csv": "Sample dataset for testing",
}

def check_file_exists(filepath, required=True):
    """Check if a file exists and return status."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else ("❌ REQUIRED" if required else "⚠️  Optional")
    return exists, status

def validate_app_py():
    """Validate app.py has necessary components."""
    issues = []
    try:
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        checks = {
            "import gradio": "gradio" in content.lower(),
            "gr.Blocks": "gr.Blocks" in content,
            "demo.launch": "demo.launch" in content,
            "SPACE_ID check": "SPACE_ID" in content or "is_spaces" in content.lower(),
        }
        
        for check, passed in checks.items():
            if not passed:
                issues.append(f"  - Missing: {check}")
        
        return len(issues) == 0, issues
    except Exception as e:
        return False, [f"Error reading app.py: {e}"]

def validate_requirements():
    """Validate requirements.txt has essential packages."""
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        essential = ["gradio", "pandas", "numpy", "scikit-learn", "joblib", "plotly"]
        missing = []
        
        for pkg in essential:
            if pkg.lower() not in content.lower():
                missing.append(pkg)
        
        return len(missing) == 0, missing
    except Exception as e:
        return False, [f"Error reading requirements.txt: {e}"]

def validate_readme_space():
    """Validate README_SPACE.md has YAML frontmatter."""
    try:
        with open("README_SPACE.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        checks = {
            "YAML frontmatter": content.startswith("---"),
            "app_file: app.py": "app_file: app.py" in content,
            "sdk: gradio": "sdk: gradio" in content.lower(),
        }
        
        issues = []
        for check, passed in checks.items():
            if not passed:
                issues.append(f"  - Missing: {check}")
        
        return len(issues) == 0, issues
    except Exception as e:
        return False, [f"Error reading README_SPACE.md: {e}"]

def create_gitattributes_for_spaces():
    """Create .gitattributes file for Spaces (tracks .pkl files with Git LFS)."""
    gitattributes_content = """# Git LFS configuration for Hugging Face Spaces
# Track large model files with Git LFS
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pkl.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.hdf5 filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text

# Optional: Track large CSV files (if needed)
# *.csv filter=lfs diff=lfs merge=lfs -text
"""
    
    with open(".gitattributes_spaces", "w", encoding="utf-8") as f:
        f.write(gitattributes_content)
    
    print("✅ Created .gitattributes_spaces (copy this to your Space as .gitattributes)")

def main():
    """Main validation and preparation function."""
    print("=" * 70)
    print("🚀 Hugging Face Spaces Deployment Preparation")
    print("=" * 70)
    print()
    
    # Check current directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found in current directory.")
        print("   Please run this script from the project root directory.")
        sys.exit(1)
    
    print("📋 Checking Required Files...")
    print("-" * 70)
    
    all_good = True
    for filename, description in REQUIRED_FILES.items():
        exists, status = check_file_exists(filename, required=True)
        print(f"{status} {filename:30s} - {description}")
        if not exists:
            all_good = False
    
    print()
    print("📦 Checking Optional Files...")
    print("-" * 70)
    
    for filename, description in OPTIONAL_FILES.items():
        exists, status = check_file_exists(filename, required=False)
        print(f"{status} {filename:30s} - {description}")
        if filename == "fraud_lgbm_calibrated.pkl" and not exists:
            print("   ⚠️  WARNING: Model file is missing! The app won't work without it.")
            all_good = False
    
    print()
    print("🔍 Validating File Contents...")
    print("-" * 70)
    
    # Validate app.py
    app_valid, app_issues = validate_app_py()
    if app_valid:
        print("✅ app.py - Valid structure")
    else:
        print("❌ app.py - Issues found:")
        for issue in app_issues:
            print(issue)
        all_good = False
    
    # Validate requirements.txt
    req_valid, req_missing = validate_requirements()
    if req_valid:
        print("✅ requirements.txt - Contains essential packages")
    else:
        print(f"❌ requirements.txt - Missing packages: {', '.join(req_missing)}")
        all_good = False
    
    # Validate README_SPACE.md
    readme_valid, readme_issues = validate_readme_space()
    if readme_valid:
        print("✅ README_SPACE.md - Valid YAML frontmatter")
    else:
        print("❌ README_SPACE.md - Issues found:")
        for issue in readme_issues:
            print(issue)
        all_good = False
    
    print()
    print("📝 Creating Git LFS Configuration...")
    print("-" * 70)
    create_gitattributes_for_spaces()
    
    print()
    print("=" * 70)
    if all_good:
        print("✅ All checks passed! Ready for deployment.")
        print()
        print("📋 Next Steps:")
        print("1. Create a new Space on Hugging Face (use Blank template)")
        print("2. Clone your Space repository")
        print("3. Copy these files to your Space:")
        print("   - app.py")
        print("   - requirements.txt")
        print("   - README_SPACE.md (rename to README.md)")
        print("   - fraud_lgbm_calibrated.pkl (use Git LFS)")
        print("   - .gitattributes_spaces (rename to .gitattributes)")
        print("4. Commit and push to deploy")
        print()
        print("📖 See SPACES_DEPLOYMENT.md for detailed instructions")
    else:
        print("⚠️  Some issues found. Please fix them before deploying.")
        print()
        print("Common fixes:")
        print("- Ensure fraud_lgbm_calibrated.pkl exists (train model if needed)")
        print("- Check that all required files are present")
        print("- Verify README_SPACE.md has correct YAML frontmatter")
    print("=" * 70)

if __name__ == "__main__":
    main()
