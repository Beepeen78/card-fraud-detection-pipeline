# 🚀 Hugging Face Spaces Deployment Guide

This guide will help you deploy the Credit Card Fraud Detection system to Hugging Face Spaces.

## 📋 Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Model File**: You need `fraud_lgbm_calibrated.pkl` (trained model)
3. **Git**: For pushing to the Space repository

## 🎯 Step-by-Step Deployment

### Step 1: Create a New Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `credit-card-fraud-detection` (or your preferred name)
   - **SDK**: Select **Gradio**
   - **Template**: Select **Blank** (not Trackio or other templates)
   - **Hardware**: Choose based on your needs:
     - **CPU Basic**: Free tier, good for testing
     - **ZeroGPU**: If you need GPU (not required for inference)
   - **Visibility**: Public or Private

### Step 2: Clone Your Space Repository

After creating the Space, Hugging Face will provide you with a Git URL. Clone it:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
```

### Step 3: Copy Required Files

Copy these files from your project to the Space repository:

**Essential Files:**
- ✅ `app.py` - Main Gradio application
- ✅ `requirements.txt` - Python dependencies
- ✅ `README.md` or `README_SPACE.md` - Space description (with YAML frontmatter)
- ✅ `fraud_lgbm_calibrated.pkl` - Trained model file

**Optional Files (for reference):**
- `generate_sample_dataset.py` - Sample data generator
- `powerbi_export.py` - Power BI integration (optional)

### Step 4: Handle the Model File

The model file (`fraud_lgbm_calibrated.pkl`) is likely too large for regular Git. You have three options:

#### Option A: Use Git Xet (Recommended for Hugging Face Spaces)

Git Xet is Hugging Face's recommended solution for handling large files in Spaces:

```bash
# Install git-xet (macOS)
brew install git-xet

# Or install via pip
pip install git-xet

# Initialize git-xet
git xet install

# Clone your Space (git-xet handles large files automatically)
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

**Note**: Git Xet is automatically configured in Hugging Face Spaces, so large files are handled seamlessly.

#### Option B: Use Git LFS (Alternative)

```bash
# Install Git LFS if not already installed
git lfs install

# Track .pkl files
git lfs track "*.pkl"

# Add the model file
git add .gitattributes
git add fraud_lgbm_calibrated.pkl
```

#### Option C: Upload via Hugging Face Hub

1. Install `huggingface_hub`:
   ```bash
   pip install huggingface_hub
   ```

2. Upload the model:
   ```python
   from huggingface_hub import HfApi
   
   api = HfApi()
   api.upload_file(
       path_or_fileobj="fraud_lgbm_calibrated.pkl",
       path_in_repo="fraud_lgbm_calibrated.pkl",
       repo_id="YOUR_USERNAME/YOUR_SPACE_NAME",
       repo_type="space"
   )
   ```

### Step 5: Update README.md

Use the `README_SPACE.md` file as your Space README, or copy its YAML frontmatter to your main README.md:

```yaml
---
title: Credit Card Fraud Detection
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---
```

### Step 6: Commit and Push

```bash
git add .
git commit -m "Initial deployment of fraud detection system"
git push
```

### Step 7: Wait for Build

Hugging Face Spaces will automatically:
1. Install dependencies from `requirements.txt`
2. Run `app.py`
3. Make your Space available at `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

The build process usually takes 2-5 minutes. You can monitor it in the Space's "Logs" tab.

## 🔧 Configuration Files Summary

### Required Files Structure

```
your-space/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── README.md                       # Space description (with YAML frontmatter)
├── fraud_lgbm_calibrated.pkl       # Trained model (via Git LFS)
└── .gitattributes                  # Git LFS configuration (if using LFS)
```

### app.py Changes

The `app.py` has been updated to automatically detect Hugging Face Spaces environment and use appropriate settings. No manual configuration needed!

### requirements.txt

The existing `requirements.txt` is Spaces-compatible. Google Cloud packages are optional and won't cause issues if not configured.

## 🐛 Troubleshooting

### Build Fails

1. **Check Logs**: Go to your Space → "Logs" tab
2. **Common Issues**:
   - Missing model file: Ensure `fraud_lgbm_calibrated.pkl` is uploaded
   - Dependency conflicts: Check `requirements.txt` versions
   - Import errors: Verify all imports in `app.py`

### Model File Not Found

- Ensure the model file is in the root directory
- If using Git LFS, verify it's properly tracked
- Check file name matches `MODEL_PATH` in `app.py` (currently `fraud_lgbm_calibrated.pkl`)

### App Crashes on Upload

- Check Space logs for error messages
- Verify CSV format matches expected columns
- Test with sample data first

### Out of Memory

- Reduce `nrows=10000` limit in `app.py` if processing large files
- Upgrade to a higher-tier hardware option

## 📝 Optional Enhancements

### Add Sample Data

You can include a sample CSV file in your Space:

```bash
# Generate sample data
python generate_sample_dataset.py

# Add to Space (if small enough)
git add sample_transactions.csv
```

### Environment Variables

If you want to add BigQuery integration, set environment variables in Space settings:
- Go to Space → Settings → Variables
- Add: `BQ_PROJECT`, `BQ_DATASET`, etc.

### Custom Domain

Spaces support custom domains. Check Space settings for configuration.

## ✅ Verification Checklist

Before deploying, ensure:

- [ ] `app.py` is in the repository root
- [ ] `requirements.txt` includes all dependencies
- [ ] `README.md` has YAML frontmatter with correct `app_file: app.py`
- [ ] Model file (`fraud_lgbm_calibrated.pkl`) is uploaded (via Git LFS or Hub)
- [ ] All imports in `app.py` are available in `requirements.txt`
- [ ] No hardcoded local paths (use relative paths)
- [ ] App works locally before deploying

## 🎉 After Deployment

Once deployed, your Space will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

You can:
- Share the link with others
- Embed it in websites
- Use the API endpoint for programmatic access
- Monitor usage in Space analytics

## 📚 Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [Git LFS Documentation](https://git-lfs.github.com/)

---

**Need Help?** Check the Space logs or open an issue in your repository.
