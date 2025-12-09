# 🚀 Deploy Now - Quick Guide

## Prerequisites Check

Before deploying, ensure you have:

1. ✅ **Hugging Face Account** - Sign up at [huggingface.co](https://huggingface.co)
2. ✅ **Space Created** - Create at [huggingface.co/spaces](https://huggingface.co/spaces)
   - SDK: **Gradio**
   - Template: **Blank** (not Trackio!)
3. ⚠️ **Model File** - `fraud_lgbm_calibrated.pkl` (if missing, train it first)

## Quick Deployment (PowerShell)

### Option 1: Automated Script (Recommended)

```powershell
# Run the deployment script
.\deploy_to_spaces.ps1 -SpaceName "Card-Fraud-detection" -Username "Beepeen24"
```

The script will:
- ✅ Check all required files
- ✅ Clone your Space repository
- ✅ Copy all necessary files
- ✅ Commit and push changes

### Option 2: Manual Steps

#### Step 1: Install Git Xet (Recommended)
```powershell
# macOS
brew install git-xet

# Or via pip
pip install git-xet
git xet install
```

#### Step 2: Clone Your Space
```powershell
git clone https://huggingface.co/spaces/Beepeen24/Card-Fraud-detection
cd Card-Fraud-detection
```

#### Step 3: Copy Files
```powershell
# From your project directory
Copy-Item ..\card-fraud-detection-pipeline\app.py .
Copy-Item ..\card-fraud-detection-pipeline\requirements.txt .
Copy-Item ..\card-fraud-detection-pipeline\README_SPACE.md README.md
Copy-Item ..\card-fraud-detection-pipeline\fraud_lgbm_calibrated.pkl .  # If exists
```

#### Step 4: Commit and Push
```powershell
git add .
git commit -m "Deploy fraud detection system"
git push
```

## If Model File is Missing

If `fraud_lgbm_calibrated.pkl` doesn't exist:

```powershell
# Train the model first
python train_improved_model.py

# Then deploy
.\deploy_to_spaces.ps1 -SpaceName "Card-Fraud-detection" -Username "Beepeen24"
```

## Verify Deployment

1. Visit: https://huggingface.co/spaces/Beepeen24/Card-Fraud-detection
2. Wait 2-5 minutes for build
3. Check "Logs" tab for any errors
4. Test by uploading a CSV file

## Troubleshooting

### Authentication Issues
```powershell
# Login to Hugging Face
huggingface-cli login
```

### Git Xet Not Working
```powershell
# Fall back to Git LFS
git lfs install
git lfs track "*.pkl"
```

### Build Fails
- Check Space logs
- Verify `requirements.txt` is correct
- Ensure model file exists and is accessible

---

**Ready?** Run: `.\deploy_to_spaces.ps1 -SpaceName "Card-Fraud-detection" -Username "Beepeen24"`
