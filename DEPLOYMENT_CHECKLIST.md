# ✅ Hugging Face Spaces Deployment Checklist

Use this checklist to ensure your deployment is successful.

## Pre-Deployment Checklist

### 📁 Required Files
- [ ] `app.py` - Main Gradio application (✅ Already configured for Spaces)
- [ ] `requirements.txt` - Python dependencies (✅ Already compatible)
- [ ] `README_SPACE.md` - Space README with YAML frontmatter (✅ Ready)
- [ ] `fraud_lgbm_calibrated.pkl` - Trained model file
- [ ] `.gitattributes` - Git LFS configuration (for .pkl files)

### 🔍 Validation
- [ ] Run `python prepare_for_spaces.py` to validate files
- [ ] Test `app.py` locally: `python app.py`
- [ ] Verify model file exists and is accessible
- [ ] Check that all imports in `app.py` are in `requirements.txt`

### 📝 File Preparation
- [ ] Copy `README_SPACE.md` → `README.md` (for the Space)
- [ ] Copy `.gitattributes_spaces` → `.gitattributes` (if created)
- [ ] Ensure model file is ready for upload (use Git LFS)

## Deployment Steps

### 1. Create Space
- [ ] Go to [huggingface.co/spaces](https://huggingface.co/spaces)
- [ ] Click "Create new Space"
- [ ] **SDK**: Select **Gradio**
- [ ] **Template**: Select **Blank** (⚠️ NOT Trackio!)
- [ ] **Hardware**: CPU Basic (free) or ZeroGPU
- [ ] **Visibility**: Public or Private
- [ ] Click "Create Space"

### 2. Clone Space Repository

**Option A: Using Git Xet (Recommended)**
```bash
# Install git-xet
brew install git-xet  # macOS
# Or: pip install git-xet

# Initialize git-xet
git xet install

# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# To clone without large files (just pointers):
# GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

- [ ] Git Xet installed
- [ ] Git Xet initialized
- [ ] Space repository cloned successfully

**Option B: Using Git LFS (Alternative)**
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Setup Git LFS for model file
git lfs install
git lfs track "*.pkl"
```

- [ ] Git LFS installed
- [ ] `.pkl` files tracked
- [ ] Space repository cloned successfully

### 4. Copy Files to Space
```bash
# From your project directory
cp app.py ../YOUR_SPACE_NAME/
cp requirements.txt ../YOUR_SPACE_NAME/
cp README_SPACE.md ../YOUR_SPACE_NAME/README.md
cp .gitattributes_spaces ../YOUR_SPACE_NAME/.gitattributes  # If created
cp fraud_lgbm_calibrated.pkl ../YOUR_SPACE_NAME/
```

- [ ] All files copied to Space directory

### 5. Commit and Push
```bash
cd ../YOUR_SPACE_NAME
git add .
git commit -m "Initial deployment of fraud detection system"
git push
```

- [ ] Files committed
- [ ] Changes pushed to Space

### 6. Monitor Build
- [ ] Go to Space → "Logs" tab
- [ ] Wait 2-5 minutes for build
- [ ] Check for any build errors
- [ ] Verify app starts successfully

### 7. Test Deployment
- [ ] Visit Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
- [ ] Upload a test CSV file
- [ ] Verify fraud detection works
- [ ] Check all visualizations load correctly
- [ ] Test threshold slider
- [ ] Verify sample dataset button works (if included)

## Post-Deployment

### ✅ Verification
- [ ] App loads without errors
- [ ] File upload works
- [ ] Model predictions are generated
- [ ] Visualizations display correctly
- [ ] No console errors in browser

### 📊 Optional Enhancements
- [ ] Add sample dataset to Space
- [ ] Update Space description
- [ ] Add tags/categories
- [ ] Set up custom domain (if needed)
- [ ] Configure environment variables (if using BigQuery)

## Troubleshooting

### Build Fails
- [ ] Check Space logs for error messages
- [ ] Verify `requirements.txt` has correct versions
- [ ] Ensure all imports are available
- [ ] Check Python version compatibility

### Model File Not Found
- [ ] Verify `fraud_lgbm_calibrated.pkl` is in root directory
- [ ] Check Git LFS is properly configured
- [ ] Ensure file name matches `MODEL_PATH` in `app.py`

### App Crashes
- [ ] Check Space logs
- [ ] Verify CSV format matches expected columns
- [ ] Test with sample data first
- [ ] Check memory limits (reduce `nrows` if needed)

## Quick Reference

**Template**: Blank (Gradio)  
**SDK Version**: Gradio 4.44.0  
**App File**: `app.py`  
**Model File**: `fraud_lgbm_calibrated.pkl`  
**Git LFS**: Required for .pkl files

---

**Need Help?** See `SPACES_DEPLOYMENT.md` for detailed instructions.
