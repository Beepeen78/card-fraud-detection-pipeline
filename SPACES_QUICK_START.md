# 🚀 Quick Start: Deploy to Hugging Face Spaces

## 📦 Files Needed for Deployment

Copy these files to your Space repository:

```
✅ app.py                          (Main application - already updated for Spaces)
✅ requirements.txt                (Dependencies - already compatible)
✅ README.md or README_SPACE.md    (Use README_SPACE.md - has YAML frontmatter)
✅ fraud_lgbm_calibrated.pkl       (Model file - use Git LFS)
```

## 🎯 3-Step Deployment

### 1. Create Space
- Go to [huggingface.co/spaces](https://huggingface.co/spaces)
- Click "Create new Space"
- **SDK**: Gradio
- **Template**: **Blank** (not Trackio!)
- **Hardware**: CPU Basic (free) or ZeroGPU

### 2. Upload Files

**Option A: Using Git Xet (Recommended)**
```bash
# Install git-xet (macOS)
brew install git-xet
# Or: pip install git-xet

# Initialize git-xet
git xet install

# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy files
cp ../card-fraud-detection-pipeline/app.py .
cp ../card-fraud-detection-pipeline/requirements.txt .
cp ../card-fraud-detection-pipeline/README_SPACE.md README.md
cp ../card-fraud-detection-pipeline/fraud_lgbm_calibrated.pkl .

# Commit and push (git-xet handles large files automatically)
git add .
git commit -m "Deploy fraud detection system"
git push
```

**Option B: Using Git LFS (Alternative)**
```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Setup Git LFS for model file
git lfs install
git lfs track "*.pkl"

# Copy files
cp ../card-fraud-detection-pipeline/app.py .
cp ../card-fraud-detection-pipeline/requirements.txt .
cp ../card-fraud-detection-pipeline/README_SPACE.md README.md
cp ../card-fraud-detection-pipeline/fraud_lgbm_calibrated.pkl .

# Commit and push
git add .
git commit -m "Deploy fraud detection system"
git push
```

### 3. Wait & Test
- Wait 2-5 minutes for build
- Visit your Space URL
- Upload a CSV and test!

## ⚠️ Important Notes

1. **Template**: Use **Blank** template, not Trackio
2. **Model File**: Use **Git Xet** (recommended) or Git LFS for `.pkl` files (they're too large for regular Git)
   - Git Xet: `brew install git-xet` then `git xet install`
   - Git LFS: `git lfs install` then `git lfs track "*.pkl"`
3. **README**: Must have YAML frontmatter (see `README_SPACE.md`)
4. **app.py**: Already updated to auto-detect Spaces environment ✅

## 🔧 Troubleshooting

- **Build fails**: Check Space logs, verify `requirements.txt` is correct
- **Model file missing**: Train model first or add `fraud_lgbm_calibrated.pkl` to Space
- **Authentication issues**: Run `huggingface-cli login` or add SSH key at https://huggingface.co/settings/keys
