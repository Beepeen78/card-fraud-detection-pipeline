# 🚀 Deploy from Git to Hugging Face Spaces

After pushing all changes to Git, you can deploy directly from your repository.

## 📋 Pre-Deployment Checklist

- [x] All files committed to Git
- [x] Changes pushed to remote repository
- [ ] SSH key added to Hugging Face (if using SSH)
- [ ] Model file ready (`fraud_lgbm_calibrated.pkl`)

## 🔄 Deployment Methods

### Method 1: Clone from Git and Deploy

If your project is in a Git repository (GitHub, GitLab, etc.):

```powershell
# 1. Clone your Space repository
cd ..
git clone https://huggingface.co/spaces/Beepeen24/Card-Fraud-detection
cd Card-Fraud-detection

# 2. Add your Git repository as a remote (if not already)
git remote add upstream https://github.com/YOUR_USERNAME/YOUR_REPO.git
# Or if using SSH:
# git remote add upstream git@github.com:YOUR_USERNAME/YOUR_REPO.git

# 3. Pull files from your Git repository
git pull upstream main --allow-unrelated-histories

# 4. Copy necessary files
# Files should already be in the Space directory after pull

# 5. Commit and push to Space
git add .
git commit -m "Deploy from Git repository"
git push
```

### Method 2: Direct File Copy from Git Clone

```powershell
# 1. Clone your main project repository
cd ..
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Clone your Space repository
cd ..
git clone https://huggingface.co/spaces/Beepeen24/Card-Fraud-detection
cd Card-Fraud-detection

# 3. Copy files from your project
Copy-Item ..\YOUR_REPO\app.py .
Copy-Item ..\YOUR_REPO\requirements.txt .
Copy-Item ..\YOUR_REPO\README_SPACE.md README.md
# Copy model file if it exists
if (Test-Path "..\YOUR_REPO\fraud_lgbm_calibrated.pkl") {
    Copy-Item ..\YOUR_REPO\fraud_lgbm_calibrated.pkl .
}

# 4. Commit and push
git add .
git commit -m "Deploy from Git repository"
git push
```

### Method 3: Using Git Subtree or Submodule

For ongoing synchronization:

```powershell
# In your Space repository
git subtree add --prefix=source https://github.com/YOUR_USERNAME/YOUR_REPO.git main --squash

# To update later:
git subtree pull --prefix=source https://github.com/YOUR_USERNAME/YOUR_REPO.git main --squash
```

## 📦 Files to Deploy

Essential files:
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - Space documentation (with YAML frontmatter)
- ⚠️ `fraud_lgbm_calibrated.pkl` - Model file (if available)

## 🔐 Authentication

### Using HTTPS (Token)
```powershell
# Login to Hugging Face
huggingface-cli login

# Then push normally
git push
```

### Using SSH
```powershell
# 1. Generate SSH key (if not done)
.\generate_ssh_key.ps1 -Email "bipinpandey244586@gmail.com"

# 2. Add public key to Hugging Face
# Go to: https://huggingface.co/settings/keys

# 3. Change remote to SSH
cd Card-Fraud-detection
git remote set-url origin git@hf.co:spaces/Beepeen24/Card-Fraud-detection.git

# 4. Push
git push
```

## 🚀 Quick Deploy Script

Create a script to automate deployment:

```powershell
# deploy_from_git.ps1
$spaceRepo = "../Card-Fraud-detection"
$projectRepo = "."

# Ensure Space is cloned
if (-not (Test-Path $spaceRepo)) {
    git clone https://huggingface.co/spaces/Beepeen24/Card-Fraud-detection $spaceRepo
}

# Copy files
Copy-Item "$projectRepo\app.py" "$spaceRepo\app.py" -Force
Copy-Item "$projectRepo\requirements.txt" "$spaceRepo\requirements.txt" -Force
Copy-Item "$projectRepo\README_SPACE.md" "$spaceRepo\README.md" -Force

# Copy model if exists
if (Test-Path "$projectRepo\fraud_lgbm_calibrated.pkl") {
    Copy-Item "$projectRepo\fraud_lgbm_calibrated.pkl" "$spaceRepo\" -Force
}

# Deploy
cd $spaceRepo
git add .
git commit -m "Deploy from Git - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
git push

Write-Host "✅ Deployment complete!" -ForegroundColor Green
```

## ✅ Verification

After deployment:

1. Visit: https://huggingface.co/spaces/Beepeen24/Card-Fraud-detection
2. Check "Logs" tab for build status
3. Wait 2-5 minutes for build
4. Test the application

## 🔄 Continuous Deployment

For automatic deployments, you can:

1. **Use GitHub Actions** to push to Space on every commit
2. **Use Git hooks** to auto-deploy on push
3. **Manual deployment** using the scripts above

---

**Current Status:**
- ✅ All files committed to local Git
- ✅ Ready to push to remote (if configured)
- ✅ Space repository ready for deployment
