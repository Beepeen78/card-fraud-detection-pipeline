# PowerShell script to deploy to Hugging Face Spaces
# Usage: .\deploy_to_spaces.ps1 -SpaceName "YOUR_SPACE_NAME" -Username "YOUR_USERNAME"

param(
    [Parameter(Mandatory=$true)]
    [string]$SpaceName,
    
    [Parameter(Mandatory=$true)]
    [string]$Username,
    
    [switch]$UseGitXet = $true,
    [switch]$SkipModel = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=" -NoNewline
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "🚀 Hugging Face Spaces Deployment Script" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

# Check if model file exists
$modelFile = "fraud_lgbm_calibrated.pkl"
if (-not (Test-Path $modelFile) -and -not $SkipModel) {
    Write-Host "⚠️  WARNING: Model file '$modelFile' not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "1. Train the model first: python train_improved_model.py"
    Write-Host "2. Skip model file (use -SkipModel flag if deploying without it)"
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Deployment cancelled." -ForegroundColor Red
        exit 1
    }
}

# Check required files
Write-Host "📋 Checking required files..." -ForegroundColor Cyan
$requiredFiles = @("app.py", "requirements.txt", "README_SPACE.md")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file - MISSING!" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "❌ Missing required files. Please ensure all files exist." -ForegroundColor Red
    exit 1
}

# Check if Space directory exists
$spaceDir = "../$SpaceName"
$spaceUrl = "https://huggingface.co/spaces/$Username/$SpaceName"

Write-Host ""
Write-Host "📦 Space Information:" -ForegroundColor Cyan
Write-Host "  Space Name: $SpaceName"
Write-Host "  Username: $Username"
Write-Host "  Space URL: $spaceUrl"
Write-Host "  Local Directory: $spaceDir"
Write-Host ""

# Check if Space is already cloned
if (Test-Path $spaceDir) {
    Write-Host "⚠️  Space directory already exists: $spaceDir" -ForegroundColor Yellow
    $response = Read-Host "Continue with existing directory? (y/n)"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Deployment cancelled." -ForegroundColor Red
        exit 1
    }
    Set-Location $spaceDir
} else {
    Write-Host "📥 Cloning Space repository..." -ForegroundColor Cyan
    
    if ($UseGitXet) {
        Write-Host "  Using Git Xet (recommended)..." -ForegroundColor Green
        
        # Check if git-xet is installed
        try {
            $xetVersion = git xet --version 2>&1
            Write-Host "  ✅ Git Xet found" -ForegroundColor Green
        } catch {
            Write-Host "  ⚠️  Git Xet not found. Installing..." -ForegroundColor Yellow
            Write-Host "  Please run: brew install git-xet (macOS) or pip install git-xet"
            Write-Host "  Then run: git xet install"
            Write-Host ""
            $response = Read-Host "Continue with Git LFS instead? (y/n)"
            if ($response -ne "y" -and $response -ne "Y") {
                exit 1
            }
            $UseGitXet = $false
        }
        
        if ($UseGitXet) {
            git clone $spaceUrl $spaceDir
            if ($LASTEXITCODE -ne 0) {
                Write-Host "❌ Failed to clone Space repository" -ForegroundColor Red
                exit 1
            }
        }
    }
    
    if (-not $UseGitXet) {
        Write-Host "  Using Git LFS..." -ForegroundColor Yellow
        git clone $spaceUrl $spaceDir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Failed to clone Space repository" -ForegroundColor Red
            exit 1
        }
        Set-Location $spaceDir
        git lfs install
        git lfs track "*.pkl"
    } else {
        Set-Location $spaceDir
    }
}

Write-Host "✅ Space repository cloned/accessed" -ForegroundColor Green
Write-Host ""

# Copy files
Write-Host "📋 Copying files to Space..." -ForegroundColor Cyan
$projectDir = (Get-Location).Path
$parentDir = Split-Path $projectDir -Parent
$sourceDir = Join-Path $parentDir "card-fraud-detection-pipeline"

if (-not (Test-Path $sourceDir)) {
    $sourceDir = ".."
}

$filesToCopy = @{
    "app.py" = "app.py"
    "requirements.txt" = "requirements.txt"
    "README_SPACE.md" = "README.md"
}

foreach ($source in $filesToCopy.Keys) {
    $dest = $filesToCopy[$source]
    $sourcePath = Join-Path $sourceDir $source
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath $dest -Force
        Write-Host "  ✅ Copied $source → $dest" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  $source not found, skipping..." -ForegroundColor Yellow
    }
}

# Copy model file if it exists
if (Test-Path (Join-Path $sourceDir $modelFile)) {
    Copy-Item (Join-Path $sourceDir $modelFile) $modelFile -Force
    Write-Host "  ✅ Copied $modelFile" -ForegroundColor Green
} elseif (-not $SkipModel) {
    Write-Host "  ⚠️  $modelFile not found, skipping..." -ForegroundColor Yellow
}

# Copy .gitattributes if using Git LFS
if (-not $UseGitXet -and (Test-Path (Join-Path $sourceDir ".gitattributes_spaces"))) {
    Copy-Item (Join-Path $sourceDir ".gitattributes_spaces") ".gitattributes" -Force
    Write-Host "  ✅ Copied .gitattributes_spaces → .gitattributes" -ForegroundColor Green
}

Write-Host ""
Write-Host "📝 Files ready for commit" -ForegroundColor Cyan
Write-Host ""

# Show git status
Write-Host "📊 Git Status:" -ForegroundColor Cyan
git status --short

Write-Host ""
$response = Read-Host "Ready to commit and push? (y/n)"
if ($response -ne "y" -and $response -ne "Y") {
    Write-Host "Files copied but not committed. You can commit manually later." -ForegroundColor Yellow
    exit 0
}

# Commit and push
Write-Host ""
Write-Host "💾 Committing changes..." -ForegroundColor Cyan
git add .
git commit -m "Deploy fraud detection system to Spaces"
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Commit failed or no changes to commit" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Pushing to Hugging Face Spaces..." -ForegroundColor Cyan
git push
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" -NoNewline
    Write-Host ("=" * 69) -ForegroundColor Green
    Write-Host "✅ Deployment Successful!" -ForegroundColor Green
    Write-Host ("=" * 70) -ForegroundColor Green
    Write-Host ""
    Write-Host "Your Space is being built. Check status at:" -ForegroundColor Cyan
    Write-Host "  $spaceUrl" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "⏳ Wait 2-5 minutes for the build to complete." -ForegroundColor Cyan
    Write-Host "📊 Monitor progress in the 'Logs' tab of your Space." -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "❌ Push failed. Please check your Git credentials and try again." -ForegroundColor Red
    Write-Host "You may need to set up Hugging Face authentication:" -ForegroundColor Yellow
    Write-Host "  huggingface-cli login" -ForegroundColor Yellow
}
