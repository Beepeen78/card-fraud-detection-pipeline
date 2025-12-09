# PowerShell script to deploy from Git repository to Hugging Face Spaces
# Usage: .\deploy_from_git.ps1

param(
    [string]$SpaceName = "Card-Fraud-detection",
    [string]$Username = "Beepeen24"
)

$ErrorActionPreference = "Stop"

Write-Host "=" -NoNewline
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "🚀 Deploy from Git to Hugging Face Spaces" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

# Get current project directory
$projectDir = Get-Location
$spaceDir = Join-Path (Split-Path $projectDir -Parent) $SpaceName
$spaceUrl = "https://huggingface.co/spaces/$Username/$SpaceName"

Write-Host "📦 Project Directory: $projectDir" -ForegroundColor Cyan
Write-Host "📦 Space Directory: $spaceDir" -ForegroundColor Cyan
Write-Host "🌐 Space URL: $spaceUrl" -ForegroundColor Cyan
Write-Host ""

# Check if Space directory exists
if (-not (Test-Path $spaceDir)) {
    Write-Host "📥 Cloning Space repository..." -ForegroundColor Yellow
    git clone $spaceUrl $spaceDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to clone Space repository" -ForegroundColor Red
        Write-Host "   Make sure you're authenticated with Hugging Face" -ForegroundColor Yellow
        Write-Host "   Run: huggingface-cli login" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "✅ Space directory already exists" -ForegroundColor Green
    Write-Host "🔄 Updating Space repository..." -ForegroundColor Yellow
    Set-Location $spaceDir
    git pull
    Set-Location $projectDir
}

Write-Host ""
Write-Host "📋 Copying files to Space..." -ForegroundColor Cyan

# Files to copy
$filesToCopy = @{
    "app.py" = "app.py"
    "requirements.txt" = "requirements.txt"
    "README_SPACE.md" = "README.md"
}

$copiedFiles = 0
foreach ($source in $filesToCopy.Keys) {
    $dest = $filesToCopy[$source]
    $sourcePath = Join-Path $projectDir $source
    $destPath = Join-Path $spaceDir $dest
    
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath $destPath -Force
        Write-Host "  ✅ $source → $dest" -ForegroundColor Green
        $copiedFiles++
    } else {
        Write-Host "  ⚠️  $source not found, skipping..." -ForegroundColor Yellow
    }
}

# Copy model file if it exists
$modelFile = "fraud_lgbm_calibrated.pkl"
$modelSource = Join-Path $projectDir $modelFile
$modelDest = Join-Path $spaceDir $modelFile

if (Test-Path $modelSource) {
    Copy-Item $modelSource $modelDest -Force
    Write-Host "  ✅ $modelFile" -ForegroundColor Green
    $copiedFiles++
} else {
    Write-Host "  ⚠️  $modelFile not found (app will need this to work)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📝 Files copied: $copiedFiles" -ForegroundColor Cyan
Write-Host ""

# Deploy to Space
Set-Location $spaceDir

Write-Host "📊 Git Status:" -ForegroundColor Cyan
git status --short

Write-Host ""
$response = Read-Host "Ready to commit and push to Space? (y/n)"
if ($response -ne "y" -and $response -ne "Y") {
    Write-Host "Deployment cancelled. Files copied but not committed." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "💾 Committing changes..." -ForegroundColor Cyan
git add .
git commit -m "Deploy from Git repository - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  No changes to commit or commit failed" -ForegroundColor Yellow
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
    Write-Host "📊 Monitor progress in the 'Logs' tab." -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "❌ Push failed. Please check:" -ForegroundColor Red
    Write-Host "  1. Hugging Face authentication: huggingface-cli login" -ForegroundColor Yellow
    Write-Host "  2. SSH key added to Hugging Face (if using SSH)" -ForegroundColor Yellow
    Write-Host "  3. Git remote URL is correct" -ForegroundColor Yellow
}

Set-Location $projectDir
