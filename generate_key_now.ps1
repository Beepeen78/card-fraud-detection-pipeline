# Quick SSH key generation
$email = "bipinpandey244586@gmail.com"
$sshDir = "$env:USERPROFILE\.ssh"
$keyPath = "$sshDir\id_ed25519"

# Create .ssh directory if needed
if (-not (Test-Path $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir -Force | Out-Null
}

# Generate key if it doesn't exist
if (-not (Test-Path "$keyPath.pub")) {
    Write-Host "Generating SSH key..."
    ssh-keygen -t ed25519 -C $email -f $keyPath -N '""' -q
}

# Read and display public key
$publicKey = Get-Content "$keyPath.pub"
Write-Host "`n=== YOUR SSH PUBLIC KEY ===" -ForegroundColor Green
Write-Host $publicKey -ForegroundColor Yellow
Write-Host "==========================`n" -ForegroundColor Green

# Copy to clipboard
$publicKey | Set-Clipboard
Write-Host "✅ Public key copied to clipboard!" -ForegroundColor Green

# Save to project directory
$publicKey | Out-File -FilePath "$PSScriptRoot\ssh_public_key.txt" -Encoding utf8
Write-Host "✅ Public key saved to: ssh_public_key.txt" -ForegroundColor Green

Write-Host "`n📝 Next: Add this key to https://huggingface.co/settings/keys" -ForegroundColor Cyan
