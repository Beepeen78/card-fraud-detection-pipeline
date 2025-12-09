# PowerShell script to generate SSH key for Git authentication
# Usage: .\generate_ssh_key.ps1

param(
    [string]$Email = "",
    [string]$KeyName = "id_ed25519",
    [string]$KeyType = "ed25519"
)

Write-Host "=" -NoNewline
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "🔑 SSH Key Generator for Git Authentication" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

# Check if SSH is available
try {
    $sshVersion = ssh -V 2>&1
    Write-Host "✅ SSH is available" -ForegroundColor Green
} catch {
    Write-Host "❌ SSH is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install OpenSSH:" -ForegroundColor Yellow
    Write-Host "1. Windows 10/11: OpenSSH is usually pre-installed"
    Write-Host "2. Or install Git for Windows which includes SSH"
    Write-Host "3. Or install OpenSSH manually"
    exit 1
}

# Get email if not provided
if ([string]::IsNullOrWhiteSpace($Email)) {
    Write-Host "📧 Enter your email address for the SSH key:" -ForegroundColor Cyan
    $Email = Read-Host "Email"
    
    if ([string]::IsNullOrWhiteSpace($Email)) {
        Write-Host "❌ Email is required" -ForegroundColor Red
        exit 1
    }
}

# Check if key already exists
$sshDir = "$env:USERPROFILE\.ssh"
$keyPath = "$sshDir\$KeyName"

if (Test-Path $keyPath) {
    Write-Host "⚠️  SSH key already exists at: $keyPath" -ForegroundColor Yellow
    $response = Read-Host "Overwrite? (y/n)"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Operation cancelled." -ForegroundColor Yellow
        exit 0
    }
}

# Create .ssh directory if it doesn't exist
if (-not (Test-Path $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir -Force | Out-Null
    Write-Host "✅ Created .ssh directory: $sshDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "🔑 Generating SSH key..." -ForegroundColor Cyan
Write-Host "  Key Type: $KeyType" -ForegroundColor Gray
Write-Host "  Key Name: $KeyName" -ForegroundColor Gray
Write-Host "  Email: $Email" -ForegroundColor Gray
Write-Host ""

# Generate SSH key
$keyComment = "$Email"

try {
    # Run ssh-keygen with empty passphrase (no passphrase)
    # Use call operator with single quotes around empty string to avoid PowerShell parsing issues
    $null = & ssh-keygen -t $KeyType -C "$keyComment" -f "$keyPath" -N '""' 2>&1
    
    # Check if key was created successfully
    if (Test-Path "$keyPath.pub") {
        Write-Host ""
        Write-Host "✅ SSH key generated successfully!" -ForegroundColor Green
        Write-Host ""
        
        # Display public key
        $publicKeyPath = "$keyPath.pub"
        if (Test-Path $publicKeyPath) {
            Write-Host "📋 Your public key:" -ForegroundColor Cyan
            Write-Host ("-" * 70) -ForegroundColor Gray
            $publicKey = Get-Content $publicKeyPath
            Write-Host $publicKey -ForegroundColor Yellow
            Write-Host ("-" * 70) -ForegroundColor Gray
            Write-Host ""
            
            # Copy to clipboard
            try {
                $publicKey | Set-Clipboard
                Write-Host "✅ Public key copied to clipboard!" -ForegroundColor Green
            } catch {
                Write-Host "⚠️  Could not copy to clipboard automatically" -ForegroundColor Yellow
            }
            
            Write-Host ""
            Write-Host "📝 Next Steps:" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "1. Add this public key to your Git hosting service:" -ForegroundColor Yellow
            Write-Host "   - GitHub: https://github.com/settings/keys" -ForegroundColor Gray
            Write-Host "   - GitLab: https://gitlab.com/-/profile/keys" -ForegroundColor Gray
            Write-Host "   - Hugging Face: https://huggingface.co/settings/keys" -ForegroundColor Gray
            Write-Host ""
            Write-Host "2. Test your SSH connection:" -ForegroundColor Yellow
            Write-Host "   ssh -T git@github.com" -ForegroundColor Gray
            Write-Host "   ssh -T git@huggingface.co" -ForegroundColor Gray
            Write-Host ""
            Write-Host "3. Configure Git to use SSH:" -ForegroundColor Yellow
            Write-Host "   git remote set-url origin git@github.com:USERNAME/REPO.git" -ForegroundColor Gray
            Write-Host ""
            
            # Save public key to a file for easy access
            $publicKeyFile = "$PSScriptRoot\ssh_public_key.txt"
            $publicKey | Out-File -FilePath $publicKeyFile -Encoding utf8
            Write-Host "💾 Public key also saved to: $publicKeyFile" -ForegroundColor Cyan
        }
        
        Write-Host ""
        Write-Host "🔒 Key Locations:" -ForegroundColor Cyan
        Write-Host "  Private Key: $keyPath" -ForegroundColor Gray
        Write-Host "  Public Key:  $keyPath.pub" -ForegroundColor Gray
        Write-Host ""
        Write-Host "⚠️  Keep your private key secure! Never share it." -ForegroundColor Red
    } else {
        Write-Host "❌ Failed to generate SSH key" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Error generating SSH key: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=" -NoNewline
Write-Host ("=" * 69) -ForegroundColor Green
Write-Host "✅ SSH Key Generation Complete!" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
