# 🔑 SSH Key Setup Guide

## Quick Generation

### Option 1: Use the Script (Recommended)
```powershell
cd "d:\Projects\credit card fraud\card-fraud-detection-pipeline"
.\generate_ssh_key.ps1
```

The script will:
- Generate an Ed25519 SSH key (most secure)
- Ask for your email address
- Display your public key
- Copy it to clipboard
- Save it to `ssh_public_key.txt`

### Option 2: Manual Generation

#### Step 1: Generate SSH Key
```powershell
# Replace with your actual email
ssh-keygen -t ed25519 -C "your-email@example.com"
```

**When prompted:**
- **File location**: Press Enter (default: `C:\Users\YourName\.ssh\id_ed25519`)
- **Passphrase**: Press Enter twice (no passphrase) OR enter a secure passphrase

#### Step 2: View Your Public Key
```powershell
# Display public key
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"

# Or copy to clipboard
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | Set-Clipboard
```

#### Step 3: Add to Git Hosting Service

**For GitHub:**
1. Go to: https://github.com/settings/keys
2. Click "New SSH key"
3. Paste your public key
4. Click "Add SSH key"

**For Hugging Face:**
1. Go to: https://huggingface.co/settings/keys
2. Click "New SSH key"
3. Paste your public key
4. Click "Add SSH key"

**For GitLab:**
1. Go to: https://gitlab.com/-/profile/keys
2. Paste your public key
3. Click "Add key"

#### Step 4: Test Connection

**Test GitHub:**
```powershell
ssh -T git@github.com
```

**Test Hugging Face:**
```powershell
ssh -T git@huggingface.co
```

You should see a success message like:
```
Hi username! You've successfully authenticated...
```

## Using SSH with Your Space

After adding your SSH key, you can clone your Space using SSH:

```powershell
# Instead of HTTPS:
# git clone https://huggingface.co/spaces/Beepeen24/Card-Fraud-detection

# Use SSH:
git clone git@hf.co:spaces/Beepeen24/Card-Fraud-detection.git
```

## Troubleshooting

### SSH Key Not Found
```powershell
# Check if key exists
Test-Path "$env:USERPROFILE\.ssh\id_ed25519"

# List all SSH keys
Get-ChildItem "$env:USERPROFILE\.ssh\*.pub"
```

### Permission Denied
- Make sure you added the **public key** (`.pub` file) to your Git hosting service
- Never share your **private key** (the file without `.pub`)

### SSH Not Installed
```powershell
# Check if SSH is available
ssh -V

# If not, install OpenSSH (Windows 10/11 usually has it)
# Or install Git for Windows which includes SSH
```

### Multiple SSH Keys
If you have multiple keys, you can specify which one to use:

```powershell
# Use specific key
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -T git@github.com
```

Or configure SSH config file at `$env:USERPROFILE\.ssh\config`:
```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519

Host hf.co
    HostName hf.co
    User git
    IdentityFile ~/.ssh/id_ed25519
```

## Security Notes

- ✅ **Public key** (`.pub` file) - Safe to share, add to Git hosting services
- ❌ **Private key** (no extension) - Keep secret, never share
- 🔒 Consider using a passphrase for extra security
- 📁 Default location: `C:\Users\YourName\.ssh\`

---

**Quick Command Reference:**
```powershell
# Generate key
ssh-keygen -t ed25519 -C "your-email@example.com"

# View public key
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"

# Copy to clipboard
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | Set-Clipboard

# Test GitHub
ssh -T git@github.com

# Test Hugging Face
ssh -T git@huggingface.co
```
