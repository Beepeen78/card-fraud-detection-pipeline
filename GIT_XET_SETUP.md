# 🔧 Git Xet Setup for Hugging Face Spaces

Git Xet is Hugging Face's recommended solution for handling large files (like model files) in Spaces. It's more efficient than Git LFS and is automatically configured in Spaces.

## 📦 Installation

### macOS
```bash
brew install git-xet
```

### Linux/Windows
```bash
pip install git-xet
```

### Initialize
```bash
git xet install
```

## 🚀 Usage

### Clone Your Space

**Full clone (with all files):**
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

**Clone without large files (just pointers - faster):**
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

### Working with Large Files

Git Xet automatically handles large files (`.pkl`, `.h5`, etc.) without needing explicit tracking like Git LFS. Just add and commit normally:

```bash
git add fraud_lgbm_calibrated.pkl
git commit -m "Add model file"
git push
```

## ✨ Advantages Over Git LFS

1. **Automatic**: No need to track specific file patterns
2. **Faster**: More efficient for large files
3. **Spaces Integration**: Automatically configured in Hugging Face Spaces
4. **Simpler**: Less configuration needed

## 📚 Resources

- [Git Xet Documentation](https://hf.co/docs/hub/git-xet)
- [Hugging Face Hub Git Xet Guide](https://huggingface.co/docs/hub/git-xet)

## 🔄 Migration from Git LFS

If you're already using Git LFS, you can switch to Git Xet:

1. Install git-xet (see above)
2. Initialize: `git xet install`
3. Remove Git LFS tracking (optional):
   ```bash
   git lfs untrack "*.pkl"
   ```
4. Continue using git normally - git-xet handles large files automatically

---

**Note**: Git Xet is the recommended approach for Hugging Face Spaces. Git LFS still works but requires more configuration.
