# ✅ Render Deployment Checklist

Use this checklist before deploying to ensure everything is ready.

## Pre-Deployment Checklist

### 1. Repository Setup
- [ ] Code is pushed to GitHub repository
- [ ] All necessary files are committed
- [ ] `.gitignore` is configured properly
- [ ] No sensitive information in code (API keys, passwords, etc.)

### 2. Model Files
- [ ] Model file (`.keras`) is in `frontend/models/` or `models/` directory
- [ ] Model file is committed to repository (or use Git LFS for large files)
- [ ] Verify model file name matches what the code expects
  - Default: `best_model.keras` or `my_model_24.keras`
  - Check: `frontend/utils/model.py` for expected paths

### 3. Dependencies
- [ ] `requirements.txt` is up to date with all dependencies
- [ ] All required packages are listed
- [ ] Python version is specified (recommended: 3.11.0)

### 4. File Structure
Verify your repository has this structure:
```
PlantPulse/
├── frontend/
│   ├── app.py              ✅ Main entry point
│   ├── pages/
│   │   ├── 1_Predict.py   ✅
│   │   └── 2_Chat.py       ✅
│   ├── utils/
│   │   ├── model.py        ✅
│   │   ├── gradcam.py      ✅
│   │   └── guidance.py    ✅
│   ├── models/
│   │   └── my_model_24.keras  ✅ Your model file
│   └── assets/
│       └── styles.css      ✅
├── requirements.txt       ✅
├── render.yaml            ✅ (optional but recommended)
└── README.md              ✅
```

### 5. Environment Variables
- [ ] `GEMINI_API_KEY` (if using chat feature)
  - Get from: https://makersuite.google.com/app/apikey
  - Add in Render dashboard under Environment Variables

### 6. Configuration Files
- [ ] `render.yaml` exists (optional)
- [ ] `.gitignore` is set up correctly
- [ ] No local-only paths or absolute paths in code

### 7. Testing Locally
- [ ] App runs locally: `streamlit run frontend/app.py`
- [ ] Model loads without errors
- [ ] Predictions work correctly
- [ ] Charts display properly
- [ ] Chat feature works (if using)

## Deployment Steps Quick Reference

1. **Login to Render**: https://dashboard.render.com
2. **New Web Service**: Click "New +" → "Web Service"
3. **Connect GitHub**: Link your repository
4. **Configure**:
   - Name: `plant-pulse`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run frontend/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
5. **Add Environment Variables**: `GEMINI_API_KEY`
6. **Deploy**: Click "Create Web Service"
7. **Wait**: First deployment takes 5-10 minutes
8. **Verify**: Visit your app URL once "Live"

## Post-Deployment Verification

- [ ] App loads without errors
- [ ] Can upload images
- [ ] Predictions work
- [ ] Visual insights generate
- [ ] Charts display data
- [ ] Cost calculator works
- [ ] Chat works (if API key configured)

## Common Issues & Solutions

### ❌ Build Fails: "Model file not found"
**Solution**: 
- Check model file is in repository
- Verify path in `frontend/utils/model.py`
- Ensure file name matches expected names

### ❌ App Crashes: "Out of Memory"
**Solution**:
- Upgrade to Starter or Professional tier
- Model may be too large for Free tier (512MB)

### ❌ Slow Loading
**Solution**:
- Free tier has cold starts (15 min inactivity)
- Upgrade to paid tier for always-on service

### ❌ Chat Not Working
**Solution**:
- Verify `GEMINI_API_KEY` is set in Environment Variables
- Check API key is valid and has credits

## Need Help?

- Check deployment logs in Render dashboard
- Review `DEPLOYMENT.md` for detailed instructions
- Render Support: https://render.com/docs

---

**Ready to deploy?** Follow the steps in `DEPLOYMENT.md`! 🚀

