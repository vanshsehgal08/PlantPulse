# 🚀 Deploying Plant Pulse on Render

This guide will walk you through deploying your Plant Pulse Streamlit application on Render.

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com) (free tier available)
2. **GitHub Account**: Your code should be in a GitHub repository
3. **Model Files**: Ensure your trained model files are included in the repository or use Git LFS for large files

## Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Commit all necessary files**:
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   ```

2. **Push to GitHub**:
   ```bash
   git push origin main
   ```

3. **Verify your repository structure**:
   ```
   PlantPulse/
   ├── frontend/
   │   ├── app.py (main entry point)
   │   ├── pages/
   │   ├── utils/
   │   └── models/ (contains your .keras model file)
   ├── requirements.txt
   ├── render.yaml (optional but recommended)
   └── README.md
   ```

### Step 2: Create a New Web Service on Render

1. **Log into Render Dashboard**: Go to [dashboard.render.com](https://dashboard.render.com)

2. **Click "New +"** button in the top right

3. **Select "Web Service"**

4. **Connect your GitHub repository**:
   - If first time: Click "Connect GitHub" and authorize Render
   - Select your repository: `PlantPulse` (or your repo name)
   - Choose the branch: `main` (or `master`)

### Step 3: Configure the Service

Fill in the following settings:

- **Name**: `plant-pulse` (or your preferred name)
- **Environment**: `Python 3`
- **Region**: Choose closest to your users
- **Branch**: `main` (or your default branch)

#### Build & Deploy Settings:

- **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```

- **Start Command**:
  ```bash
  streamlit run frontend/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
  ```

#### Environment Variables:

Add these environment variables in the Render dashboard:

1. **GEMINI_API_KEY** (if using chat feature):
   - Key: `GEMINI_API_KEY`
   - Value: Your Gemini API key
   - Click "Add" for each variable

2. **PYTHON_VERSION** (optional, recommended):
   - Key: `PYTHON_VERSION`
   - Value: `3.11.0` (or your preferred Python version)

#### Instance Type:

- **Free Tier**: Choose "Free" (limited resources, may have slower cold starts)
- **Paid Tier**: Choose "Starter" or higher for better performance

### Step 4: Deploy

1. **Click "Create Web Service"**

2. **Monitor the deployment**:
   - Render will automatically:
     - Install dependencies from `requirements.txt`
     - Run your start command
     - Deploy your application

3. **Wait for "Live" status**:
   - First deployment takes 5-10 minutes
   - Subsequent deployments are faster

### Step 5: Verify Deployment

1. **Access your app**: Click on your service name to get the URL
   - Format: `https://plant-pulse.onrender.com` (or your custom domain)

2. **Test the application**:
   - Upload an image
   - Test disease detection
   - Verify charts and calculator work

## Important Notes

### File Size Limits

⚠️ **Model Files**: If your `.keras` model files are large (>100MB):
- Use **Git LFS** (Git Large File Storage):
  ```bash
  git lfs install
  git lfs track "*.keras"
  git add .gitattributes
  git add *.keras
  git commit -m "Add model files with LFS"
  git push
  ```
- Or upload models to cloud storage and download during build

### Memory Considerations

- **Free Tier**: Limited to 512MB RAM (may struggle with TensorFlow)
- **Starter Tier**: 512MB RAM (recommended minimum)
- **Professional Tier**: Better for production with larger models

### Cold Starts

- Free tier services **sleep after 15 minutes** of inactivity
- First request after sleep takes 30-60 seconds (cold start)
- Consider upgrading to paid tier to avoid sleep

### Troubleshooting

#### Build Fails

1. **Check logs**: Click "Logs" tab in Render dashboard
2. **Common issues**:
   - Missing dependencies in `requirements.txt`
   - Incorrect Python version
   - Model file not found

#### App Crashes

1. **Check runtime logs**: View logs in dashboard
2. **Memory issues**: Upgrade instance size
3. **Import errors**: Verify all paths are relative to project root

#### Model Not Loading

1. **Verify model path**: Check `frontend/utils/model.py`
2. **Ensure model is in repository**: 
   ```bash
   ls -lh frontend/models/*.keras
   ```
3. **Check file size**: Ensure Git LFS is working if files are large

## Alternative: Using render.yaml (Recommended)

If you created the `render.yaml` file:

1. **In Render Dashboard**: Select "Infrastructure as Code"
2. **Create Blueprint**: Choose your repository
3. **Render will use your `render.yaml` configuration automatically**

This makes deployments repeatable and easier to manage.

## Custom Domain (Optional)

1. Go to your service settings
2. Click "Custom Domains"
3. Add your domain and follow DNS configuration instructions

## Updating Your Deployment

Whenever you push to your GitHub repository:

1. **Render automatically detects changes**
2. **Creates a new deployment**
3. **Deploys automatically** (can disable auto-deploy in settings)

Or manually trigger:
1. Click "Manual Deploy" → "Deploy latest commit"

## Cost Estimate

- **Free Tier**: $0/month (with limitations)
- **Starter Tier**: ~$7/month (better performance)
- **Professional Tier**: $25+/month (production-ready)

## Support

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Streamlit on Render**: [render.com/docs/deploy-streamlit](https://render.com/docs/deploy-streamlit)
- **Community**: Render Discord/Slack

---

**Good luck with your deployment! 🌿**

