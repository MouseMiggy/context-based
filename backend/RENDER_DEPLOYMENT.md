# Deploy AgriLink Semantic Search Backend to Render

This guide walks you through deploying the FastAPI semantic search backend to Render.

---

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code should be pushed to GitHub
3. **Firebase Service Account**: JSON credentials for Firestore access

---

## Step 1: Prepare Your Repository

Ensure these files exist in the `backend` folder:

- ✅ `main.py` - FastAPI application
- ✅ `requirements.txt` - Python dependencies
- ✅ `Procfile` - Render startup command
- ✅ `render.yaml` - Render configuration (optional)

---

## Step 2: Push to GitHub

```bash
cd "e:\AgriLink Github\AgriLinkWebsite"
git add backend/
git commit -m "Add Render deployment files for semantic search backend"
git push origin main
```

---

## Step 3: Create New Web Service on Render

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New +"** → Select **"Web Service"**
3. **Connect Repository**:
   - Click "Connect account" to link your GitHub
   - Select your `AgriLinkWebsite` repository
   - Click "Connect"

---

## Step 4: Configure Web Service

### Basic Settings:
- **Name**: `agrilink-semantic-search`
- **Region**: Choose closest to your users (e.g., Singapore)
- **Branch**: `main` (or your default branch)
- **Root Directory**: `backend`
- **Runtime**: `Python 3`

### Build & Deploy Settings:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Instance Type:
- **Free** (for testing) or **Starter** ($7/month - recommended for production)
- ⚠️ **Note**: Free tier spins down after inactivity, causing slow first requests

---

## Step 5: Set Environment Variables

Click **"Environment"** tab and add these variables:

### Required Variables:

1. **FIRESTORE_PROJECT_ID**
   - Value: Your Firebase project ID (e.g., `agrilink-12345`)
   - Get from Firebase Console → Project Settings

2. **GOOGLE_APPLICATION_CREDENTIALS_JSON**
   - Value: Your entire Firebase service account JSON as a string
   - Get from Firebase Console → Project Settings → Service Accounts → Generate New Private Key
   - Copy the **entire JSON content** and paste it as the value
   - Example format:
   ```json
   {"type":"service_account","project_id":"agrilink-12345","private_key_id":"abc123","private_key":"-----BEGIN PRIVATE KEY-----\n...","client_email":"firebase-adminsdk@agrilink.iam.gserviceaccount.com","client_id":"123456789","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk%40agrilink.iam.gserviceaccount.com"}
   ```

3. **PYTHON_VERSION** (optional)
   - Value: `3.11.0`

---

## Step 6: Deploy

1. Click **"Create Web Service"**
2. Render will automatically:
   - Clone your repository
   - Install dependencies (this takes 5-10 minutes for ML models)
   - Start your FastAPI server
3. Monitor the **Logs** tab for deployment progress

---

## Step 7: Verify Deployment

Once deployed, you'll get a URL like: `https://agrilink-semantic-search.onrender.com`

Test the API:

```bash
# Test root endpoint
curl https://agrilink-semantic-search.onrender.com/

# Expected response:
{"message":"AgriLink Semantic Search API is running"}
```

---

## Step 8: Update Frontend Environment Variable

Update your frontend `.env.local` or Vercel environment variables:

```env
NEXT_PUBLIC_SEMANTIC_SEARCH_URL=https://agrilink-semantic-search.onrender.com
```

For Vercel deployment:
1. Go to Vercel Dashboard → Your Project → Settings → Environment Variables
2. Add/Update: `NEXT_PUBLIC_SEMANTIC_SEARCH_URL`
3. Redeploy your frontend

---

## Step 9: Initial Data Ingestion (Optional)

If you need to generate embeddings for existing listings:

1. **SSH into Render** (Starter plan or higher):
   ```bash
   render ssh agrilink-semantic-search
   python ingest_listings.py
   ```

2. **Or use the API endpoint** from your local machine:
   ```bash
   # For each listing, call:
   curl -X POST https://agrilink-semantic-search.onrender.com/embed-listing \
     -H "Content-Type: application/json" \
     -d '{"listingId": "your-listing-id"}'
   ```

---

## Troubleshooting

### Issue: "Module not found" errors
**Solution**: Check `requirements.txt` has all dependencies with versions

### Issue: "Firestore authentication failed"
**Solution**: Verify `GOOGLE_APPLICATION_CREDENTIALS_JSON` is set correctly with the full JSON

### Issue: "Model download timeout"
**Solution**: 
- Increase build timeout in Render settings
- Or use Starter plan (more resources)

### Issue: "Service keeps spinning down"
**Solution**: Upgrade to Starter plan ($7/month) for always-on service

### Issue: "Out of memory"
**Solution**: 
- Upgrade to larger instance type
- Or optimize model loading (use smaller model)

---

## Cost Estimation

- **Free Tier**: $0/month (spins down after 15 min inactivity)
- **Starter**: $7/month (512 MB RAM, always-on)
- **Standard**: $25/month (2 GB RAM, recommended for production)

---

## Monitoring

1. **Logs**: View real-time logs in Render Dashboard → Logs tab
2. **Metrics**: Monitor CPU, memory, and request count
3. **Alerts**: Set up email alerts for service failures

---

## Security Best Practices

1. ✅ Never commit `.env` file or service account JSON to Git
2. ✅ Use environment variables for all secrets
3. ✅ Enable HTTPS (automatic on Render)
4. ✅ Update CORS origins to specific domains in production:
   ```python
   allow_origins=["https://agrilinkph.vercel.app"]
   ```

---

## Updating Your Deployment

When you make code changes:

1. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update semantic search backend"
   git push origin main
   ```

2. Render will **automatically redeploy** (if auto-deploy is enabled)
3. Or manually deploy from Render Dashboard → Manual Deploy → Deploy latest commit

---

## API Endpoints

Once deployed, your API will have these endpoints:

- `GET /` - Health check
- `POST /embed` - Generate embedding for text
- `POST /search` - Semantic search for listings
- `POST /embed-listing` - Generate embedding for specific listing

Full API documentation: `https://your-app.onrender.com/docs`

---

## Support

- Render Docs: https://render.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- Sentence Transformers: https://www.sbert.net

---

**Your semantic search backend is now live! 🚀**
