# TeaVision AI — Free Deployment Guide

This guide covers free hosting options for TeaVision AI.

---

## Option 1: Render (Recommended - Easiest)

### Why Render?
- **Free tier**: 750 hours/month for web services
- **Automatic deploys**: Pushes to GitHub auto-deploy
- **Simple setup**: Blueprint configuration included

### Limitations on Free Tier
- **No GPU**: PyTorch runs on CPU (slower inference ~5-10s per image)
- **No Ollama**: LLM-based agent recommendations won't work (falls back to rule-based)
- **Sleep after 15 min**: Service sleeps after inactivity, cold start ~30-60s
- **512 MB RAM**: May struggle with large models

### Setup Steps

1. **Fork/Clone the Repo to Your GitHub**
2. **Go to [https://render.com](https://render.com)**
3. **Sign up with GitHub**
4. **New > Blueprint**
5. **Connect your GitHub repo**
6. **Click "Apply"** — Render will auto-configure from `render.yaml`

Your app will be live at:
- Frontend: `https://teavision-frontend.onrender.com`
- API: `https://teavision-api.onrender.com`

---

## Option 2: Railway (Best for Full Stack)

### Why Railway?
- **$5 free credits/month**
- **Better performance** than Render free tier
- **PostgreSQL/Redis addon** if needed

### Setup

1. Go to [https://railway.app](https://railway.app)
2. Sign up with GitHub
3. New Project > Deploy from GitHub
4. Select your repo
5. Add environment variables:
   ```
   PYTHON_VERSION=3.10
   PORT=8000
   ```

---

## Option 3: Vercel (Frontend) + Render (Backend)

### Why Split?
- Vercel has **better frontend CDN** and faster static hosting
- Backend stays on Render

### Setup Vercel Frontend

1. Go to [https://vercel.com](https://vercel.com)
2. Import GitHub repo
3. Set root directory to `frontend`
4. Add environment variable:
   ```
   VITE_API_URL=https://teavision-api.onrender.com/api
   ```
5. Deploy

---

## Option 4: Hugging Face Spaces (ML Demo)

### Why HF Spaces?
- **Free GPU** for ML inference (limited)
- Great for **ML demos**
- 16GB RAM

### Setup

1. Create a `app.py` Gradio interface
2. Push to Hugging Face Spaces
3. GPU assigned automatically for inference

I can create a Gradio demo if you want this option.

---

## Environment Variables Reference

| Variable | Purpose | Default |
|----------|---------|---------|
| `VITE_API_URL` | Frontend API endpoint | `/api` (proxy in dev) |
| `OLLAMA_BASE_URL` | Ollama LLM endpoint | `http://localhost:11434` |
| `FAST_MODE_ENABLED` | Use fast rule-based fallback | `True` |
| `PORT` | Server port | `8000` |

---

## Model Checkpoints

**Important**: The model `.pth` files are ~200MB each. They're included in the repo but:
- If repo is too large for free tier, use Git LFS
- Or download checkpoints on first run from a URL

---

## Testing Your Deployment

```bash
# Test health endpoint
curl https://your-api.onrender.com/health

# Test classification
curl -X POST https://your-api.onrender.com/api/classes

# Upload image
curl -X POST -F "file=@test_leaf.jpg" https://your-api.onrender.com/api/analyze
```

---

## Known Issues on Free Hosting

| Issue | Cause | Workaround |
|-------|-------|------------|
| Slow first load | Container sleeping | Wait 30-60s for cold start |
| No LLM recommendations | Ollama not available | Uses rule-based fallback |
| Timeout on inference | Free tier CPU slow | Use smaller batch, wait longer |
| 512MB OOM | Model too large | Use FP16 or quantized model |

---

## Recommended: Local Development + Demo Link

For your capstone presentation:
1. **Run locally** for demo (faster, full features)
2. **Deploy to Render** for public link in report
3. **Include screenshots** of both local and deployed versions

---

*Last Updated: February 25, 2026*
