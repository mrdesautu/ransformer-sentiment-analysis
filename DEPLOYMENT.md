# 🚀 Deployment Options

This document outlines several options to deploy your Transformer Sentiment Analysis project for professional showcase and technical evaluation.

---

## 📋 Table of Contents
1. [Quick Demo Options (No Cloud Required)](#quick-demo-options)
2. [Cloud Deployment Options](#cloud-deployment-options)
3. [Recommended Approach](#recommended-approach)
4. [Cost Comparison](#cost-comparison)

---

## 🎯 Quick Demo Options (No Cloud Required)

### Option 1: Video Demo + GitHub
**Best for: Portfolio showcase**

**Pros:**
- ✅ Free
- ✅ Shows functionality without infrastructure costs
- ✅ Immediate availability for technical evaluation

**What to do:**
1. Record a 3-5 minute demo video showing:
   - The web interface
   - Single text analysis
   - Batch analysis
   - Interpretability features
   - API endpoints

2. Upload to:
   - YouTube (unlisted)
   - Loom
   - LinkedIn video

3. Add to your GitHub README:
```markdown
## 🎥 Live Demo
[Watch Demo Video](your-video-link)

## 🔗 Try it Yourself
Clone and run locally:
\`\`\`bash
git clone https://github.com/yourusername/transformer-sentiment
cd transformer-sentiment
pip install -r requirements.txt
python serve_web.py
\`\`\`
```

---

### Option 2: Hugging Face Spaces (FREE & EASY)
**Best for: Interactive demo without server management**

**Pros:**
- ✅ Completely FREE
- ✅ Easy to set up (10-15 minutes)
- ✅ Professional URL: `https://huggingface.co/spaces/username/transformer-sentiment`
- ✅ Automatic SSL, no server management
- ✅ Built-in Gradio/Streamlit support

**Steps:**
1. Create account at https://huggingface.co
2. Create a new Space
3. Choose Gradio or Streamlit
4. Upload your model and code

**Example Gradio app.py:**
```python
import gradio as gr
from src.inference import SentimentInference

# Load model
pipeline = SentimentInference("./model")

def analyze(text):
    result = pipeline.predict_single(text)
    return result['predicted_label'], result['confidence']

# Create interface
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Enter text to analyze"),
    outputs=[
        gr.Label(label="Sentiment"),
        gr.Number(label="Confidence")
    ],
    title="Transformer Sentiment Analysis",
    description="Analyze sentiment using DistilBERT"
)

demo.launch()
```

**Cost:** FREE ✅

---

## ☁️ Cloud Deployment Options

### Option 3: Render.com (FREE TIER)
**Best for: Full web app with API**

**Pros:**
- ✅ FREE tier available
- ✅ Automatic deployments from GitHub
- ✅ Custom domain support
- ✅ SSL included
- ✅ Easy setup

**Cons:**
- ⚠️ Sleeps after 15 minutes of inactivity (on free tier)
- ⚠️ Limited to 512MB RAM (need to use DistilBERT, not larger models)

**Steps:**
1. Create account at https://render.com
2. Connect your GitHub repository
3. Create a Web Service
4. Use this configuration:

**render.yaml:**
```yaml
services:
  # API Service
  - type: web
    name: sentiment-api
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python -m src.api --host 0.0.0.0 --port 8000"
    envVars:
      - key: MODEL_PATH
        value: ./mi_modelo_entrenado
    
  # Web Interface Service
  - type: web
    name: sentiment-web
    env: static
    staticPublishPath: ./web
```

**Cost:** FREE (with limitations) or $7/month for always-on

---

### Option 4: Railway.app (FREE TIER)
**Best for: Simple deployment with good free tier**

**Pros:**
- ✅ $5 free credits per month
- ✅ Easy GitHub integration
- ✅ No sleep on free tier
- ✅ Good performance

**Cons:**
- ⚠️ Limited free credits ($5/month = ~500 hours)

**Steps:**
1. Sign up at https://railway.app
2. Create new project from GitHub repo
3. Add environment variables
4. Deploy

**Cost:** First $5/month free, then pay-as-you-go

---

### Option 5: Google Cloud Run (PAY-AS-YOU-GO)
**Best for: Production-grade with minimal costs**

**Pros:**
- ✅ Only pay when used (per request)
- ✅ Scales automatically
- ✅ Professional infrastructure
- ✅ Free tier: 2 million requests/month

**Cons:**
- ⚠️ Requires Docker knowledge
- ⚠️ Slightly more complex setup

**Steps:**
1. Install Google Cloud CLI
2. Build Docker image:
```bash
docker build -t gcr.io/YOUR_PROJECT/sentiment-api .
docker push gcr.io/YOUR_PROJECT/sentiment-api
```

3. Deploy:
```bash
gcloud run deploy sentiment-api \
  --image gcr.io/YOUR_PROJECT/sentiment-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**Cost:** ~$0-5/month for demo usage

---

### Option 6: Heroku (PAID - No longer has free tier)
**Not recommended due to cost, but included for reference**

- Cost: Minimum $7/month
- Was popular but removed free tier in 2022

---

## 🏆 Recommended Approach

### For Portfolio Demo:

**Best Option: Hugging Face Spaces + GitHub**

**Why:**
1. ✅ **Completely FREE**
2. ✅ **Professional URL**
3. ✅ **Interactive demo**
4. ✅ **No maintenance required**
5. ✅ **Can show in interviews immediately**

**Setup Steps:**

1. **Create Simplified Gradio Interface:**
```bash
pip install gradio
```

Create `gradio_app.py`:
```python
import gradio as gr
from src.inference import SentimentInference
from src.interpretability import InterpretabilityPipeline
import matplotlib.pyplot as plt
import io
from PIL import Image

# Load models
inference = SentimentInference("./mi_modelo_entrenado")
interpret = InterpretabilityPipeline("./mi_modelo_entrenado")

def analyze_sentiment(text):
    result = inference.predict_with_probabilities(text)
    return {
        "Sentiment": result['predicted_label'],
        "Confidence": result['confidence'],
        "Probabilities": result['probability_distribution']
    }

def analyze_interpretability(text):
    # Generate attention visualization
    interpret.attention_viz.plot_attention_summary(text, save_path='attention.png')
    img = Image.open('attention.png')
    
    # Get prediction
    result = inference.predict_single(text)
    
    return img, result['predicted_label'], result['confidence']

# Create Gradio interface with tabs
with gr.Blocks(title="Transformer Sentiment Analysis") as demo:
    gr.Markdown("# 🧠 Transformer Sentiment Analysis")
    gr.Markdown("Advanced sentiment analysis using DistilBERT with interpretability features")
    
    with gr.Tab("Basic Analysis"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter text to analyze",
                    placeholder="This movie is amazing!",
                    lines=3
                )
                analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
            
            with gr.Column():
                sentiment_output = gr.Label(label="Results")
        
        analyze_btn.click(
            fn=analyze_sentiment,
            inputs=text_input,
            outputs=sentiment_output
        )
    
    with gr.Tab("Interpretability"):
        with gr.Row():
            with gr.Column():
                interp_input = gr.Textbox(
                    label="Enter text for analysis",
                    placeholder="This is incredible!",
                    lines=3
                )
                interp_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Column():
                attention_plot = gr.Image(label="Attention Visualization")
                sentiment_label = gr.Textbox(label="Predicted Sentiment")
                confidence = gr.Number(label="Confidence")
        
        interp_btn.click(
            fn=analyze_interpretability,
            inputs=interp_input,
            outputs=[attention_plot, sentiment_label, confidence]
        )
    
    gr.Markdown("""
    ## 📊 Features
    - Fine-tuned DistilBERT model
    - Attention mechanism visualization
    - Probability distributions
    - Production-ready API
    
    ## 🔗 Links
    - [GitHub Repository](your-repo-url)
    - [Full Documentation](your-docs-url)
    """)

if __name__ == "__main__":
    demo.launch()
```

2. **Upload to Hugging Face:**
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create Space
# Go to https://huggingface.co/new-space
# Choose Gradio
# Upload your files
```

3. **Create requirements.txt for Hugging Face:**
```
transformers
torch
gradio
matplotlib
seaborn
numpy
pillow
```

4. **Update your GitHub README:**
```markdown
# Transformer Sentiment Analysis

## 🎮 Try Live Demo
👉 [Interactive Demo on Hugging Face](https://huggingface.co/spaces/username/transformer-sentiment)

## 🎥 Video Demo
[Watch Full Demo](video-link)
```

---

## 💰 Cost Comparison

| Option | Cost | Uptime | Complexity | Best For |
|--------|------|--------|------------|----------|
| **Hugging Face Spaces** | FREE | Always on | ⭐ Easy | Portfolio |
| **Video Demo** | FREE | N/A | ⭐ Very Easy | Quick showcase |
| **Render.com** | FREE | Sleeps | ⭐⭐ Medium | Full app |
| **Railway.app** | $5 free/mo | Always on | ⭐⭐ Medium | Active demo |
| **Google Cloud Run** | ~$0-5/mo | On-demand | ⭐⭐⭐ Complex | Production |
| **AWS/Azure** | $10-50/mo | Always on | ⭐⭐⭐⭐ Very Complex | Enterprise |

---

## 🎯 My Recommendation

### For Professional Demo:

**1. Primary: Hugging Face Spaces**
- Free, professional, always-on
- Easy to set up
- Shows technical skills
- Can demo in interview instantly

**2. Backup: Video Demo**
- Records full functionality
- No downtime worries
- Good for LinkedIn/portfolio

**3. Code: Well-documented GitHub**
- Clean README
- Setup instructions
- Architecture diagrams
- CI/CD setup

### Complete Portfolio Package:
```
📦 Your Portfolio
├── 🎮 Live Demo (Hugging Face Spaces)
├── 🎥 Video Walkthrough (YouTube/Loom)
├── 💻 Source Code (GitHub)
├── 📖 Documentation (README + docs/)
└── 📊 Technical Blog Post (Medium/Dev.to)
```

---

## 🚀 Next Steps

1. **Create Gradio app** (use code above)
2. **Deploy to Hugging Face Spaces** (~15 minutes)
3. **Record 5-minute demo video**
4. **Update GitHub README** with links
5. **Add to LinkedIn/resume**

**Need help with setup?** I can guide you through any of these options!
