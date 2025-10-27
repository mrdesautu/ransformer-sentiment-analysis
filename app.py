#!/usr/bin/env python3
"""
Sentiment Analysis App for HuggingFace Spaces
Clean, robust implementation with proper visualization
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SENTIMENT ANALYZER CLASS
# ============================================================================

class SentimentAnalyzer:
    """Production-ready sentiment analyzer"""
    
    def __init__(self, model_name: str = "MartinRodrigo/distilbert-sentiment-imdb"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load model from HuggingFace Hub"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✓ Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"✗ Error loading model: {e}")
            logger.info("Falling back to base DistilBERT model...")
            self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
    
    def predict(self, text: str) -> Dict:
        """Predict sentiment for a single text"""
        if not text or not text.strip():
            return {
                "sentiment": "ERROR",
                "confidence": 0.0,
                "probabilities": {"Negative": 0.5, "Positive": 0.5},
                "error": "Please enter some text"
            }
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Extract results
            probs_cpu = probs.cpu().numpy()[0]
            predicted_class = int(np.argmax(probs_cpu))
            confidence = float(probs_cpu[predicted_class])
            
            sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": {
                    "Negative": float(probs_cpu[0]),
                    "Positive": float(probs_cpu[1])
                },
                "error": None
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "sentiment": "ERROR",
                "confidence": 0.0,
                "probabilities": {"Negative": 0.5, "Positive": 0.5},
                "error": str(e)
            }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for multiple texts"""
        return [self.predict(text) for text in texts if text.strip()]

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Create a bar chart for sentiment probabilities"""
    fig = go.Figure()
    
    sentiments = list(probabilities.keys())
    values = list(probabilities.values())
    colors = ["#ff4444" if s == "Negative" else "#44ff44" for s in sentiments]
    
    fig.add_trace(go.Bar(
        x=sentiments,
        y=values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition='outside',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Sentiment Probability Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'white'}
        },
        xaxis_title="Sentiment",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=350,
        template="plotly_dark",
        showlegend=False,
        margin=dict(t=60, b=60, l=60, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_batch_chart(results: List[Dict], texts: List[str]) -> go.Figure:
    """Create a bar chart for batch analysis results"""
    fig = go.Figure()
    
    text_labels = [f"Text {i+1}" for i in range(len(results))]
    confidences = [r['confidence'] for r in results]
    sentiments = [r['sentiment'] for r in results]
    colors = ["#ff4444" if s == "NEGATIVE" else "#44ff44" for s in sentiments]
    
    # Truncate texts for hover
    hover_texts = [t[:100] + "..." if len(t) > 100 else t for t in texts]
    
    fig.add_trace(go.Bar(
        x=text_labels,
        y=confidences,
        marker_color=colors,
        text=[f"{c:.1%}" for c in confidences],
        textposition='outside',
        textfont=dict(size=12, color='white'),
        customdata=list(zip(sentiments, hover_texts)),
        hovertemplate='<b>%{x}</b><br>' +
                      'Sentiment: %{customdata[0]}<br>' +
                      'Confidence: %{y:.1%}<br>' +
                      'Text: %{customdata[1]}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Batch Analysis Results",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'white'}
        },
        xaxis_title="Text Number",
        yaxis_title="Confidence",
        yaxis_range=[0, 1.1],
        height=400,
        template="plotly_dark",
        showlegend=False,
        margin=dict(t=60, b=80, l=60, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

def analyze_text(text: str) -> Tuple[str, str, go.Figure]:
    """Analyze single text and return formatted results"""
    result = analyzer.predict(text)
    
    if result['error']:
        return (
            f"⚠️ **Error:** {result['error']}",
            "0.0",
            create_probability_chart({"Negative": 0.5, "Positive": 0.5})
        )
    
    # Format output
    emoji = "😊" if result['sentiment'] == "POSITIVE" else "😞"
    sentiment_text = f"{emoji} **{result['sentiment']}**"
    confidence_text = f"{result['confidence']:.1%}"
    
    # Create chart
    chart = create_probability_chart(result['probabilities'])
    
    return sentiment_text, confidence_text, chart

def analyze_batch(text_input: str) -> Tuple[str, go.Figure]:
    """Analyze multiple texts"""
    if not text_input or not text_input.strip():
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[{
                'text': "Please enter texts to analyze",
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': {'size': 16, 'color': 'gray'}
            }]
        )
        return "Please enter texts (one per line)", empty_fig
    
    # Split texts
    texts = [line.strip() for line in text_input.split('\n') if line.strip()]
    
    if not texts:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[{
                'text': "No valid texts found",
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': {'size': 16, 'color': 'gray'}
            }]
        )
        return "No valid texts found", empty_fig
    
    # Analyze all texts
    results = analyzer.predict_batch(texts)
    
    # Create summary
    summary_lines = ["### 📊 Analysis Results\n"]
    for i, (text, result) in enumerate(zip(texts, results), 1):
        emoji = "😊" if result['sentiment'] == "POSITIVE" else "😞"
        text_preview = text[:60] + "..." if len(text) > 60 else text
        summary_lines.append(
            f"{i}. {emoji} **{result['sentiment']}** ({result['confidence']:.1%}) - *{text_preview}*"
        )
    
    summary = "\n".join(summary_lines)
    
    # Create chart
    chart = create_batch_chart(results, texts)
    
    return summary, chart

# ============================================================================
# INITIALIZE ANALYZER
# ============================================================================

logger.info("Initializing sentiment analyzer...")
analyzer = SentimentAnalyzer()

# ============================================================================
# EXAMPLES
# ============================================================================

SINGLE_EXAMPLES = [
    ["This movie is absolutely fantastic! I loved every minute of it."],
    ["Terrible experience. Waste of time and money."],
    ["The product is okay, nothing special but it works."],
    ["Best purchase ever! Highly recommend to everyone!"],
    ["Disappointed with the quality. Expected much better."]
]

BATCH_EXAMPLE = """This movie is absolutely fantastic! I loved every minute of it.
Terrible experience. Waste of time and money.
The product is okay, nothing special but it works.
Best purchase ever! Highly recommend to everyone!"""

# ============================================================================
# GRADIO UI
# ============================================================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="green",
    ),
    css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        #component-0 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
    """
) as demo:
    
    # Header
    gr.Markdown("""
    # 🎭 Sentiment Analysis with DistilBERT
    
    Analyze text sentiment with a fine-tuned transformer model trained on IMDB reviews.
    
    **Model:** DistilBERT | **Accuracy:** 80% | **F1:** 0.7981
    """)
    
    # Main interface
    with gr.Tabs():
        # TAB 1: Single Analysis
        with gr.TabItem("🔍 Single Analysis"):
            gr.Markdown("### Analyze individual texts")
            
            with gr.Row():
                with gr.Column(scale=1):
                    single_input = gr.Textbox(
                        label="Enter your text",
                        placeholder="Type or paste your text here...",
                        lines=6,
                        max_lines=10
                    )
                    single_btn = gr.Button(
                        "🚀 Analyze",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Examples(
                        examples=SINGLE_EXAMPLES,
                        inputs=single_input,
                        label="Try these examples:"
                    )
                
                with gr.Column(scale=1):
                    single_sentiment = gr.Markdown(
                        label="Result",
                        value="*Results will appear here*"
                    )
                    single_confidence = gr.Textbox(
                        label="Confidence Score",
                        interactive=False
                    )
                    single_plot = gr.Plot(label="Probability Distribution")
        
        # TAB 2: Batch Processing
        with gr.TabItem("📊 Batch Processing"):
            gr.Markdown("### Process multiple texts at once (one per line)")
            
            with gr.Row():
                with gr.Column(scale=1):
                    batch_input = gr.Textbox(
                        label="Enter multiple texts (one per line)",
                        placeholder="Enter texts, one per line...",
                        lines=10,
                        value=BATCH_EXAMPLE
                    )
                    batch_btn = gr.Button(
                        "🚀 Process Batch",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    batch_results = gr.Markdown(
                        label="Results Summary",
                        value="*Results will appear here*"
                    )
                    batch_plot = gr.Plot(label="Batch Analytics")
        
        # TAB 3: About
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About This Model
            
            ### 🏗️ Architecture
            - **Base Model:** DistilBERT (Distilled BERT)
            - **Parameters:** 66 million
            - **Training Data:** IMDB Movie Reviews (50k reviews)
            - **Fine-tuning:** Binary sentiment classification
            
            ### 📊 Performance Metrics
            - **Test Accuracy:** 80.0%
            - **F1 Score:** 0.7981
            - **Precision:** High
            - **Recall:** Balanced
            
            ### ⚡ Features
            - Fast inference (~100ms per prediction)
            - Batch processing support
            - Interactive visualizations
            - Production-ready deployment
            
            ### 🔗 Resources
            - **Model Repository:** [MartinRodrigo/distilbert-sentiment-imdb](https://huggingface.co/MartinRodrigo/distilbert-sentiment-imdb)
            - **Space:** [transformer-sentiment-analysis](https://huggingface.co/spaces/MartinRodrigo/transformer-sentiment-analysis)
            - **GitHub:** [ransformer-sentiment-analysis](https://github.com/mrdesautu/ransformer-sentiment-analysis)
            
            ### 🛠️ Tech Stack
            - **Framework:** PyTorch + Transformers
            - **UI:** Gradio
            - **Visualization:** Plotly
            - **Tracking:** MLflow (local development)
            
            ---
            
            Built with ❤️ by Martin Rodrigo
            """)
    
    # Connect event handlers
    single_btn.click(
        fn=analyze_text,
        inputs=[single_input],
        outputs=[single_sentiment, single_confidence, single_plot]
    )
    
    batch_btn.click(
        fn=analyze_batch,
        inputs=[batch_input],
        outputs=[batch_results, batch_plot]
    )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True
    )
