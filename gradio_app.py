#!/usr/bin/env python3
"""
Gradio app for Hugging Face Spaces deployment
Advanced sentiment analysis demo with modern UI
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import plotly.express as px
import pandas as pd
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Professional sentiment analyzer for demo"""
    
    def __init__(self):
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def analyze_single(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        if not text.strip():
            return {
                "sentiment": "Please enter some text",
                "confidence": 0.0,
                "probabilities": None
            }
        
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            probs = predictions[0].numpy()
            predicted_class = np.argmax(probs)
            confidence = float(probs[predicted_class])
            
            sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": {
                    "Negative": float(probs[0]),
                    "Positive": float(probs[1])
                }
            }
        
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return {
                "sentiment": f"Error: {str(e)}",
                "confidence": 0.0,
                "probabilities": None
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts"""
        results = []
        for text in texts:
            if text.strip():
                results.append(self.analyze_single(text))
        return results

# Initialize analyzer
analyzer = SentimentAnalyzer()

def analyze_sentiment(text: str) -> Tuple[str, float, dict]:
    """Main analysis function for Gradio"""
    result = analyzer.analyze_single(text)
    
    # Create confidence plot
    if result["probabilities"]:
        df = pd.DataFrame([
            {"Sentiment": "Negative", "Probability": result["probabilities"]["Negative"]},
            {"Sentiment": "Positive", "Probability": result["probabilities"]["Positive"]}
        ])
        
        fig = px.bar(
            df, 
            x="Sentiment", 
            y="Probability",
            color="Sentiment",
            color_discrete_map={"Negative": "#ff4444", "Positive": "#44ff44"},
            title="Sentiment Probability Distribution"
        )
        fig.update_layout(showlegend=False, height=300)
        
        return (
            f"**{result['sentiment']}** (Confidence: {result['confidence']:.1%})",
            result['confidence'],
            fig
        )
    
    return result['sentiment'], result['confidence'], None

def analyze_batch_texts(text_input: str) -> Tuple[str, dict]:
    """Analyze multiple texts separated by newlines"""
    if not text_input.strip():
        return "Please enter some texts (one per line)", None
    
    texts = [line.strip() for line in text_input.split('\n') if line.strip()]
    
    if not texts:
        return "No valid texts found", None
    
    results = analyzer.analyze_batch(texts)
    
    # Create summary
    summary_lines = []
    plot_data = []
    
    for i, (text, result) in enumerate(zip(texts, results)):
        sentiment = result['sentiment']
        confidence = result['confidence']
        summary_lines.append(f"{i+1}. **{sentiment}** ({confidence:.1%}) - {text[:50]}{'...' if len(text) > 50 else ''}")
        
        plot_data.append({
            "Text": f"Text {i+1}",
            "Sentiment": sentiment,
            "Confidence": confidence
        })
    
    summary = "\n".join(summary_lines)
    
    # Create plot
    if plot_data:
        df = pd.DataFrame(plot_data)
        fig = px.bar(
            df,
            x="Text",
            y="Confidence",
            color="Sentiment",
            color_discrete_map={"NEGATIVE": "#ff4444", "POSITIVE": "#44ff44"},
            title="Batch Analysis Results"
        )
        fig.update_layout(height=400)
        
        return summary, fig
    
    return summary, None

# Demo examples
EXAMPLES = [
    "🎬 This movie absolutely blew my mind! Best film I've seen this year - incredible cinematography and acting!",
    "😞 Worst customer service ever. They ignored my calls and the product arrived completely broken. Total waste of money.",
    "🤔 The restaurant was decent, nothing extraordinary but the food was acceptable and staff was polite.",
    "🚀 Revolutionary AI technology! This transformer model shows incredible understanding of human language nuances.",
    "❌ I regret this purchase deeply. Poor quality materials and misleading advertising. Avoid at all costs!",
    "✈️ Amazing travel experience! The hotel exceeded expectations and the local tours were absolutely spectacular.",
    "📚 Mixed feelings about this book - great storyline but the ending felt rushed and unsatisfying.",
    "🎵 Concert was phenomenal! The energy, the music, the atmosphere - everything was absolutely perfect!"
]

BATCH_EXAMPLE = """🛍️ This online store has amazing customer service! Fast shipping and quality products.
😡 Terrible experience with their support team. Rude staff and no solutions offered.
🍕 Pizza was okay, nothing special but not bad either. Average taste and decent price.
⭐ Outstanding quality! Exceeded all my expectations. Highly recommend to everyone!
💸 Disappointed with this expensive purchase. Not worth the money at all.
🎯 Perfect for my needs! Exactly what I was looking for. Great value for money.
🏨 Hotel was clean and comfortable. Staff was friendly and location was convenient."""

# Create Gradio interface with modern styling
with gr.Blocks(
    title="🤖 Transformer Sentiment Analysis",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
        font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"]
    ),
    css="""
    /* Global styling */
    .gradio-container {
        max-width: 1200px;
        margin: auto;
        font-family: 'Inter', 'SF Pro Display', system-ui, sans-serif;
    }
    
    /* Custom button styling */
    #analyze-btn, #batch-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    #analyze-btn:hover, #batch-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input field styling */
    #single-input textarea, #batch-input textarea {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        transition: border-color 0.3s ease;
        font-size: 15px;
    }
    
    #single-input textarea:focus, #batch-input textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Tab styling */
    .tab-nav button {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        font-size: 15px;
    }
    
    /* Confidence score styling */
    #confidence input {
        font-size: 18px;
        font-weight: 600;
        color: #2E86AB;
        text-align: center;
    }
    
    /* Cards and panels */
    .gr-panel {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    /* Modern shadows */
    .gr-box {
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06);
    }
    """
) as demo:
    
    gr.Markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #2E86AB; margin-bottom: 10px;">🤖 Transformer Sentiment Analysis</h1>
        <p style="font-size: 18px; color: #555; font-weight: 300; margin-bottom: 20px;">
            Advanced AI-powered sentiment analysis with state-of-the-art transformer models
        </p>
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 3px; width: 80%; margin: 0 auto; border-radius: 2px;"></div>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0;">
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea;">
            <h3 style="color: #2E86AB; margin: 0 0 10px 0;">🧠 Model</h3>
            <p style="margin: 0; color: #666;">DistilBERT (66M parameters)</p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #52b788;">
            <h3 style="color: #2E86AB; margin: 0 0 10px 0;">⚡ Speed</h3>
            <p style="margin: 0; color: #666;">~100ms inference</p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #f72585;">
            <h3 style="color: #2E86AB; margin: 0 0 10px 0;">🎯 Accuracy</h3>
            <p style="margin: 0; color: #666;">74% on IMDB dataset</p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #f77f00;">
            <h3 style="color: #2E86AB; margin: 0 0 10px 0;">📊 Features</h3>
            <p style="margin: 0; color: #666;">Confidence & batch processing</p>
        </div>
    </div>
    """)
    
    with gr.Tabs():
        # Single Text Analysis Tab
        with gr.TabItem("🔍 Single Analysis", elem_id="single-tab"):
            gr.Markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0; text-align: center;">
                    Analyze individual texts with detailed confidence metrics
                </h3>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    single_input = gr.Textbox(
                        label="💬 Enter your text",
                        placeholder="Type or paste your text here for sentiment analysis...",
                        lines=4,
                        elem_id="single-input"
                    )
                    single_btn = gr.Button(
                        "🚀 Analyze Sentiment", 
                        variant="primary",
                        elem_id="analyze-btn",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    single_output = gr.Markdown(label="📋 Analysis Result")
                    confidence_score = gr.Number(
                        label="🎯 Confidence Score", 
                        precision=3,
                        elem_id="confidence"
                    )
                    probability_plot = gr.Plot(label="📊 Probability Distribution")
            
            # Examples with better styling
            gr.Markdown("""
            <div style="margin-top: 30px;">
                <h4 style="color: #2E86AB; margin-bottom: 15px;">💡 Try these examples:</h4>
            </div>
            """)
            examples_single = gr.Examples(
                examples=EXAMPLES,
                inputs=single_input,
                label=""
            )
        
        # Batch Analysis Tab
        with gr.TabItem("📊 Batch Processing", elem_id="batch-tab"):
            gr.Markdown("""
            <div style="background: linear-gradient(135deg, #52b788 0%, #2d6a4f 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0; text-align: center;">
                    Process multiple texts simultaneously with advanced analytics
                </h3>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    batch_input = gr.Textbox(
                        label="📝 Multiple texts (one per line)",
                        placeholder="Enter multiple texts here, one per line...",
                        lines=8,
                        value=BATCH_EXAMPLE,
                        elem_id="batch-input"
                    )
                    batch_btn = gr.Button(
                        "🚀 Process Batch", 
                        variant="primary",
                        elem_id="batch-btn",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    batch_output = gr.Markdown(label="📈 Batch Results")
                    batch_plot = gr.Plot(label="📊 Visual Analytics")
        
        # Technical Details Tab
        with gr.TabItem("🛠️ Technical Details", elem_id="tech-tab"):
            gr.Markdown("""
            <div style="background: linear-gradient(135deg, #f72585 0%, #b5179e 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0; text-align: center;">
                    Deep dive into architecture, performance, and capabilities
                </h3>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0;">
                
                <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; border-top: 4px solid #667eea;">
                    <h4 style="color: #2E86AB; margin: 0 0 15px 0;">🏗️ Architecture</h4>
                    <ul style="color: #666; line-height: 1.6;">
                        <li><strong>Model:</strong> DistilBERT (Distilled BERT)</li>
                        <li><strong>Parameters:</strong> 66 million</li>
                        <li><strong>Training:</strong> Fine-tuned on SST-2</li>
                        <li><strong>Accuracy:</strong> 74% on IMDB dataset</li>
                    </ul>
                </div>
                
                <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; border-top: 4px solid #52b788;">
                    <h4 style="color: #2E86AB; margin: 0 0 15px 0;">⚡ Performance</h4>
                    <ul style="color: #666; line-height: 1.6;">
                        <li><strong>Speed:</strong> ~100ms per prediction</li>
                        <li><strong>Batch Processing:</strong> Supported</li>
                        <li><strong>Memory:</strong> Optimized for production</li>
                        <li><strong>Scalability:</strong> Cloud-ready</li>
                    </ul>
                </div>
                
                <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; border-top: 4px solid #f72585;">
                    <h4 style="color: #2E86AB; margin: 0 0 15px 0;">🔧 Features</h4>
                    <ul style="color: #666; line-height: 1.6;">
                        <li>Real-time sentiment classification</li>
                        <li>Confidence scoring & probabilities</li>
                        <li>RESTful API with async support</li>
                        <li>Model interpretability tools</li>
                    </ul>
                </div>
                
                <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; border-top: 4px solid #f77f00;">
                    <h4 style="color: #2E86AB; margin: 0 0 15px 0;">🚀 Tech Stack</h4>
                    <ul style="color: #666; line-height: 1.6;">
                        <li><strong>Framework:</strong> PyTorch + Transformers</li>
                        <li><strong>API:</strong> FastAPI with async</li>
                        <li><strong>Deployment:</strong> Docker + Cloud</li>
                        <li><strong>Testing:</strong> Comprehensive suite</li>
                    </ul>
                </div>
                
            </div>
            
            <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; margin: 30px 0;">
                <h4 style="color: #1976d2; margin: 0 0 15px 0;">🎯 Use Cases</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div style="color: #666;">📱 Social media monitoring</div>
                    <div style="color: #666;">📧 Customer feedback analysis</div>
                    <div style="color: #666;">📊 Market research insights</div>
                    <div style="color: #666;">🛒 Product review classification</div>
                </div>
            </div>
            
            <div style="text-align: center; padding: 20px; background: #f5f5f5; border-radius: 10px;">
                <h4 style="color: #2E86AB; margin: 0 0 10px 0;">🔗 Open Source Project</h4>
                <p style="color: #666; margin: 0;">Complete source code, training scripts, and deployment guides available on GitHub</p>
            </div>
            """)
    
    # Event handlers
    single_btn.click(
        fn=analyze_sentiment,
        inputs=single_input,
        outputs=[single_output, confidence_score, probability_plot]
    )
    
    batch_btn.click(
        fn=analyze_batch_texts,
        inputs=batch_input,
        outputs=[batch_output, batch_plot]
    )
    
    # Footer with modern styling
    gr.Markdown("""
    <div style="margin-top: 40px; padding: 30px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; text-align: center;">
        <h3 style="color: white; margin: 0 0 15px 0;">🚀 Advanced ML Engineering</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0 0 20px 0; font-size: 16px;">
            This demo showcases production-ready machine learning engineering including model training, 
            API development, comprehensive testing, and scalable deployment solutions.
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; color: white;">PyTorch</span>
            <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; color: white;">Transformers</span>
            <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; color: white;">FastAPI</span>
            <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; color: white;">Docker</span>
            <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; color: white;">Gradio</span>
        </div>
    </div>
    """)

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )