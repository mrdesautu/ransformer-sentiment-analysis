#!/usr/bin/env python3
"""
Gradio app for Hugging Face Spaces deployment
Advanced sentiment analysis demo with modern UI and MLflow integration
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Tuple
import logging
import mlflow
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Sentiment analyzer for demo"""
    
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

def get_mlflow_metrics() -> Tuple[pd.DataFrame, str]:
    """Get MLflow experiment metrics and summary"""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("./mlruns")
        
        # Get all experiments
        experiments = mlflow.search_experiments()
        experiment_names = ['production-training', 'sentiment-analysis-training']
        
        # Search for runs
        runs = mlflow.search_runs(
            experiment_names=experiment_names,
            order_by=["metrics.test_eval_accuracy DESC"]
        )
        
        if len(runs) == 0:
            return pd.DataFrame(), "No MLflow runs found"
        
        # Filter runs with test metrics
        runs_with_test = runs[runs['metrics.test_eval_accuracy'].notna()].copy()
        
        if len(runs_with_test) == 0:
            return pd.DataFrame(), "No completed runs with test metrics"
        
        # Get top 5 runs
        top_runs = runs_with_test.head(5)
        
        # Create summary
        summary_lines = [
            f"**📊 MLflow Experiment Summary**\n",
            f"Total runs: {len(runs)}",
            f"Completed runs: {len(runs_with_test)}\n",
            f"**🏆 Top Model:**",
            f"- Accuracy: {top_runs.iloc[0]['metrics.test_eval_accuracy']:.2%}",
            f"- F1 Score: {top_runs.iloc[0]['metrics.test_eval_f1']:.4f}",
            f"- Loss: {top_runs.iloc[0]['metrics.test_eval_loss']:.4f}",
        ]
        
        summary = "\n".join(summary_lines)
        
        return top_runs, summary
        
    except Exception as e:
        logger.error(f"Error getting MLflow metrics: {e}")
        return pd.DataFrame(), f"Error loading MLflow data: {str(e)}"

def create_metrics_plot() -> go.Figure:
    """Create a comprehensive metrics visualization"""
    try:
        top_runs, _ = get_mlflow_metrics()
        
        if top_runs.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No MLflow data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Prepare data for plotting
        plot_data = []
        for idx, row in top_runs.iterrows():
            run_name = row.get('tags.mlflow.runName', f'Run {idx}')
            plot_data.append({
                'Run': run_name[:20],  # Truncate long names
                'Accuracy': row.get('metrics.test_eval_accuracy', 0) * 100,
                'F1 Score': row.get('metrics.test_eval_f1', 0) * 100,
                'Loss': row.get('metrics.test_eval_loss', 0)
            })
        
        df = pd.DataFrame(plot_data)
        
        # Create subplots
        fig = go.Figure()
        
        # Add accuracy bars
        fig.add_trace(go.Bar(
            name='Accuracy (%)',
            x=df['Run'],
            y=df['Accuracy'],
            marker_color='#667eea',
            text=df['Accuracy'].round(1),
            textposition='outside',
            texttemplate='%{text}%'
        ))
        
        # Add F1 score bars
        fig.add_trace(go.Bar(
            name='F1 Score (%)',
            x=df['Run'],
            y=df['F1 Score'],
            marker_color='#52b788',
            text=df['F1 Score'].round(1),
            textposition='outside',
            texttemplate='%{text}%'
        ))
        
        fig.update_layout(
            title='Model Performance Comparison (Top 5 Runs)',
            xaxis_title='Model Run',
            yaxis_title='Score (%)',
            barmode='group',
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating metrics plot: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_loss_plot() -> go.Figure:
    """Create loss comparison plot"""
    try:
        top_runs, _ = get_mlflow_metrics()
        
        if top_runs.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No MLflow data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Prepare data
        plot_data = []
        for idx, row in top_runs.iterrows():
            run_name = row.get('tags.mlflow.runName', f'Run {idx}')
            plot_data.append({
                'Run': run_name[:20],
                'Test Loss': row.get('metrics.test_eval_loss', 0),
                'Eval Loss': row.get('metrics.eval_loss', 0)
            })
        
        df = pd.DataFrame(plot_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Run'],
            y=df['Test Loss'],
            mode='lines+markers',
            name='Test Loss',
            line=dict(color='#f72585', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Run'],
            y=df['Eval Loss'],
            mode='lines+markers',
            name='Eval Loss',
            line=dict(color='#f77f00', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title='Loss Comparison Across Runs',
            xaxis_title='Model Run',
            yaxis_title='Loss',
            height=400,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating loss plot: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def get_experiment_details() -> str:
    """Get detailed experiment information"""
    try:
        mlflow.set_tracking_uri("./mlruns")
        runs = mlflow.search_runs(
            experiment_names=['production-training', 'sentiment-analysis-training']
        )
        
        if len(runs) == 0:
            return "**No experiments found**\n\nStart training to see results here!"
        
        runs_with_test = runs[runs['metrics.test_eval_accuracy'].notna()]
        
        details = [
            "# 📊 Experiment Details\n",
            f"**Total Experiments:** {len(runs)}",
            f"**Completed Runs:** {len(runs_with_test)}",
            f"**Best Accuracy:** {runs_with_test['metrics.test_eval_accuracy'].max():.2%}" if len(runs_with_test) > 0 else "N/A",
            f"**Average Accuracy:** {runs_with_test['metrics.test_eval_accuracy'].mean():.2%}" if len(runs_with_test) > 0 else "N/A",
            "\n## 🏆 Top 3 Models\n"
        ]
        
        for idx, row in runs_with_test.head(3).iterrows():
            run_name = row.get('tags.mlflow.runName', 'unnamed')
            accuracy = row.get('metrics.test_eval_accuracy', 0)
            f1 = row.get('metrics.test_eval_f1', 0)
            loss = row.get('metrics.test_eval_loss', 0)
            lr = row.get('params.learning_rate', 'N/A')
            
            details.append(f"### {idx+1}. {run_name}")
            details.append(f"- **Accuracy:** {accuracy:.2%}")
            details.append(f"- **F1 Score:** {f1:.4f}")
            details.append(f"- **Loss:** {loss:.4f}")
            details.append(f"- **Learning Rate:** {lr}\n")
        
        return "\n".join(details)
        
    except Exception as e:
        return f"**Error loading experiment details:** {str(e)}"


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
        neutral_hue="slate"
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
        
        # MLflow Metrics Tab (NEW!)
        with gr.TabItem("📈 MLflow Metrics", elem_id="mlflow-tab"):
            gr.Markdown("""
            <div style="background: linear-gradient(135deg, #f77f00 0%, #d62828 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0; text-align: center;">
                    Track and compare model training experiments
                </h3>
            </div>
            """)
            
            with gr.Row():
                mlflow_refresh_btn = gr.Button(
                    "🔄 Refresh Metrics",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Row():
                with gr.Column(scale=1):
                    mlflow_summary = gr.Markdown(
                        value=get_mlflow_metrics()[1],
                        label="📊 Summary"
                    )
                    mlflow_details = gr.Markdown(
                        value=get_experiment_details(),
                        label="📋 Experiment Details"
                    )
                
                with gr.Column(scale=2):
                    metrics_plot = gr.Plot(
                        value=create_metrics_plot(),
                        label="📊 Performance Metrics"
                    )
                    loss_plot = gr.Plot(
                        value=create_loss_plot(),
                        label="📉 Loss Comparison"
                    )
        
        # Technical Details Tab
        with gr.TabItem("🛠️ Technical Details", elem_id="tech-tab"):
            gr.Markdown("## Deep dive into architecture, performance, and capabilities\n")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
### 🏗️ Architecture
- **Model:** DistilBERT (Distilled BERT)
- **Parameters:** 66 million
- **Training:** Fine-tuned on SST-2
- **Accuracy:** 80% on IMDB dataset
""")
                
                with gr.Column():
                    gr.Markdown("""
### ⚡ Performance
- **Speed:** ~100ms per prediction
- **Batch Processing:** Supported
- **Memory:** Optimized for production
- **Scalability:** Cloud-ready
""")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
### 🔧 Features
- Real-time sentiment classification
- Confidence scoring & probabilities
- RESTful API with async support
- Model interpretability tools
- MLflow experiment tracking
""")
                
                with gr.Column():
                    gr.Markdown("""
### 🚀 Tech Stack
- **Framework:** PyTorch + Transformers
- **API:** FastAPI with async
- **Tracking:** MLflow experiments
- **Testing:** Comprehensive suite
""")
            
            gr.Markdown("---")
            
            gr.Markdown("""
### 🎯 Use Cases
- 📱 Social media monitoring
- 📧 Customer feedback analysis
- 📊 Market research insights
- 🛒 Product review classification
""")
            
            gr.Markdown("---")
            
            gr.Markdown("""
### 🔗 Open Source Project
Complete source code, training scripts, and deployment guides available on GitHub
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
    
    # MLflow refresh handler
    def refresh_mlflow():
        _, summary = get_mlflow_metrics()
        details = get_experiment_details()
        metrics = create_metrics_plot()
        loss = create_loss_plot()
        return summary, details, metrics, loss
    
    mlflow_refresh_btn.click(
        fn=refresh_mlflow,
        inputs=[],
        outputs=[mlflow_summary, mlflow_details, metrics_plot, loss_plot]
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
            <span style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; color: white;">MLflow</span>
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