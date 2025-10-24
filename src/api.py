"""Production-ready FastAPI server for sentiment analysis."""

import os
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import json

from src.inference import SentimentInference
from src.data_utils import load_config
from src.interpretability import InterpretabilityPipeline, AttentionVisualizer
import base64
import io


# Global model instance
inference_pipeline: Optional[SentimentInference] = None
interpretability_pipeline: Optional[InterpretabilityPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup."""
    global inference_pipeline, interpretability_pipeline
    
    # Load configuration
    config = load_config()
    
    # Determine model path
    model_path = os.environ.get("MODEL_PATH", "./results")
    if not os.path.exists(model_path):
        model_path = config["model"]["name"]  # Fall back to base model
    
    print(f"🚀 Loading model: {model_path}")
    
    # Initialize inference pipeline
    inference_pipeline = SentimentInference(
        model_path=model_path,
        batch_size=config["api"]["max_batch_size"]
    )
    
    # Initialize interpretability pipeline
    try:
        interpretability_pipeline = InterpretabilityPipeline(model_path)
        print("🔍 Interpretability pipeline loaded!")
    except Exception as e:
        print(f"⚠️  Could not load interpretability pipeline: {e}")
        interpretability_pipeline = None
    
    print("✅ Model loaded successfully!")
    yield
    
    # Cleanup
    print("🧹 Shutting down...")


app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready sentiment analysis using Transformer models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class TextInput(BaseModel):
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=10000)


class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    model_path: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int


class ProbabilityResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    probability_distribution: Dict[str, float]
    model_path: str


class ModelInfo(BaseModel):
    model_path: str
    device: str
    total_parameters: int
    trainable_parameters: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class InterpretabilityResponse(BaseModel):
    text: str
    predicted_class: int
    confidence: float
    attention_summary_plot: str  # base64 encoded image
    attention_heatmap_plot: str  # base64 encoded image
    shap_explanation: Optional[str] = None  # base64 encoded image if available


class AttentionWeightsResponse(BaseModel):
    text: str
    tokens: List[str]
    attention_weights: List[List[List[List[float]]]]  # [layer][head][seq][seq]
    predicted_class: int
    confidence: float


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global inference_pipeline
    
    return HealthResponse(
        status="healthy" if inference_pipeline is not None else "unhealthy",
        model_loaded=inference_pipeline is not None,
        device=inference_pipeline.device if inference_pipeline else "unknown"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(input_data: TextInput):
    """Predict sentiment for a single text."""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = inference_pipeline.predict_single(input_data.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_sentiment(input_data: BatchTextInput):
    """Predict sentiment for multiple texts."""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = inference_pipeline.predict_batch(input_data.texts)
        predictions = [PredictionResponse(**result) for result in results]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/predict/probabilities", response_model=ProbabilityResponse)
async def predict_with_probabilities(input_data: TextInput):
    """Predict sentiment with full probability distribution."""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = inference_pipeline.predict_with_probabilities(input_data.text)
        return ProbabilityResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Probability prediction failed: {str(e)}")


@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    """Predict sentiment for texts in uploaded file (one text per line)."""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith(('.txt', '.csv')):
        raise HTTPException(status_code=400, detail="Only .txt and .csv files are supported")
    
    try:
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Split by lines and filter empty lines
        texts = [line.strip() for line in text_content.split('\n') if line.strip()]
        
        if len(texts) > 1000:
            raise HTTPException(status_code=400, detail="File contains too many texts (max 1000)")
        
        results = inference_pipeline.predict_batch(texts)
        predictions = [PredictionResponse(**result) for result in results]
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions)
        )
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported (use UTF-8)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        summary = inference_pipeline.get_model_summary()
        return ModelInfo(
            model_path=summary["model_path"],
            device=summary["device"],
            total_parameters=summary["total_parameters"],
            trainable_parameters=summary["trainable_parameters"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.post("/model/benchmark")
async def benchmark_model(input_data: BatchTextInput, background_tasks: BackgroundTasks):
    """Benchmark model performance."""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        benchmark_result = inference_pipeline.benchmark_inference(input_data.texts)
        return benchmark_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@app.get("/model/attention")
async def get_attention_weights(text: str):
    """Get attention weights for interpretability (for debugging/research)."""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = inference_pipeline.get_attention_weights(text)
        # Convert numpy arrays to lists for JSON serialization
        result["attention_weights"] = [layer.tolist() for layer in result["attention_weights"]]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attention extraction failed: {str(e)}")


@app.post("/interpret", response_model=InterpretabilityResponse)
async def interpret_text(input_data: TextInput):
    """Provide full interpretability analysis for a text."""
    global interpretability_pipeline
    
    if interpretability_pipeline is None:
        raise HTTPException(status_code=503, detail="Interpretability pipeline not available")
    
    try:
        import matplotlib.pyplot as plt
        import tempfile
        import os
        
        # Create temporary directory for plots
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run analysis
            report = interpretability_pipeline.full_analysis(input_data.text, temp_dir)
            
            # Read and encode plots as base64
            def encode_plot(filename):
                plot_path = os.path.join(temp_dir, filename)
                if os.path.exists(plot_path):
                    with open(plot_path, 'rb') as f:
                        plot_data = f.read()
                    return base64.b64encode(plot_data).decode('utf-8')
                return ""
            
            attention_summary = encode_plot("attention_summary.png")
            attention_heatmap = encode_plot("attention_heatmap.png")
            shap_explanation = encode_plot("shap_explanation.png") if os.path.exists(os.path.join(temp_dir, "shap_explanation.png")) else None
            
            return InterpretabilityResponse(
                text=input_data.text,
                predicted_class=report["predicted_class"],
                confidence=report["confidence"],
                attention_summary_plot=attention_summary,
                attention_heatmap_plot=attention_heatmap,
                shap_explanation=shap_explanation
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interpretability analysis failed: {str(e)}")


@app.post("/interpret/attention", response_model=AttentionWeightsResponse)
async def get_detailed_attention(input_data: TextInput):
    """Get detailed attention weights for visualization."""
    global interpretability_pipeline
    
    if interpretability_pipeline is None:
        raise HTTPException(status_code=503, detail="Interpretability pipeline not available")
    
    try:
        # Get attention weights
        attention_data = interpretability_pipeline.attention_viz.get_attention_weights(input_data.text)
        
        # Get prediction
        import torch
        inputs = interpretability_pipeline.tokenizer(input_data.text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = interpretability_pipeline.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0, predicted_class].item()
        
        # Convert attention weights to lists for JSON serialization
        attention_weights_list = [layer.tolist() for layer in attention_data["attention_weights"]]
        
        return AttentionWeightsResponse(
            text=input_data.text,
            tokens=attention_data["tokens"],
            attention_weights=attention_weights_list,
            predicted_class=predicted_class,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attention analysis failed: {str(e)}")


def create_app(model_path: Optional[str] = None) -> FastAPI:
    """Factory function to create FastAPI app with custom model path."""
    if model_path:
        os.environ["MODEL_PATH"] = model_path
    return app


def main():
    """Run the FastAPI server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run sentiment analysis API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", type=str, help="Path to model")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Set model path if provided
    if args.model:
        os.environ["MODEL_PATH"] = args.model
    
    # Run server
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()