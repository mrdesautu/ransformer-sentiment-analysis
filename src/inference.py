"""Advanced inference pipeline with batch processing and model switching."""

import json
import os
from typing import List, Dict, Any, Optional, Union
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline
)
from src.data_utils import load_config


class SentimentInference:
    """Advanced sentiment analysis inference pipeline."""
    
    def __init__(
        self, 
        model_path: str, 
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model or model name
            device: Device to run inference on (auto-detect if None)
            batch_size: Batch size for batch inference
        """
        self.model_path = model_path
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🚀 Loading model from: {model_path}")
        print(f"🔧 Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load model info if available
        self.model_info = self._load_model_info()
        
        # Create pipeline for easy inference
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            batch_size=self.batch_size
        )
        
        print("✅ Model loaded successfully!")
    
    def _load_model_info(self) -> Optional[Dict[str, Any]]:
        """Load model information if available."""
        info_path = os.path.join(self.model_path, "model_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                return json.load(f)
        return None
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        result = self.pipeline(text)[0]
        
        return {
            "text": text,
            "predicted_label": result["label"],
            "confidence": result["score"],
            "model_path": self.model_path
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        results = self.pipeline(texts)
        
        predictions = []
        for text, result in zip(texts, results):
            predictions.append({
                "text": text,
                "predicted_label": result["label"],
                "confidence": result["score"],
                "model_path": self.model_path
            })
        
        return predictions
    
    def predict_with_probabilities(self, text: str) -> Dict[str, Any]:
        """
        Predict with full probability distribution.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with full probability distribution
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]
        
        # Get label mapping
        id2label = self.model.config.id2label
        
        # Create probability distribution
        prob_dist = {}
        for label_id, prob in enumerate(probabilities):
            label = id2label.get(label_id, f"LABEL_{label_id}")
            prob_dist[label] = float(prob)
        
        # Get predicted label
        predicted_id = np.argmax(probabilities)
        predicted_label = id2label.get(predicted_id, f"LABEL_{predicted_id}")
        
        return {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": float(probabilities[predicted_id]),
            "probability_distribution": prob_dist,
            "model_path": self.model_path
        }
    
    def get_attention_weights(self, text: str) -> Dict[str, Any]:
        """
        Get attention weights for interpretability.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with attention weights and tokens
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions
        
        # Convert to numpy and get tokens
        attention_weights = [att.cpu().numpy() for att in attentions]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return {
            "text": text,
            "tokens": tokens,
            "attention_weights": attention_weights,
            "num_layers": len(attention_weights),
            "num_heads": attention_weights[0].shape[1]
        }
    
    def benchmark_inference(self, texts: List[str], num_runs: int = 5) -> Dict[str, Any]:
        """
        Benchmark inference performance.
        
        Args:
            texts: List of texts to benchmark
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        times = []
        
        # Warm up
        self.predict_batch(texts[:min(5, len(texts))])
        
        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            self.predict_batch(texts)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = len(texts) / avg_time
        
        return {
            "num_texts": len(texts),
            "num_runs": num_runs,
            "avg_time_seconds": avg_time,
            "std_time_seconds": std_time,
            "throughput_texts_per_second": throughput,
            "device": self.device,
            "batch_size": self.batch_size
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            "model_path": self.model_path,
            "device": self.device,
            "total_parameters": param_count,
            "trainable_parameters": trainable_params,
            "model_config": self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else str(self.model.config)
        }
        
        if self.model_info:
            summary["training_info"] = self.model_info
            
        return summary


def create_inference_pipeline(model_path: str, **kwargs) -> SentimentInference:
    """Factory function to create inference pipeline."""
    return SentimentInference(model_path, **kwargs)


def main():
    """CLI entry point for inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run sentiment analysis inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model or model name")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--texts", type=str, nargs="+", help="Multiple texts to analyze")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--probabilities", action="store_true", help="Show full probability distribution")
    parser.add_argument("--attention", action="store_true", help="Show attention weights")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    
    args = parser.parse_args()
    
    # Create inference pipeline
    pipeline = SentimentInference(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Single text prediction
    if args.text:
        if args.probabilities:
            result = pipeline.predict_with_probabilities(args.text)
        elif args.attention:
            result = pipeline.get_attention_weights(args.text)
        else:
            result = pipeline.predict_single(args.text)
        
        print(json.dumps(result, indent=2))
    
    # Batch prediction
    elif args.texts:
        if args.benchmark:
            benchmark_result = pipeline.benchmark_inference(args.texts)
            print("Benchmark Results:")
            print(json.dumps(benchmark_result, indent=2))
        
        results = pipeline.predict_batch(args.texts)
        print(json.dumps(results, indent=2))
    
    # Model summary
    else:
        summary = pipeline.get_model_summary()
        print("Model Summary:")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()