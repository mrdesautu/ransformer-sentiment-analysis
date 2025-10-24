"""Transformer Sentiment Analysis Package.

A comprehensive transformer-based sentiment analysis toolkit with training,
inference, interpretability, and production deployment capabilities.
"""

__version__ = "1.0.0"
__author__ = "Transformer Project"

from .main import predict
from .inference import SentimentInference, create_inference_pipeline
from .data_utils import load_config, load_and_prepare_dataset
from .model_utils import compute_metrics, load_model_and_tokenizer

__all__ = [
    "predict",
    "SentimentInference", 
    "create_inference_pipeline",
    "load_config",
    "load_and_prepare_dataset", 
    "compute_metrics",
    "load_model_and_tokenizer"
]