"""Comprehensive test suite for the transformer project."""

import pytest
import torch
import numpy as np
import json
import os
from unittest.mock import Mock, patch, MagicMock
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.main import predict
from src.data_utils import load_config, compute_class_distribution
from src.model_utils import compute_metrics, get_model_size
from src.inference import SentimentInference
from src.interpretability import AttentionVisualizer


class TestBasicInference:
    """Test basic inference functionality."""
    
    def test_predict_with_mock_pipeline(self, monkeypatch):
        """Test predict function with mocked pipeline."""
        class MockPipeline:
            def __call__(self, text):
                return [{"label": "POSITIVE", "score": 0.95}]
        
        monkeypatch.setattr("src.main.pipeline", lambda task, model: MockPipeline())
        
        result = predict("Great movie!", model_name="test-model", task="sentiment-analysis")
        
        assert result["text"] == "Great movie!"
        assert result["model"] == "test-model"
        assert result["task"] == "sentiment-analysis"
        assert result["result"][0]["label"] == "POSITIVE"
        assert result["result"][0]["score"] == 0.95
    
    def test_predict_type_validation(self):
        """Test input type validation."""
        with pytest.raises(TypeError):
            predict(123)
        
        with pytest.raises(TypeError):
            predict(None)
        
        with pytest.raises(TypeError):
            predict(["list", "not", "string"])


class TestDataUtils:
    """Test data utility functions."""
    
    def test_load_config(self, tmp_path):
        """Test configuration loading."""
        config = {
            "model": {"name": "test-model", "num_labels": 2},
            "training": {"learning_rate": 2e-5}
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)
        
        loaded_config = load_config(str(config_file))
        assert loaded_config == config
    
    def test_compute_class_distribution(self):
        """Test class distribution computation."""
        # Mock dataset
        mock_dataset = {"label": [0, 1, 0, 1, 1, 1]}
        
        distribution = compute_class_distribution(mock_dataset)
        
        assert "class_0" in distribution
        assert "class_1" in distribution
        assert abs(distribution["class_0"] - 0.333) < 0.01  # 2/6
        assert abs(distribution["class_1"] - 0.667) < 0.01  # 4/6


class TestModelUtils:
    """Test model utility functions."""
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        # Mock evaluation prediction
        predictions = np.array([[0.3, 0.7], [0.8, 0.2], [0.1, 0.9]])
        labels = np.array([1, 0, 1])
        
        eval_pred = (predictions, labels)
        
        with patch('src.model_utils.evaluate') as mock_evaluate:
            # Mock the evaluate.load function
            mock_accuracy = Mock()
            mock_accuracy.compute.return_value = {"accuracy": 0.67}
            
            mock_f1 = Mock()
            mock_f1.compute.return_value = {"f1": 0.65}
            
            mock_evaluate.load.side_effect = lambda metric: {
                "accuracy": mock_accuracy,
                "f1": mock_f1
            }[metric]
            
            metrics = compute_metrics(eval_pred)
            
            assert "accuracy" in metrics
            assert "f1" in metrics
            assert isinstance(metrics["accuracy"], float)
            assert isinstance(metrics["f1"], float)
    
    def test_get_model_size(self):
        """Test model size computation."""
        # Create a simple mock model
        mock_model = Mock()
        
        # Mock parameters
        param1 = Mock()
        param1.nelement.return_value = 1000
        param1.element_size.return_value = 4
        
        param2 = Mock()
        param2.nelement.return_value = 500
        param2.element_size.return_value = 4
        
        mock_model.parameters.return_value = [param1, param2]
        mock_model.buffers.return_value = []
        
        size_info = get_model_size(mock_model)
        
        assert "param_count" in size_info
        assert "total_size_mb" in size_info
        assert size_info["param_count"] == 1500


class TestAdvancedInference:
    """Test advanced inference pipeline."""
    
    @pytest.fixture
    def mock_inference_pipeline(self):
        """Create a mock inference pipeline."""
        with patch('src.inference.AutoTokenizer'), \
             patch('src.inference.AutoModelForSequenceClassification'), \
             patch('src.inference.pipeline') as mock_pipeline:
            
            mock_pipeline.return_value = Mock()
            mock_pipeline.return_value.side_effect = lambda text: [
                {"label": "POSITIVE", "score": 0.9} if "good" in text.lower() 
                else {"label": "NEGATIVE", "score": 0.8}
            ]
            
            inference = SentimentInference("test-model")
            return inference
    
    def test_predict_single(self, mock_inference_pipeline):
        """Test single prediction."""
        result = mock_inference_pipeline.predict_single("This is good!")
        
        assert result["text"] == "This is good!"
        assert result["predicted_label"] == "POSITIVE"
        assert result["confidence"] == 0.9
    
    def test_predict_batch(self, mock_inference_pipeline):
        """Test batch prediction."""
        texts = ["Good movie", "Bad film", "Great show"]
        results = mock_inference_pipeline.predict_batch(texts)
        
        assert len(results) == 3
        assert all("predicted_label" in result for result in results)
        assert all("confidence" in result for result in results)
    
    def test_benchmark_inference(self, mock_inference_pipeline):
        """Test inference benchmarking."""
        texts = ["Test text"] * 10
        
        benchmark_result = mock_inference_pipeline.benchmark_inference(texts, num_runs=2)
        
        assert "num_texts" in benchmark_result
        assert "avg_time_seconds" in benchmark_result
        assert "throughput_texts_per_second" in benchmark_result
        assert benchmark_result["num_texts"] == 10


class TestInterpretability:
    """Test interpretability functionality."""
    
    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Mock tokenizer behavior
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2023, 2003, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "this", "is", "[SEP]"]
        
        # Mock model behavior
        mock_outputs = Mock()
        mock_outputs.attentions = [torch.randn(1, 8, 4, 4)]  # 1 layer, 8 heads, 4x4 attention
        mock_outputs.logits = torch.tensor([[0.2, 0.8]])
        
        mock_model.return_value = mock_outputs
        mock_model.parameters.return_value = [torch.randn(10, 10)]
        
        return mock_model, mock_tokenizer
    
    def test_attention_visualizer_init(self, mock_model_and_tokenizer):
        """Test attention visualizer initialization."""
        model, tokenizer = mock_model_and_tokenizer
        
        visualizer = AttentionVisualizer(model, tokenizer)
        
        assert visualizer.model == model
        assert visualizer.tokenizer == tokenizer
    
    def test_get_attention_weights(self, mock_model_and_tokenizer):
        """Test attention weights extraction."""
        model, tokenizer = mock_model_and_tokenizer
        
        visualizer = AttentionVisualizer(model, tokenizer)
        
        with patch.object(visualizer.tokenizer, '__call__', return_value={
            "input_ids": torch.tensor([[101, 2023, 2003, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }):
            attention_data = visualizer.get_attention_weights("This is test")
            
            assert "tokens" in attention_data
            assert "attention_weights" in attention_data
            assert len(attention_data["attention_weights"]) > 0


class TestAPIIntegration:
    """Integration tests for the API."""
    
    @pytest.fixture
    def mock_app(self):
        """Create mock FastAPI app for testing."""
        from fastapi.testclient import TestClient
        from src.api import app
        
        # Mock the global inference pipeline
        with patch('src.api.inference_pipeline') as mock_pipeline:
            mock_pipeline.predict_single.return_value = {
                "text": "test",
                "predicted_label": "POSITIVE",
                "confidence": 0.9,
                "model_path": "test-model"
            }
            mock_pipeline.device = "cpu"
            
            client = TestClient(app)
            return client, mock_pipeline
    
    def test_health_endpoint(self, mock_app):
        """Test health check endpoint."""
        client, _ = mock_app
        
        with patch('src.api.inference_pipeline', Mock(device="cpu")):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_training_pipeline_dry_run(self, tmp_path):
        """Test training pipeline without actual training."""
        config = {
            "model": {
                "name": "distilbert-base-uncased",
                "num_labels": 2,
                "max_length": 128
            },
            "training": {
                "output_dir": str(tmp_path),
                "learning_rate": 2e-5,
                "per_device_train_batch_size": 2,
                "num_train_epochs": 1,
                "evaluation_strategy": "no",
                "save_strategy": "no"
            },
            "data": {
                "dataset_name": "imdb",
                "train_size": 10,
                "eval_size": 5,
                "test_size": 5
            }
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)
        
        # This would be a real integration test if we wanted to download models
        # For now, we just test that the config loads correctly
        loaded_config = load_config(str(config_file))
        assert loaded_config["model"]["name"] == "distilbert-base-uncased"


@pytest.mark.parametrize("text,expected_type", [
    ("Happy text", str),
    ("Sad text", str),
    ("", str),
    ("A" * 1000, str)  # Long text
])
def test_prediction_output_types(text, expected_type):
    """Parametrized test for prediction output types."""
    with patch('src.main.pipeline') as mock_pipeline:
        mock_pipeline.return_value = Mock()
        mock_pipeline.return_value.return_value = [{"label": "POSITIVE", "score": 0.9}]
        
        result = predict(text)
        assert isinstance(result["text"], expected_type)
        assert isinstance(result["predicted_label"], str)
        assert isinstance(result["confidence"], (float, int))