# Advanced Transformer Sentiment Analysis

A production-ready sentiment analysis toolkit with Hugging Face Transformers, MLflow experiment tracking, and interactive Gradio UI.

## 🎭 Live Demo

**🌐 Try it now:** [HuggingFace Space](https://huggingface.co/spaces/MartinRodrigo/transformer-sentiment-analysis)

**📦 Model:** [MartinRodrigo/distilbert-sentiment-imdb](https://huggingface.co/MartinRodrigo/distilbert-sentiment-imdb)

## 🚀 Features

- **MLflow Experiment Tracking** - Automatic logging of hyperparameters, metrics, and models
- **Interactive Gradio UI** - Real-time analysis with MLflow metrics visualization
- **Hyperparameter Optimization** - Bayesian optimization with Optuna
- **Production API** - FastAPI with batch processing and benchmarking
- **Model Interpretability** - Attention visualization and SHAP explanations
- **Comprehensive Testing** - Unit and integration tests

## 🏗️ Project Structure

```
├── src/
│   ├── train.py             # Training with MLflow tracking
│   ├── api.py               # FastAPI production server
│   ├── inference.py         # Batch inference
│   └── interpretability.py  # Attention & SHAP tools
├── app.py                   # HuggingFace Spaces app
├── gradio_app.py            # Local UI with MLflow
├── hyperparameter_experiments.py  # Experiment strategies
├── optuna_optimization.py   # Bayesian optimization
├── upload_to_huggingface.py # HF Hub deployment
└── guide/                   # Documentation
```

## ⚡ Quick Start

### Installation

```bash
git clone <repo-url>
cd transformer-sentiment-analysis
pip install -r requirements.txt
```

### Train Model with MLflow

```bash
python -m src.train \
  --config config.json \
  --output_dir ./trained_model \
  --run-name "production-v1"

# View experiments
mlflow ui --host 127.0.0.1 --port 5000
```

### Launch Gradio UI

**Local with MLflow integration:**
```bash
python gradio_app.py
# Open http://localhost:7860
```

**HuggingFace Spaces (deployed):**
- Visit: https://huggingface.co/spaces/MartinRodrigo/transformer-sentiment-analysis
- Features: Real-time analysis, batch processing, interactive visualizations

### Hyperparameter Optimization

```bash
# Interactive menu
python hyperparameter_experiments.py

# Bayesian optimization
python optuna_optimization.py --n-trials 20
```

### Deploy to HuggingFace

**Model deployment:**
```bash
python upload_to_huggingface.py \
  --model-dir ./trained_model \
  --repo-name your-username/sentiment-model
```

**Space deployment (current):**
- Model: [MartinRodrigo/distilbert-sentiment-imdb](https://huggingface.co/MartinRodrigo/distilbert-sentiment-imdb)
- Space: [MartinRodrigo/transformer-sentiment-analysis](https://huggingface.co/spaces/MartinRodrigo/transformer-sentiment-analysis)

## 🎯 Key Features

### MLflow Experiment Tracking
- Real-time metric logging (accuracy, F1, loss)
- Hyperparameter comparison and visualization
- Model artifact versioning
- Interactive UI at http://localhost:5000
- 7 pre-configured experiment strategies
- Bayesian optimization with Optuna

### Gradio Interactive UI
- **Local version** (`gradio_app.py`): MLflow integration with experiment tracking
- **HuggingFace Space** (`app.py`): Clean deployment without MLflow dependencies
- Real-time sentiment analysis with probability visualization
- Batch text processing with analytics charts
- Interactive dark-themed Plotly visualizations
- Pre-loaded examples and user-friendly interface

### Production API
**Endpoints:**
- `POST /predict` - Single/batch predictions
- `GET /model/info` - Model metadata
- `GET /health` - Health check

**Features:** Auto-batching, validation, CORS support

### Model Interpretability
- Attention visualization (layer-wise heatmaps)
- SHAP explanations
- Token importance scoring

## 📊 Performance

- **Model**: DistilBERT (~67M params, 255MB)
- **Accuracy**: 80% on IMDB test set
- **Speed**: ~100ms per prediction (CPU)
- **Best F1**: 0.7981

## �️ Configuration

Edit `config.json` to customize training:

```json
{
  "model": {"name": "distilbert-base-uncased", "num_labels": 2},
  "training": {"learning_rate": 2e-5, "num_train_epochs": 3},
  "data": {"dataset_name": "imdb", "train_size": 4000},
  "mlflow": {"enabled": true, "experiment_name": "sentiment-analysis"}
}
```

## 📚 Documentation

Guides in `guide/` directory:
- **MLFLOW_GUIDE.md** - Setup and usage
- **EXPERIMENTATION_GUIDE.md** - Hyperparameter strategies
- **TRAINING_AND_UPLOAD_GUIDE.md** - End-to-end workflow

## 🤝 Contributing

Production-ready ML practices:
- Modular architecture
- MLflow experiment tracking
- Comprehensive testing
- Interactive UI with metrics
- HuggingFace Hub deployment

---

**🌟 Star if useful for learning ML engineering!**

