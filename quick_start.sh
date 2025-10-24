#!/bin/bash

# Quick start script for the Transformer Sentiment Analysis project
# This script demonstrates all major functionalities

echo "🚀 Transformer Sentiment Analysis - Quick Start Demo"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Helper function
run_command() {
    echo -e "${BLUE}Running:${NC} $1"
    echo -e "${YELLOW}$2${NC}"
    echo "---"
}

echo -e "${GREEN}1. Basic Inference (using pre-trained model)${NC}"
run_command "Basic sentiment analysis" \
"python -m src.main --text 'I love this new transformer project!' --model distilbert-base-uncased-finetuned-sst-2-english"

echo -e "${GREEN}2. Advanced Inference with Probabilities${NC}"
run_command "Advanced inference with full probability distribution" \
"python -m src.inference --model distilbert-base-uncased-finetuned-sst-2-english --text 'This movie is fantastic!' --probabilities"

echo -e "${GREEN}3. Batch Inference${NC}"
run_command "Batch processing multiple texts" \
"python -m src.inference --model distilbert-base-uncased-finetuned-sst-2-english --texts 'Great movie' 'Terrible film' 'Okay show' --benchmark"

echo -e "${GREEN}4. Model Training (Fine-tuning)${NC}"
run_command "Train a custom model on IMDB dataset" \
"python -m src.train --config config.json --output_dir ./my_model"

echo -e "${GREEN}5. Model Interpretability${NC}"
run_command "Analyze model attention and generate explanations" \
"python -m src.interpretability --model distilbert-base-uncased-finetuned-sst-2-english --text 'This is an amazing project!' --output ./analysis"

echo -e "${GREEN}6. FastAPI Server${NC}"
run_command "Start production API server" \
"python -m src.api --model distilbert-base-uncased-finetuned-sst-2-english --host 0.0.0.0 --port 8000"

echo -e "${GREEN}7. Docker Deployment${NC}"
run_command "Deploy with Docker" \
"./deploy.sh deploy production"

echo -e "${GREEN}8. Run Tests${NC}"
run_command "Execute test suite" \
"pytest tests/ -v"

echo ""
echo -e "${GREEN}📚 API Usage Examples:${NC}"
echo "Once the API is running, you can test it with:"
echo ""
echo "# Health check"
echo "curl http://localhost:8000/health"
echo ""
echo "# Single prediction"
echo "curl -X POST http://localhost:8000/predict \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"text\": \"I love this API!\"}'"
echo ""
echo "# Batch prediction"
echo "curl -X POST http://localhost:8000/predict/batch \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"texts\": [\"Great!\", \"Terrible!\", \"Okay.\"]}'"
echo ""
echo "# Probability distribution"
echo "curl -X POST http://localhost:8000/predict/probabilities \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"text\": \"This is amazing!\"}'"

echo ""
echo -e "${GREEN}🔧 Development Commands:${NC}"
echo ""
echo "# Install dependencies"
echo "pip install -r requirements.txt"
echo ""
echo "# Run training with GPU (if available)"
echo "python -m src.train --config config.json --gpu --output_dir ./gpu_model"
echo ""
echo "# Monitor training with custom config"
echo "python -m src.train --config my_config.json --output_dir ./custom_model"
echo ""
echo "# Run interpretability analysis"
echo "python -m src.interpretability --model ./my_model --text 'Analyze this text' --output ./my_analysis"

echo ""
echo -e "${GREEN}🏗️ Project Structure:${NC}"
echo "src/"
echo "├── main.py           # Basic inference CLI"
echo "├── train.py          # Training pipeline"
echo "├── inference.py      # Advanced inference with batching"
echo "├── api.py            # FastAPI production server"
echo "├── interpretability.py # Attention visualization & SHAP"
echo "├── data_utils.py     # Dataset utilities"
echo "└── model_utils.py    # Model helpers and metrics"
echo ""
echo "tests/"
echo "├── test_main.py      # Basic tests"
echo "└── test_advanced.py  # Comprehensive test suite"
echo ""
echo "Configuration:"
echo "├── config.json       # Model and training configuration"
echo "├── requirements.txt  # Python dependencies"
echo "├── Dockerfile        # Container configuration"
echo "├── docker-compose.yml # Multi-service deployment"
echo "└── deploy.sh         # Production deployment script"

echo ""
echo -e "${GREEN}✨ Ready to explore transformer-based sentiment analysis!${NC}"