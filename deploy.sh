#!/bin/bash

# Production deployment script for Transformer Sentiment Analysis API
# Usage: ./deploy.sh [environment] [options]

set -e  # Exit on any error

# Configuration
PROJECT_NAME="transformer-sentiment"
DOCKER_IMAGE="${PROJECT_NAME}:latest"
BACKUP_DIR="./backups"
LOG_DIR="./logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    log_info "Dependencies check passed"
}

# Create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    mkdir -p $BACKUP_DIR
    mkdir -p $LOG_DIR
    mkdir -p ./monitoring
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    docker build -t $DOCKER_IMAGE .
    log_info "Docker image built successfully"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Run tests in container
    docker run --rm -v $(pwd):/app -w /app $DOCKER_IMAGE pytest tests/ -v
    
    if [ $? -eq 0 ]; then
        log_info "All tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
}

# Backup current deployment
backup_deployment() {
    if [ -f "docker-compose.yml" ]; then
        log_info "Creating backup..."
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        cp docker-compose.yml $BACKUP_DIR/docker-compose_$TIMESTAMP.yml
        log_info "Backup created: $BACKUP_DIR/docker-compose_$TIMESTAMP.yml"
    fi
}

# Deploy application
deploy() {
    local environment=${1:-production}
    
    log_info "Deploying to $environment environment..."
    
    # Set environment variables
    case $environment in
        "production")
            export MODEL_PATH="./results"
            export WORKERS=4
            ;;
        "staging")
            export MODEL_PATH="distilbert-base-uncased-finetuned-sst-2-english"
            export WORKERS=2
            ;;
        "development")
            export MODEL_PATH="distilbert-base-uncased-finetuned-sst-2-english"
            export WORKERS=1
            ;;
        *)
            log_error "Unknown environment: $environment"
            exit 1
            ;;
    esac
    
    # Stop existing containers
    log_info "Stopping existing containers..."
    docker-compose down || true
    
    # Start new deployment
    log_info "Starting new deployment..."
    docker-compose up -d
    
    # Wait for health check
    log_info "Waiting for health check..."
    sleep 30
    
    # Check if API is responding
    for i in {1..10}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_info "Deployment successful! API is responding"
            return 0
        fi
        log_warn "Attempt $i: API not responding yet, waiting..."
        sleep 10
    done
    
    log_error "Deployment failed: API not responding after 100 seconds"
    docker-compose logs
    exit 1
}

# Rollback deployment
rollback() {
    log_warn "Rolling back deployment..."
    
    # Find latest backup
    LATEST_BACKUP=$(ls -t $BACKUP_DIR/docker-compose_*.yml 2>/dev/null | head -n1)
    
    if [ -z "$LATEST_BACKUP" ]; then
        log_error "No backup found for rollback"
        exit 1
    fi
    
    log_info "Rolling back to: $LATEST_BACKUP"
    
    # Stop current deployment
    docker-compose down
    
    # Restore backup
    cp $LATEST_BACKUP docker-compose.yml
    
    # Restart with backup configuration
    docker-compose up -d
    
    log_info "Rollback completed"
}

# Show status
show_status() {
    log_info "Deployment Status:"
    docker-compose ps
    
    echo ""
    log_info "API Health:"
    curl -s http://localhost:8000/health | python -m json.tool || echo "API not responding"
    
    echo ""
    log_info "Container Logs (last 20 lines):"
    docker-compose logs --tail=20
}

# Monitor deployment
monitor() {
    log_info "Monitoring deployment..."
    docker-compose logs -f
}

# Update model
update_model() {
    local model_path=$1
    
    if [ -z "$model_path" ]; then
        log_error "Model path required"
        exit 1
    fi
    
    log_info "Updating model to: $model_path"
    
    # Update environment variable
    export MODEL_PATH=$model_path
    
    # Restart services
    docker-compose restart transformer-api
    
    log_info "Model updated successfully"
}

# Cleanup old resources
cleanup() {
    log_info "Cleaning up old resources..."
    
    # Remove old Docker images
    docker image prune -f
    
    # Remove old backups (keep last 10)
    ls -t $BACKUP_DIR/docker-compose_*.yml 2>/dev/null | tail -n +11 | xargs rm -f
    
    # Remove old logs (older than 7 days)
    find $LOG_DIR -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    log_info "Cleanup completed"
}

# Main script
main() {
    local command=${1:-deploy}
    local environment=${2:-production}
    
    case $command in
        "deploy")
            check_dependencies
            setup_directories
            build_image
            run_tests
            backup_deployment
            deploy $environment
            ;;
        "rollback")
            rollback
            ;;
        "status")
            show_status
            ;;
        "monitor")
            monitor
            ;;
        "update-model")
            update_model $2
            ;;
        "cleanup")
            cleanup
            ;;
        "build")
            build_image
            ;;
        "test")
            run_tests
            ;;
        *)
            echo "Usage: $0 {deploy|rollback|status|monitor|update-model|cleanup|build|test} [environment|model_path]"
            echo ""
            echo "Commands:"
            echo "  deploy [env]     - Deploy application (env: production|staging|development)"
            echo "  rollback         - Rollback to previous deployment"
            echo "  status           - Show deployment status"
            echo "  monitor          - Monitor deployment logs"
            echo "  update-model     - Update model path"
            echo "  cleanup          - Clean up old resources"
            echo "  build            - Build Docker image only"
            echo "  test             - Run tests only"
            echo ""
            echo "Examples:"
            echo "  $0 deploy production"
            echo "  $0 update-model ./new-model"
            echo "  $0 status"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"