"""Model utilities and helper functions."""

import json
import os
from typing import Dict, Any, Optional
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate


def load_model_and_tokenizer(model_name: str, num_labels: int = 2):
    """Load pre-trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    return model, tokenizer


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }


def detailed_evaluation(y_true, y_pred, class_names: Optional[list] = None) -> Dict[str, Any]:
    """
    Perform detailed evaluation with classification report and confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        class_names: Names of classes for visualization
        
    Returns:
        Dictionary with evaluation metrics and plots
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"]
    }


def save_model_info(model_path: str, config: Dict[str, Any], metrics: Dict[str, Any]):
    """Save model information and metrics."""
    info = {
        "model_config": config,
        "training_metrics": metrics,
        "model_path": model_path
    }
    
    with open(os.path.join(model_path, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)


def get_model_size(model) -> Dict[str, Any]:
    """Get model size information."""
    param_size = 0
    param_count = 0
    
    for param in model.parameters():
        param_count += param.nelement()
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "param_count": param_count,
        "param_size_mb": param_size / 1024**2,
        "buffer_size_mb": buffer_size / 1024**2,
        "total_size_mb": size_mb
    }


def plot_training_history(trainer_log_history: list, save_path: str = "training_history.png"):
    """Plot training history from trainer logs."""
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    epochs = []
    
    for log in trainer_log_history:
        if "train_loss" in log:
            train_losses.append(log["train_loss"])
            epochs.append(log["epoch"])
        if "eval_loss" in log:
            eval_losses.append(log["eval_loss"])
        if "eval_accuracy" in log:
            eval_accuracies.append(log["eval_accuracy"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, label="Train Loss", marker='o')
    if eval_losses:
        # Crear epochs correspondientes a las evaluaciones
        eval_epochs = [i+1 for i in range(len(eval_losses))]
        ax1.plot(eval_epochs, eval_losses, label="Eval Loss", marker='s')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Evaluation Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    if eval_accuracies:
        # Crear epochs correspondientes a las evaluaciones de accuracy
        eval_acc_epochs = [i+1 for i in range(len(eval_accuracies))]
        ax2.plot(eval_acc_epochs, eval_accuracies, 
                label="Eval Accuracy", marker='s', color='green')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Evaluation Accuracy")
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def estimate_gpu_memory(model, batch_size: int, seq_length: int) -> Dict[str, float]:
    """Estimate GPU memory requirements."""
    model_size = get_model_size(model)["total_size_mb"]
    
    # Rough estimation for activations (this is a simplified calculation)
    activation_size_mb = batch_size * seq_length * model.config.hidden_size * 4 / 1024**2
    
    # Gradients are roughly the same size as model parameters
    gradient_size_mb = model_size
    
    # Add some overhead
    overhead_mb = 500
    
    total_mb = model_size + activation_size_mb + gradient_size_mb + overhead_mb
    
    return {
        "model_size_mb": model_size,
        "activation_size_mb": activation_size_mb,
        "gradient_size_mb": gradient_size_mb,
        "overhead_mb": overhead_mb,
        "total_estimated_mb": total_mb,
        "total_estimated_gb": total_mb / 1024
    }