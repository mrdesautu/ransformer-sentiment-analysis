"""Training script for fine-tuning transformer models."""

import os
import argparse
import json
from typing import Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from src.data_utils import load_config, load_and_prepare_dataset, prepare_labels_for_classification
from src.model_utils import compute_metrics, save_model_info, plot_training_history, get_model_size


def setup_training_args(config: dict, output_dir: str) -> TrainingArguments:
    """Setup training arguments from config."""
    training_config = config["training"]
    training_config["output_dir"] = output_dir
    
    return TrainingArguments(**training_config)


def train_model(
    config_path: str = "config.json",
    output_dir: str = "./results",
    resume_from_checkpoint: Optional[str] = None
):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for model and results
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Load configuration
    config = load_config(config_path)
    
    print("🚀 Starting training with configuration:")
    print(json.dumps(config, indent=2))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model_name = config["model"]["name"]
    num_labels = config["model"]["num_labels"]
    max_length = config["model"]["max_length"]
    
    print(f"📦 Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Print model information
    model_info = get_model_size(model)
    print(f"📊 Model info: {model_info['param_count']:,} parameters, {model_info['total_size_mb']:.1f} MB")
    
    # Load and prepare dataset
    data_config = config["data"]
    print(f"📚 Loading dataset: {data_config['dataset_name']}")
    
    train_dataset, eval_dataset, test_dataset = load_and_prepare_dataset(
        dataset_name=data_config["dataset_name"],
        tokenizer_name=model_name,
        train_size=data_config["train_size"],
        eval_size=data_config["eval_size"],
        test_size=data_config["test_size"],
        max_length=max_length
    )
    
    # Prepare labels
    train_dataset = prepare_labels_for_classification(train_dataset)
    eval_dataset = prepare_labels_for_classification(eval_dataset)
    test_dataset = prepare_labels_for_classification(test_dataset)
    
    print(f"📈 Dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")
    
    # Setup training arguments
    training_args = setup_training_args(config, output_dir)
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    print("🎯 Starting training...")
    if resume_from_checkpoint:
        print(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save the model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Plot training history
    if hasattr(trainer.state, 'log_history'):
        print("📊 Plotting training history...")
        plot_training_history(
            trainer.state.log_history, 
            os.path.join(output_dir, "training_history.png")
        )
    
    # Final evaluation on test set
    print("🔍 Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    
    print("✅ Training completed!")
    print("📋 Final test results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model info and metrics
    save_model_info(output_dir, config, test_results)
    
    return trainer, test_results


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train a transformer model for sentiment analysis")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage (if available)")
    
    args = parser.parse_args()
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"🚀 GPU available: {device}")
        if args.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        print("💻 Running on CPU")
    
    # Run training
    trainer, results = train_model(
        config_path=args.config,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume
    )
    
    print(f"🎉 Training finished! Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()