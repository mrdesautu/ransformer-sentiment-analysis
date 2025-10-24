"""Data utilities for loading and preprocessing datasets."""

import json
from typing import Dict, Any, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import numpy as np


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_and_prepare_dataset(
    dataset_name: str,
    tokenizer_name: str,
    train_size: int = 4000,
    eval_size: int = 1000,
    test_size: int = 500,
    max_length: int = 512
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load dataset and prepare for training.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'imdb')
        tokenizer_name: Name of the tokenizer to use
        train_size: Number of training samples
        eval_size: Number of evaluation samples  
        test_size: Number of test samples
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, eval_dataset, test_dataset)
    """
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Prepare train/eval/test splits
    train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(train_size))
    
    # Use test set for both eval and final test
    test_full = tokenized_dataset["test"].shuffle(seed=42)
    eval_dataset = test_full.select(range(eval_size))
    test_dataset = test_full.select(range(eval_size, eval_size + test_size))
    
    return train_dataset, eval_dataset, test_dataset


def prepare_labels_for_classification(dataset: Dataset) -> Dataset:
    """Ensure labels are properly formatted for classification."""
    def format_labels(example):
        example["labels"] = example["label"]
        return example
    
    return dataset.map(format_labels)


class DataCollector:
    """Custom data collector for handling various data preprocessing needs."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        """Standard data collation for transformer training."""
        batch = self.tokenizer.pad(features, return_tensors="pt")
        return batch


def compute_class_distribution(dataset: Dataset) -> Dict[str, float]:
    """Compute class distribution in the dataset."""
    labels = dataset["label"] if "label" in dataset.column_names else dataset["labels"]
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    distribution = {}
    for label, count in zip(unique, counts):
        distribution[f"class_{label}"] = count / total
        
    return distribution


def get_sample_texts(dataset: Dataset, n_samples: int = 5) -> list:
    """Get sample texts from dataset for inspection."""
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    samples = []
    
    for idx in indices:
        sample = dataset[idx]
        samples.append({
            "text": sample["text"][:200] + "..." if len(sample["text"]) > 200 else sample["text"],
            "label": sample["label"] if "label" in sample else sample["labels"]
        })
    
    return samples