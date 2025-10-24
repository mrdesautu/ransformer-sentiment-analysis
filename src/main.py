"""Simple inference CLI using Hugging Face transformers.pipeline.

This module exposes `predict(text, model_name, task)` for programmatic use
and a CLI entrypoint.
"""
from typing import Any, Dict
import argparse
import json

from transformers import pipeline


def predict(text: str, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english", task: str = "sentiment-analysis") -> Dict[str, Any]:
    """Run a transformers pipeline on the given text.

    Inputs:
      - text: input string
      - model_name: model id or path
      - task: transformers task name

    Returns a dict with keys: text, model, task, result
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    pipe = pipeline(task, model=model_name)
    result = pipe(text)

    return {
        "text": text,
        "model": model_name,
        "task": task,
        "result": result,
    }


def _cli():
    parser = argparse.ArgumentParser(description="Minimal transformer inference CLI")
    parser.add_argument("--text", type=str, required=True, help="Input text to analyze")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased-finetuned-sst-2-english", help="Model name or path")
    parser.add_argument("--task", type=str, default="sentiment-analysis", help="Transformers task (default: sentiment-analysis)")
    args = parser.parse_args()

    out = predict(args.text, model_name=args.model, task=args.task)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _cli()
