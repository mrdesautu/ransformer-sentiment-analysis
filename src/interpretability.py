"""Model interpretability and visualization tools."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server deployment
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")


class AttentionVisualizer:
    """Visualize attention weights from transformer models."""
    
    def __init__(self, model, tokenizer):
        """
        Initialize attention visualizer.
        
        Args:
            model: Transformer model
            tokenizer: Corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def get_attention_weights(self, text: str) -> Dict[str, Any]:
        """Get attention weights for a given text."""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions
        
        # Convert to numpy
        attention_weights = [att.cpu().numpy() for att in attentions]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return {
            "tokens": tokens,
            "attention_weights": attention_weights,
            "input_ids": inputs["input_ids"].cpu().numpy(),
            "predictions": torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        }
    
    def plot_attention_heatmap(
        self, 
        text: str, 
        layer: int = -1, 
        head: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Plot attention heatmap for a specific layer and head.
        
        Args:
            text: Input text
            layer: Layer index (-1 for last layer)
            head: Attention head index
            save_path: Path to save the plot
        """
        attention_data = self.get_attention_weights(text)
        tokens = attention_data["tokens"]
        attention_weights = attention_data["attention_weights"]
        
        # Select layer and head
        layer_attention = attention_weights[layer][0, head]  # [seq_len, seq_len]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Filter out special tokens for cleaner visualization
        token_labels = []
        for token in tokens:
            if token.startswith('##'):
                token_labels.append(token[2:])
            elif token in ['[CLS]', '[SEP]', '[PAD]']:
                token_labels.append(token)
            else:
                token_labels.append(token)
        
        # Truncate if too many tokens
        max_tokens = 50
        if len(token_labels) > max_tokens:
            layer_attention = layer_attention[:max_tokens, :max_tokens]
            token_labels = token_labels[:max_tokens]
        
        sns.heatmap(
            layer_attention,
            xticklabels=token_labels,
            yticklabels=token_labels,
            cmap='Blues',
            cbar=True,
            square=True
        )
        
        plt.title(f'Attention Weights - Layer {layer}, Head {head}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attention_summary(
        self, 
        text: str, 
        save_path: Optional[str] = None
    ):
        """
        Plot attention summary across all layers and heads.
        
        Args:
            text: Input text
            save_path: Path to save the plot
        """
        attention_data = self.get_attention_weights(text)
        attention_weights = attention_data["attention_weights"]
        tokens = attention_data["tokens"]
        
        num_layers = len(attention_weights)
        num_heads = attention_weights[0].shape[1]
        
        # Calculate average attention per layer
        layer_avg_attention = []
        for layer_att in attention_weights:
            # Average across heads and sequence positions
            avg_att = np.mean(layer_att[0])  # [num_heads, seq_len, seq_len]
            layer_avg_attention.append(avg_att)
        
        # Calculate attention variance per head
        head_attention_variance = []
        for head in range(num_heads):
            head_variances = []
            for layer_att in attention_weights:
                head_att = layer_att[0, head]  # [seq_len, seq_len]
                variance = np.var(head_att)
                head_variances.append(variance)
            head_attention_variance.append(head_variances)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Average attention per layer
        ax1.plot(range(num_layers), layer_avg_attention, marker='o')
        ax1.set_title('Average Attention Weight per Layer')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Average Attention')
        ax1.grid(True)
        
        # Plot 2: Attention variance per head across layers
        for head in range(min(num_heads, 8)):  # Show max 8 heads
            ax2.plot(range(num_layers), head_attention_variance[head], 
                    marker='o', label=f'Head {head}')
        ax2.set_title('Attention Variance per Head Across Layers')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Attention Variance')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Last layer attention heatmap (head 0)
        last_layer_att = attention_weights[-1][0, 0]
        max_tokens = 20
        if len(tokens) > max_tokens:
            last_layer_att = last_layer_att[:max_tokens, :max_tokens]
            display_tokens = tokens[:max_tokens]
        else:
            display_tokens = tokens
        
        im = ax3.imshow(last_layer_att, cmap='Blues')
        ax3.set_title('Last Layer Attention (Head 0)')
        ax3.set_xticks(range(len(display_tokens)))
        ax3.set_yticks(range(len(display_tokens)))
        ax3.set_xticklabels(display_tokens, rotation=45, ha='right')
        ax3.set_yticklabels(display_tokens)
        
        # Plot 4: Token attention sum (how much attention each token receives)
        token_attention_sum = np.sum(last_layer_att, axis=0)
        ax4.bar(range(len(display_tokens)), token_attention_sum)
        ax4.set_title('Total Attention Received per Token')
        ax4.set_xlabel('Token')
        ax4.set_ylabel('Total Attention')
        ax4.set_xticks(range(len(display_tokens)))
        ax4.set_xticklabels(display_tokens, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class SHAPExplainer:
    """SHAP-based explainability for transformer models."""
    
    def __init__(self, model, tokenizer):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Transformer model
            tokenizer: Corresponding tokenizer
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for this functionality. Install with: pip install shap")
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Create prediction function for SHAP
        self.explainer = shap.Explainer(self._predict_fn, self.tokenizer)
    
    def _predict_fn(self, texts):
        """Prediction function for SHAP."""
        predictions = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                predictions.append(probs.cpu().numpy()[0])
        
        return np.array(predictions)
    
    def explain_text(self, text: str, max_evals: int = 100):
        """
        Generate SHAP explanations for a text.
        
        Args:
            text: Input text to explain
            max_evals: Maximum number of evaluations for SHAP
            
        Returns:
            SHAP explanation object
        """
        shap_values = self.explainer([text], max_evals=max_evals)
        return shap_values
    
    def plot_shap_explanation(
        self, 
        text: str, 
        class_index: int = 1,
        max_evals: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP explanation for a specific class.
        
        Args:
            text: Input text
            class_index: Class index to explain
            max_evals: Maximum evaluations for SHAP
            save_path: Path to save the plot
        """
        shap_values = self.explain_text(text, max_evals=max_evals)
        
        # Plot explanation
        shap.plots.text(shap_values[0, :, class_index])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')


class InterpretabilityPipeline:
    """Complete interpretability pipeline combining multiple methods."""
    
    def __init__(self, model_path: str):
        """
        Initialize interpretability pipeline.
        
        Args:
            model_path: Path to trained model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        # Initialize visualizers
        self.attention_viz = AttentionVisualizer(self.model, self.tokenizer)
        
        if SHAP_AVAILABLE:
            self.shap_explainer = SHAPExplainer(self.model, self.tokenizer)
        else:
            self.shap_explainer = None
            print("Warning: SHAP not available. Install with: pip install shap")
    
    def full_analysis(
        self, 
        text: str, 
        output_dir: str = "./interpretability_output"
    ):
        """
        Perform full interpretability analysis.
        
        Args:
            text: Text to analyze
            output_dir: Directory to save outputs
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔍 Analyzing text: {text[:100]}...")
        
        # 1. Get prediction
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0, predicted_class].item()
        
        print(f"📊 Prediction: Class {predicted_class}, Confidence: {confidence:.3f}")
        
        # 2. Attention visualization
        print("🎯 Generating attention visualizations...")
        self.attention_viz.plot_attention_summary(
            text, 
            save_path=os.path.join(output_dir, "attention_summary.png")
        )
        
        self.attention_viz.plot_attention_heatmap(
            text,
            layer=-1,
            head=0,
            save_path=os.path.join(output_dir, "attention_heatmap.png")
        )
        
        # 3. SHAP explanation (if available)
        if self.shap_explainer:
            print("🔬 Generating SHAP explanations...")
            try:
                self.shap_explainer.plot_shap_explanation(
                    text,
                    class_index=predicted_class,
                    save_path=os.path.join(output_dir, "shap_explanation.png")
                )
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
        
        # 4. Generate report
        report = {
            "text": text,
            "predicted_class": int(predicted_class),
            "confidence": float(confidence),
            "model_path": self.model.config._name_or_path,
            "analysis_files": {
                "attention_summary": "attention_summary.png",
                "attention_heatmap": "attention_heatmap.png",
                "shap_explanation": "shap_explanation.png" if self.shap_explainer else None
            }
        }
        
        report_path = os.path.join(output_dir, "analysis_report.json")
        with open(report_path, "w") as f:
            import json
            json.dump(report, f, indent=2)
        
        print(f"✅ Analysis complete! Results saved to: {output_dir}")
        return report


def main():
    """CLI for interpretability analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model interpretability analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    parser.add_argument("--output", type=str, default="./interpretability_output", help="Output directory")
    parser.add_argument("--attention-only", action="store_true", help="Only run attention analysis")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = InterpretabilityPipeline(args.model)
    
    if args.attention_only:
        pipeline.attention_viz.plot_attention_summary(args.text)
    else:
        pipeline.full_analysis(args.text, args.output)


if __name__ == "__main__":
    main()