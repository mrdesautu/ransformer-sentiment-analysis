#!/usr/bin/env python3
"""
Script para subir modelo entrenado a Hugging Face Hub.
Incluye el modelo, tokenizer, métricas y metadatos completos.
"""

import argparse
import json
from pathlib import Path
import os

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    from huggingface_hub import login as hf_login
except ImportError:
    print("❌ huggingface_hub no está instalado")
    print("   Instala con: pip install huggingface_hub")
    exit(1)


def load_model_info(model_dir):
    """Carga la información del modelo entrenado."""
    model_info_path = Path(model_dir) / "model_info.json"
    
    if not model_info_path.exists():
        print(f"⚠️  No se encontró model_info.json en {model_dir}")
        return None
    
    with open(model_info_path, 'r') as f:
        return json.load(f)


def create_model_card(model_info, model_dir, additional_info=None):
    """Crea un README.md (Model Card) para Hugging Face."""
    
    config = model_info.get('config', {})
    metrics = model_info.get('test_metrics', {})
    
    # Extraer información
    model_name = config.get('model', {}).get('name', 'unknown')
    dataset = config.get('data', {}).get('dataset_name', 'unknown')
    
    # Crear Model Card
    model_card = f"""---
language: en
license: apache-2.0
tags:
- sentiment-analysis
- transformers
- {model_name}
- text-classification
datasets:
- {dataset}
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: {model_name}-sentiment
  results:
  - task:
      type: text-classification
      name: Sentiment Analysis
    dataset:
      name: {dataset.upper()}
      type: {dataset}
    metrics:
    - type: accuracy
      value: {metrics.get('test_accuracy', 0):.4f}
      name: Test Accuracy
    - type: f1
      value: {metrics.get('test_f1', 0):.4f}
      name: F1 Score
    - type: precision
      value: {metrics.get('test_precision', 0):.4f}
      name: Precision
    - type: recall
      value: {metrics.get('test_recall', 0):.4f}
      name: Recall
---

# {model_name.upper()} Fine-tuned for Sentiment Analysis

## 📊 Model Description

This model is a fine-tuned version of `{model_name}` for sentiment analysis on the {dataset.upper()} dataset.

**Model Architecture:** {model_name}  
**Task:** Binary Sentiment Classification (Positive/Negative)  
**Language:** English  
**Training Date:** {model_info.get('training_date', 'N/A')}

## 🎯 Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | {metrics.get('test_accuracy', 0):.4f} |
| **F1 Score** | {metrics.get('test_f1', 0):.4f} |
| **Precision** | {metrics.get('test_precision', 0):.4f} |
| **Recall** | {metrics.get('test_recall', 0):.4f} |
| **Loss** | {metrics.get('test_loss', 0):.4f} |

## 🔧 Training Details

### Hyperparameters

```json
{json.dumps(config.get('training', {}), indent=2)}
```

### Dataset
- **Training samples:** {config.get('data', {}).get('train_size', 'N/A')}
- **Validation samples:** {config.get('data', {}).get('eval_size', 'N/A')}
- **Test samples:** {config.get('data', {}).get('test_size', 'N/A')}

## 🚀 Usage

### With Transformers Pipeline

```python
from transformers import pipeline

# Load the model
classifier = pipeline("sentiment-analysis", model="YOUR_USERNAME/YOUR_MODEL_NAME")

# Predict
result = classifier("I love this movie!")
print(result)
# [{{'label': 'POSITIVE', 'score': 0.9998}}]
```

### Manual Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "YOUR_USERNAME/YOUR_MODEL_NAME"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare input
text = "This is an amazing product!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
# Get result
label_id = torch.argmax(predictions).item()
score = predictions[0][label_id].item()

labels = ["NEGATIVE", "POSITIVE"]
print(f"Label: {{labels[label_id]}}, Score: {{score:.4f}}")
```

## 📈 Training Curves

Training history visualization is available in the model files.

## 🏷️ Label Mapping

```
0: NEGATIVE
1: POSITIVE
```

## ⚙️ Model Configuration

```json
{json.dumps(config.get('model', {}), indent=2)}
```

## 📝 Citation

If you use this model, please cite:

```bibtex
@misc{{sentiment-model-{dataset},
  author = {{Your Name}},
  title = {{{model_name} Fine-tuned for Sentiment Analysis}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_NAME}}}}
}}
```

## 🤝 Contact

For questions or feedback, please open an issue in the repository.

## 📄 License

Apache 2.0

## 🔗 Related Models

- [{model_name}](https://huggingface.co/{model_name})

---

**Generated with MLflow tracking** 🚀
"""

    # Guardar Model Card
    readme_path = Path(model_dir) / "README.md"
    with open(readme_path, 'w') as f:
        f.write(model_card)
    
    print(f"✅ Model Card creado: {readme_path}")
    
    return readme_path


def upload_to_huggingface(
    model_dir,
    repo_name,
    organization=None,
    private=False,
    token=None
):
    """
    Sube el modelo a Hugging Face Hub.
    
    Args:
        model_dir: Directorio con el modelo entrenado
        repo_name: Nombre del repositorio en HF
        organization: Organización (opcional)
        private: Si el repo debe ser privado
        token: Token de HF (opcional, usa login si no se provee)
    """
    
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        print(f"❌ Directorio no encontrado: {model_dir}")
        return False
    
    # Login si se provee token
    if token:
        hf_login(token=token)
    else:
        print("\n🔐 Necesitas estar autenticado en Hugging Face")
        print("   Opción 1: Ejecuta 'huggingface-cli login' en la terminal")
        print("   Opción 2: Provee --token YOUR_TOKEN")
        try:
            hf_login()
        except Exception as e:
            print(f"❌ Error de autenticación: {e}")
            return False
    
    # Crear nombre completo del repo
    if organization:
        repo_id = f"{organization}/{repo_name}"
    else:
        repo_id = repo_name
    
    print(f"\n📤 Subiendo modelo a: {repo_id}")
    
    # Crear repositorio
    try:
        api = HfApi()
        url = create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✅ Repositorio creado/actualizado: {url}")
    except Exception as e:
        print(f"⚠️  Error creando repo (puede que ya exista): {e}")
    
    # Cargar info del modelo y crear Model Card
    model_info = load_model_info(model_dir)
    if model_info:
        create_model_card(model_info, model_dir)
    else:
        print("⚠️  Creando Model Card básico...")
        basic_readme = model_dir / "README.md"
        if not basic_readme.exists():
            with open(basic_readme, 'w') as f:
                f.write(f"# {repo_name}\n\nSentiment analysis model.\n")
    
    # Subir todos los archivos
    try:
        print(f"\n📁 Subiendo archivos desde: {model_dir}")
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
        )
        
        print(f"\n✅ ¡Modelo subido exitosamente!")
        print(f"🔗 URL: https://huggingface.co/{repo_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error subiendo modelo: {e}")
        return False


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Subir modelo entrenado a Hugging Face Hub"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directorio del modelo entrenado (ej: ./results)"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Nombre del repositorio en HF (ej: distilbert-sentiment-imdb)"
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="Organización de HF (opcional)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Hacer el repositorio privado"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token de Hugging Face (opcional)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("📤 SUBIR MODELO A HUGGING FACE HUB")
    print("="*70)
    
    success = upload_to_huggingface(
        model_dir=args.model_dir,
        repo_name=args.repo_name,
        organization=args.organization,
        private=args.private,
        token=args.token
    )
    
    if success:
        print("\n" + "="*70)
        print("🎉 ¡Proceso completado!")
        print("="*70)
        
        repo_id = f"{args.organization}/{args.repo_name}" if args.organization else args.repo_name
        
        print(f"\n🔗 Tu modelo está disponible en:")
        print(f"   https://huggingface.co/{repo_id}")
        
        print(f"\n💡 Para usarlo:")
        print(f'   from transformers import pipeline')
        print(f'   classifier = pipeline("sentiment-analysis", model="{repo_id}")')
        print(f'   result = classifier("I love this!")')
    else:
        print("\n❌ Error subiendo el modelo")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
