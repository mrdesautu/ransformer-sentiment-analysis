# Guía de MLflow para Sentiment Analysis

Esta guía explica cómo usar MLflow para rastrear experimentos, hiperparámetros y métricas durante el entrenamiento de modelos.

## 🎯 ¿Qué trackea MLflow automáticamente?

### Hiperparámetros registrados:
- **Modelo**: `model_name`, `num_labels`, `max_length`, `model_parameters`, `model_size_mb`
- **Entrenamiento**: `learning_rate`, `batch_size_train`, `batch_size_eval`, `num_epochs`, `weight_decay`
- **Datos**: `dataset_name`, `train_size`, `eval_size`, `test_size`

### Métricas registradas:
- **Durante entrenamiento** (cada logging_step): `loss`, `eval_loss`, `eval_accuracy`, `eval_f1`, etc.
- **Métricas finales**: `test_loss`, `test_accuracy`, `test_f1`, `test_precision`, `test_recall`

### Artefactos guardados:
- Modelo entrenado completo (PyTorch)
- Gráfico de historial de entrenamiento (`training_history.png`)
- Archivo de configuración (`config.json`)

---

## 🚀 Uso básico

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Entrenar con MLflow (habilitado por defecto)
```bash
python -m src.train --config config.json --output_dir ./my_model
```

### 3. Entrenar con nombre de experimento personalizado
```bash
python -m src.train \
  --config config.json \
  --output_dir ./my_model \
  --experiment-name "distilbert-imdb-experiment" \
  --run-name "run-lr-3e5"
```

### 4. Entrenar sin MLflow (si es necesario)
```bash
python -m src.train --config config.json --output_dir ./my_model --no-mlflow
```

---

## 📊 Ver experimentos en la UI de MLflow

### Iniciar la UI de MLflow:
```bash
mlflow ui
```

Luego abre tu navegador en: **http://localhost:5000**

### Especificar ubicación personalizada de mlruns:
```bash
mlflow ui --backend-store-uri ./mlruns
```

---

## 🔧 Configuración avanzada

### Usar servidor remoto de MLflow

1. Configurar variables de entorno:
```bash
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
export MLFLOW_EXPERIMENT_NAME="production-sentiment-model"
```

2. Entrenar:
```bash
python -m src.train --config config.json --output_dir ./my_model
```

### Configurar desde código (mlflow_config.py)

```python
from mlflow_config import setup_mlflow

# Configurar tracking URI y experimento
setup_mlflow(
    tracking_uri="http://localhost:5000",
    experiment_name="my-custom-experiment",
    artifact_location="s3://my-bucket/mlflow-artifacts"
)
```

---

## 📈 Comparar experimentos

En la UI de MLflow puedes:

1. **Ver todas las corridas** de un experimento
2. **Comparar hiperparámetros** entre diferentes runs
3. **Visualizar métricas** en gráficos interactivos
4. **Descargar modelos** guardados
5. **Registrar modelos** en el Model Registry

### Ejemplo: Comparar learning rates

Ejecuta varios entrenamientos con diferentes learning rates:

```bash
# Run 1: lr=2e-5
python -m src.train --config config.json --run-name "lr-2e5"

# Run 2: lr=3e-5
# (Modifica config.json con "learning_rate": 3e-5)
python -m src.train --config config.json --run-name "lr-3e5"

# Run 3: lr=5e-5
# (Modifica config.json con "learning_rate": 5e-5)
python -m src.train --config config.json --run-name "lr-5e5"
```

Luego en la UI selecciona las 3 corridas y haz clic en "Compare".

---

## 🗂️ Estructura de MLflow

```
mlruns/
├── 0/                          # Experimento por defecto
│   └── meta.yaml
├── 1/                          # Experimento "sentiment-analysis-training"
│   ├── meta.yaml
│   ├── <run-id-1>/
│   │   ├── artifacts/
│   │   │   ├── model/         # Modelo PyTorch guardado
│   │   │   ├── training_history.png
│   │   │   └── config.json
│   │   ├── metrics/           # Métricas por step
│   │   ├── params/            # Hiperparámetros
│   │   └── tags/
│   └── <run-id-2>/
│       └── ...
└── models/                     # Model Registry
```

---

## 🎓 Mejores prácticas

### 1. Usa nombres descriptivos
```bash
python -m src.train \
  --run-name "distilbert-imdb-4k-lr2e5-bs8" \
  --experiment-name "sentiment-imdb-experiments"
```

### 2. Documenta tus runs con tags
Puedes agregar tags programáticamente en `src/train.py`:
```python
mlflow.set_tag("git_commit", "abc123")
mlflow.set_tag("author", "tu-nombre")
mlflow.set_tag("purpose", "baseline-experiment")
```

### 3. Versiona tus experimentos
- Crea experimentos separados para diferentes datasets o arquitecturas
- Usa run names consistentes para facilitar búsquedas

### 4. Guarda artefactos adicionales
```python
# En src/train.py, puedes agregar:
mlflow.log_artifact("confusion_matrix.png")
mlflow.log_dict({"vocab_size": 30000}, "model_metadata.json")
```

---

## 🔍 Buscar y filtrar runs

### En la UI de MLflow:

1. **Filtrar por métrica**:
   ```
   metrics.test_accuracy > 0.90
   ```

2. **Filtrar por parámetro**:
   ```
   params.learning_rate = "2e-5"
   ```

3. **Filtrar por nombre**:
   ```
   attributes.run_name LIKE '%distilbert%'
   ```

4. **Combinar filtros**:
   ```
   metrics.test_accuracy > 0.90 AND params.num_epochs = "3"
   ```

---

## 🚢 Model Registry (producción)

### Registrar un modelo desde la UI:
1. Abre un run exitoso
2. En la sección "Artifacts", haz clic en el modelo
3. Click en "Register Model"
4. Elige un nombre (ej: `sentiment-model-production`)

### Registrar desde código:
```python
# Ya implementado en src/train.py
mlflow.pytorch.log_model(
    model, 
    "model",
    registered_model_name="sentiment-model-production"
)
```

### Transicionar modelos entre stages:
- **None** → **Staging** → **Production** → **Archived**

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="sentiment-model-production",
    version=1,
    stage="Production"
)
```

---

## 🐛 Troubleshooting

### Problema: "No experiments found"
**Solución**: Verifica que el directorio `mlruns/` existe o configura `MLFLOW_TRACKING_URI`.

### Problema: "Port 5000 already in use"
**Solución**: Usa un puerto diferente:
```bash
mlflow ui --port 5001
```

### Problema: Métricas no aparecen
**Solución**: Verifica que `use_mlflow=True` en `train_model()` y que no usaste `--no-mlflow`.

### Problema: Modelo muy grande para registrar
**Solución**: Usa artifact storage externo (S3, Azure Blob, GCS):
```bash
export MLFLOW_ARTIFACT_LOCATION="s3://my-bucket/mlflow"
```

---

## 📚 Recursos adicionales

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Projects](https://mlflow.org/docs/latest/projects.html)

---

## 🎯 Ejemplo completo de workflow

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar modelo baseline
python -m src.train \
  --config config.json \
  --output_dir ./models/baseline \
  --experiment-name "sentiment-analysis" \
  --run-name "baseline-distilbert"

# 3. Experimentar con hiperparámetros
# Edita config.json: learning_rate = 3e-5
python -m src.train \
  --config config.json \
  --output_dir ./models/lr-3e5 \
  --experiment-name "sentiment-analysis" \
  --run-name "experiment-lr-3e5"

# 4. Iniciar UI de MLflow
mlflow ui

# 5. Abrir navegador en http://localhost:5000
# 6. Comparar runs, seleccionar el mejor modelo
# 7. Registrar modelo ganador en Model Registry
```

---

¡Ahora tienes un sistema completo de tracking de experimentos! 🎉
