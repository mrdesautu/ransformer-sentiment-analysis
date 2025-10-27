# 🚀 Guía Completa: Entrenar y Subir Modelo a Hugging Face

## 📋 Proceso Completo

### Paso 1: Entrenar el Modelo

#### Opción A: Entrenamiento rápido (prueba)
```bash
python -m src.train \
  --config config_rapido.json \
  --output_dir ./quick_model \
  --run-name "quick-test" \
  --experiment-name "experiments"
```

#### Opción B: Entrenamiento completo (producción)
```bash
python -m src.train \
  --config config.json \
  --output_dir ./trained_model \
  --run-name "production-v1" \
  --experiment-name "production-training"
```

**Tiempo estimado:**
- CPU: 30-60 minutos (config.json)
- CPU: 5-10 minutos (config_rapido.json)
- GPU: 10-15 minutos (config.json)

---

### Paso 2: Verificar Resultados en MLflow

Mientras el modelo entrena, abre otra terminal:

```bash
mlflow ui
```

Luego abre: http://localhost:5000

**Qué verás:**
- ✅ Experimento "production-training"
- ✅ Run "production-v1"
- ✅ Hiperparámetros registrados
- ✅ Métricas en tiempo real
- ✅ Gráficos de training
- ✅ Modelo guardado

---

### Paso 3: Revisar el Modelo Entrenado

Una vez completado el entrenamiento, revisa:

```bash
ls -lah ./trained_model/
```

**Archivos generados:**
```
trained_model/
├── config.json                 # Configuración del modelo
├── model.safetensors          # Pesos del modelo
├── tokenizer_config.json      # Configuración del tokenizer
├── vocab.txt                  # Vocabulario
├── special_tokens_map.json    # Tokens especiales
├── model_info.json           # Métricas y metadatos
├── training_history.png      # Gráfico de entrenamiento
└── README.md                 # (se creará al subir a HF)
```

**Ver métricas:**
```bash
cat ./trained_model/model_info.json
```

---

### Paso 4: Probar el Modelo Localmente

```python
from transformers import pipeline

# Cargar modelo local
classifier = pipeline(
    "sentiment-analysis", 
    model="./trained_model"
)

# Probar
result = classifier("This movie is absolutely amazing!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

result = classifier("I hated this product, waste of money.")
print(result)
# [{'label': 'NEGATIVE', 'score': 0.9995}]
```

---

### Paso 5: Autenticarse en Hugging Face

#### Opción A: CLI (recomendado)
```bash
huggingface-cli login
```

Te pedirá tu token. Puedes obtenerlo en:
https://huggingface.co/settings/tokens

#### Opción B: Token directo
Guarda tu token para usar con el script de upload.

---

### Paso 6: Subir a Hugging Face

```bash
python upload_to_huggingface.py \
  --model-dir ./trained_model \
  --repo-name distilbert-sentiment-imdb
```

**Con opciones adicionales:**
```bash
python upload_to_huggingface.py \
  --model-dir ./trained_model \
  --repo-name distilbert-sentiment-imdb \
  --organization tu-organizacion \
  --private
```

**Qué hace el script:**
1. ✅ Valida autenticación de HF
2. ✅ Crea repositorio (si no existe)
3. ✅ Genera Model Card automático con métricas
4. ✅ Sube todos los archivos del modelo
5. ✅ Muestra URL del modelo publicado

---

### Paso 7: Verificar en Hugging Face

Una vez subido, ve a:
```
https://huggingface.co/TU_USERNAME/distilbert-sentiment-imdb
```

**Deberías ver:**
- ✅ Model Card con métricas
- ✅ Archivos del modelo
- ✅ Widget de inferencia (puedes probar el modelo directamente)
- ✅ Ejemplos de código

---

### Paso 8: Usar el Modelo desde Hugging Face

Una vez publicado, cualquiera puede usarlo:

```python
from transformers import pipeline

# Cargar desde HF Hub
classifier = pipeline(
    "sentiment-analysis", 
    model="TU_USERNAME/distilbert-sentiment-imdb"
)

# Usar
result = classifier("I love this!")
print(result)
```

---

## 🎯 Flujo Completo (Comandos Secuenciales)

```bash
# 1. Entrenar modelo
python -m src.train \
  --config config.json \
  --output_dir ./trained_model \
  --run-name "production-v1"

# 2. Ver métricas (en otra terminal)
mlflow ui

# 3. Probar localmente
python -c "
from transformers import pipeline
clf = pipeline('sentiment-analysis', model='./trained_model')
print(clf('Amazing product!'))
"

# 4. Login a HF
huggingface-cli login

# 5. Subir a HF
python upload_to_huggingface.py \
  --model-dir ./trained_model \
  --repo-name distilbert-sentiment-imdb

# 6. Probar desde HF
python -c "
from transformers import pipeline
clf = pipeline('sentiment-analysis', model='TU_USERNAME/distilbert-sentiment-imdb')
print(clf('Amazing product!'))
"
```

---

## 📊 Entrenamiento en Progreso

### Cómo saber si está entrenando correctamente:

**Señales buenas ✅:**
- Loss disminuye con cada epoch
- Accuracy aumenta
- No hay errores en la consola
- MLflow muestra métricas actualizándose

**Señales de problemas ❌:**
- Loss no cambia (stuck)
- Accuracy ~0.50 (random)
- Errores de memoria (OOM)
- Entrenamiento muy lento

### Si el entrenamiento es muy lento en CPU:

1. **Usar config_rapido.json:**
   ```bash
   python -m src.train --config config_rapido.json --output_dir ./quick_model
   ```

2. **Reducir datos aún más:**
   Edita `config.json`:
   ```json
   "data": {
     "train_size": 1000,
     "eval_size": 200,
     "test_size": 100
   }
   ```

3. **Reducir épocas:**
   ```json
   "training": {
     "num_train_epochs": 2
   }
   ```

---

## 🔍 Monitoreo Durante Entrenamiento

### Ver logs en tiempo real:

El entrenamiento muestra:
```
Epoch 1/3:  50%|████████████▌            | 250/500 [15:23<15:22, 3.69s/it]
  loss: 0.3245
  eval_loss: 0.2876
  eval_accuracy: 0.8750
```

### Interpretar métricas:

- **loss**: Qué tan "equivocado" está el modelo (↓ mejor)
- **eval_loss**: Loss en validation set (↓ mejor)
- **eval_accuracy**: % de aciertos en validation (↑ mejor)

**Meta para buen modelo:**
- Loss final: < 0.3
- Accuracy final: > 0.85

---

## 💡 Mejores Prácticas

### 1. Versiona tus modelos
```bash
# Versión 1
python upload_to_huggingface.py \
  --model-dir ./trained_model \
  --repo-name distilbert-sentiment-v1

# Versión 2 (después de mejorar)
python upload_to_huggingface.py \
  --model-dir ./trained_model_v2 \
  --repo-name distilbert-sentiment-v2
```

### 2. Nombre descriptivo
Incluye información clave en el nombre:
- Modelo base: `distilbert`, `bert`, `roberta`
- Tarea: `sentiment`, `classification`
- Dataset: `imdb`, `amazon`, etc.
- Versión: `v1`, `v2`

Ejemplos:
- `distilbert-sentiment-imdb-v1`
- `bert-base-sentiment-amazon`
- `roberta-sentiment-multilingual`

### 3. Documenta en Model Card
El script genera automáticamente, pero puedes editar después en HF:
- Agrega ejemplos de uso
- Limitaciones conocidas
- Casos de uso recomendados

### 4. Privacidad
Para modelos experimentales:
```bash
python upload_to_huggingface.py \
  --model-dir ./trained_model \
  --repo-name my-experimental-model \
  --private
```

---

## 🐛 Troubleshooting

### Error: "Authentication required"
```bash
# Solución:
huggingface-cli login
# O provee token:
python upload_to_huggingface.py --model-dir ./model --repo-name my-model --token YOUR_TOKEN
```

### Error: "Repository already exists"
**No es problema.** El script actualiza el repo existente.

### Error: "Model files not found"
Verifica que el directorio tenga los archivos necesarios:
```bash
ls ./trained_model/
# Debe contener: model.safetensors, config.json, tokenizer files
```

### Modelo sube pero no funciona en HF
Verifica que el `model_info.json` se generó correctamente:
```bash
cat ./trained_model/model_info.json
```

---

## ✅ Checklist Final

Antes de subir a producción:

- [ ] Modelo entrenado sin errores
- [ ] Métricas > 85% accuracy
- [ ] Probado localmente y funciona
- [ ] Model Card completo
- [ ] Nombre descriptivo
- [ ] Ejemplos de uso documentados
- [ ] Subido a HF exitosamente
- [ ] Probado desde HF (pip install transformers y usar)

---

## 📚 Recursos

- **Hugging Face Hub Docs:** https://huggingface.co/docs/hub/
- **Model Card Guide:** https://huggingface.co/docs/hub/model-cards
- **Transformers Docs:** https://huggingface.co/docs/transformers/

---

**Última actualización:** 24 de octubre de 2025  
**Estado actual:** Entrenamiento en progreso... ⏳
