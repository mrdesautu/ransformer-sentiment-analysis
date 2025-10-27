# 🧪 Guía Completa de Experimentación con Hiperparámetros

## 📋 Índice
1. [Scripts Disponibles](#scripts-disponibles)
2. [Estrategias de Experimentación](#estrategias-de-experimentación)
3. [Hiperparámetros Clave](#hiperparámetros-clave)
4. [Cómo Ejecutar Experimentos](#cómo-ejecutar-experimentos)
5. [Análisis de Resultados](#análisis-de-resultados)
6. [Mejores Prácticas](#mejores-prácticas)

---

## 🛠️ Scripts Disponibles

### 1. `hyperparameter_experiments.py` - Experimentación Manual
Script interactivo para probar diferentes configuraciones de hiperparámetros.

**Uso:**
```bash
python hyperparameter_experiments.py
```

**Estrategias incluidas:**
- Comparar Learning Rates
- Comparar Batch Sizes
- Comparar Número de Épocas
- Comparar Weight Decay (regularización)
- Comparar Modelos Diferentes
- Grid Search (combinaciones)
- Mejores Prácticas (3 configuraciones probadas)

### 2. `optuna_optimization.py` - Optimización Automática
Búsqueda automática de hiperparámetros usando Optuna.

**Uso:**
```bash
# 10 trials (por defecto)
python optuna_optimization.py

# 20 trials (más exhaustivo)
python optuna_optimization.py --n-trials 20
```

**Qué optimiza:**
- Learning rate (1e-5 a 1e-4)
- Batch size (4, 8, 16)
- Número de épocas (2 a 5)
- Weight decay (0.001 a 0.1)

### 3. `analyze_results.py` - Análisis Automático
Genera reportes y gráficos comparativos de tus experimentos.

**Uso:**
```bash
python analyze_results.py
```

**Genera:**
- Estadísticas de todas las métricas
- Top 5 mejores configuraciones
- Gráficos de comparación
- Matriz de correlaciones
- Reportes en texto

---

## 🎯 Hiperparámetros Clave

### 1. Learning Rate (tasa de aprendizaje)
**¿Qué hace?** Controla qué tan grande es el paso en cada actualización del modelo.

**Valores típicos:**
- `1e-5` - Muy conservador, entrenamiento lento pero estable
- `2e-5` - **Valor por defecto recomendado** para BERT/DistilBERT
- `3e-5` - Más agresivo, converge más rápido
- `5e-5` - Muy agresivo, puede ser inestable

**Cómo afecta:**
- ⬇️ Muy bajo: entrenamiento muy lento, puede quedar atascado
- ⬆️ Muy alto: entrenamiento inestable, puede diverger

**Experimento sugerido:**
```bash
python hyperparameter_experiments.py
# Selecciona opción 1: Comparar Learning Rates
```

---

### 2. Batch Size (tamaño de lote)
**¿Qué hace?** Número de ejemplos procesados antes de actualizar los pesos.

**Valores típicos:**
- `4` - Poco uso de memoria, entrenamiento más ruidoso
- `8` - **Valor por defecto recomendado**, buen balance
- `16` - Más rápido, requiere más memoria
- `32` - Muy rápido, requiere GPU con mucha memoria

**Cómo afecta:**
- ⬇️ Más pequeño: menos memoria, más ruido (puede ayudar a generalizar), más lento
- ⬆️ Más grande: más memoria, más estable, más rápido

**Experimento sugerido:**
```bash
python hyperparameter_experiments.py
# Selecciona opción 2: Comparar Batch Sizes
```

---

### 3. Número de Épocas
**¿Qué hace?** Cuántas veces el modelo ve todo el dataset.

**Valores típicos:**
- `2` - Entrenamiento rápido, puede no converger
- `3` - **Valor por defecto recomendado**
- `5` - Más entrenamiento, riesgo de overfitting
- `10+` - Solo para datasets muy grandes

**Cómo afecta:**
- ⬇️ Muy pocas: modelo no aprende suficiente (underfitting)
- ⬆️ Muchas: modelo memoriza datos (overfitting)

**Señales de overfitting:**
- Training accuracy sube, validation accuracy baja
- Loss de validación empieza a subir

**Experimento sugerido:**
```bash
python hyperparameter_experiments.py
# Selecciona opción 3: Comparar Número de Épocas
```

---

### 4. Weight Decay (regularización L2)
**¿Qué hace?** Penaliza pesos grandes para prevenir overfitting.

**Valores típicos:**
- `0.001` - Poca regularización
- `0.01` - **Valor por defecto recomendado**
- `0.1` - Mucha regularización

**Cómo afecta:**
- ⬇️ Muy bajo: modelo puede overfittear
- ⬆️ Muy alto: modelo underfittea, no aprende bien

**Experimento sugerido:**
```bash
python hyperparameter_experiments.py
# Selecciona opción 4: Comparar Weight Decay
```

---

### 5. Modelo Base
**¿Qué hace?** Arquitectura pre-entrenada a usar.

**Opciones:**
- `distilbert-base-uncased` - **Más rápido**, 66M parámetros, ~250MB
- `bert-base-uncased` - Más preciso, 110M parámetros, ~420MB
- `roberta-base` - Variante de BERT, 125M parámetros
- `albert-base-v2` - Más ligero, 12M parámetros

**Trade-offs:**
- ⚡ Velocidad vs 🎯 Precisión
- 💾 Memoria vs 📊 Accuracy

**Experimento sugerido:**
```bash
python hyperparameter_experiments.py
# Selecciona opción 5: Comparar Modelos Diferentes
```

---

## 🚀 Cómo Ejecutar Experimentos

### Flujo de Trabajo Recomendado

#### **Paso 1: Baseline**
Entrena un modelo con configuración por defecto para tener una referencia.

```bash
python -m src.train --config config_rapido.json --run-name "baseline"
```

#### **Paso 2: Exploración Rápida**
Prueba diferentes learning rates (el parámetro más importante).

```bash
python hyperparameter_experiments.py
# Selecciona opción 1
```

#### **Paso 3: Ajuste Fino**
Prueba combinaciones con grid search.

```bash
python hyperparameter_experiments.py
# Selecciona opción 6
```

#### **Paso 4: Optimización Automática** (Opcional)
Deja que Optuna encuentre la mejor configuración.

```bash
python optuna_optimization.py --n-trials 15
```

#### **Paso 5: Análisis**
Compara todos los resultados.

```bash
python analyze_results.py
```

#### **Paso 6: UI de MLflow**
Visualiza los experimentos interactivamente.

```bash
mlflow ui
# Abre http://localhost:5000
```

---

## 📊 Análisis de Resultados

### En la UI de MLflow

1. **Ver todos los runs:**
   - Abre http://localhost:5000
   - Selecciona el experimento
   - Ordena por `test_accuracy` (click en la columna)

2. **Comparar runs:**
   - Selecciona 2+ runs (checkbox)
   - Click en "Compare"
   - Ve diferencias en parámetros y métricas

3. **Gráficos:**
   - En "Charts", crea gráficos personalizados
   - Eje X: learning_rate, Eje Y: test_accuracy
   - Agrupa por batch_size

### Con analyze_results.py

```bash
python analyze_results.py
```

Esto genera:
- `./analysis/<experimento>/accuracy_vs_lr.png` - Accuracy vs Learning Rate
- `./analysis/<experimento>/accuracy_vs_batch.png` - Accuracy vs Batch Size
- `./analysis/<experimento>/correlation_heatmap.png` - Correlaciones
- `./analysis/<experimento>/metrics_comparison.png` - Comparación de métricas
- `./analysis/<experimento>/report.txt` - Reporte en texto

---

## 🎓 Mejores Prácticas

### 1. **Empieza simple, itera después**
```bash
# Primero: baseline
python -m src.train --config config_rapido.json --run-name "baseline"

# Luego: variar UN parámetro a la vez
# Learning rate
# Batch size
# Etc.
```

### 2. **Usa nombres descriptivos**
```bash
python -m src.train \
  --run-name "distilbert-lr3e5-bs16-ep3" \
  --experiment-name "production-tuning"
```

### 3. **Documenta tus hallazgos**
Crea un `EXPERIMENT_LOG.md`:
```markdown
## 2025-10-24: Learning Rate Comparison

**Objetivo:** Encontrar mejor learning rate

**Resultados:**
- lr=2e-5: accuracy=0.85
- lr=3e-5: accuracy=0.87 ✅ MEJOR
- lr=5e-5: accuracy=0.83 (inestable)

**Conclusión:** Usar 3e-5 para futuros experimentos
```

### 4. **Prioriza parámetros por impacto**

**Alto impacto** (probar primero):
1. Learning rate
2. Número de épocas
3. Modelo base

**Medio impacto**:
4. Batch size
5. Weight decay

**Bajo impacto** (ajuste fino):
6. Warmup steps
7. Learning rate scheduler

### 5. **Valida con datos de test separados**
Asegúrate de que `config.json` tenga:
```json
"data": {
  "train_size": 4000,
  "eval_size": 1000,  // Para early stopping
  "test_size": 500     // Para evaluación final
}
```

### 6. **Detecta overfitting**
En MLflow, compara:
- `eval_accuracy` vs `test_accuracy`
- Si `eval_accuracy` >> `test_accuracy`: overfitting

Soluciones:
- Aumentar `weight_decay`
- Reducir épocas
- Usar más datos de entrenamiento

---

## 🔍 Interpretación de Métricas

### Accuracy (exactitud)
- **Qué es:** % de predicciones correctas
- **Objetivo:** Maximizar
- **Bueno:** >0.85 para sentiment analysis en IMDB
- **Excelente:** >0.90

### F1-Score
- **Qué es:** Balance entre precision y recall
- **Objetivo:** Maximizar
- **Mejor que accuracy cuando:** clases desbalanceadas

### Loss (pérdida)
- **Qué es:** Qué tan "equivocado" está el modelo
- **Objetivo:** Minimizar
- **Típico:** 0.2-0.5 para modelos buenos

### Precision vs Recall
- **Precision:** De las predicciones positivas, ¿cuántas son correctas?
- **Recall:** De los casos positivos reales, ¿cuántos detectamos?

---

## 🎯 Ejemplos de Escenarios

### Escenario 1: "Mi modelo no mejora"
**Síntomas:**
- Accuracy se queda en ~0.50 (random)
- Loss no baja

**Posibles causas:**
- Learning rate muy bajo → Prueba 3e-5 o 5e-5
- Muy pocas épocas → Aumenta a 5
- Modelo muy simple → Prueba `bert-base` en vez de `distilbert`

**Solución:**
```bash
python hyperparameter_experiments.py
# Opción 7: Mejores Prácticas - configuración "aggressive"
```

### Escenario 2: "Overfitting claro"
**Síntomas:**
- Training accuracy: 0.95
- Test accuracy: 0.75

**Soluciones:**
```bash
# Aumentar regularización
# Modifica config.json: "weight_decay": 0.1

# Reducir épocas
# Modifica config.json: "num_train_epochs": 2

# Entrenar
python -m src.train --config config.json --run-name "fix-overfit"
```

### Escenario 3: "Quiero el mejor modelo posible"
**Estrategia:**
```bash
# 1. Optimización automática
python optuna_optimization.py --n-trials 20

# 2. Tomar mejores parámetros de Optuna
# (se guardan en ./experiments/optuna/best_config.json)

# 3. Entrenar con dataset completo
python -m src.train \
  --config ./experiments/optuna/best_config.json \
  --run-name "final-model-full-data"

# 4. Evaluar
mlflow ui
```

---

## 📚 Recursos Adicionales

- **Hugging Face Training Tips:** https://huggingface.co/docs/transformers/training
- **Learning Rate Finder:** https://arxiv.org/abs/1506.01186
- **Optuna Documentation:** https://optuna.readthedocs.io/

---

## ✅ Checklist de Experimentación

Antes de decidir que tu modelo está "listo":

- [ ] Entrenaste un baseline
- [ ] Probaste al menos 3 learning rates diferentes
- [ ] Verificaste que no hay overfitting
- [ ] Comparaste al menos 2 modelos base
- [ ] Evaluaste en un test set separado
- [ ] Documentaste tus resultados
- [ ] Guardaste el mejor modelo en MLflow Model Registry

---

**Última actualización:** 24 de octubre de 2025  
**Autor:** Configuración automática MLflow  
**Versión:** 1.0
