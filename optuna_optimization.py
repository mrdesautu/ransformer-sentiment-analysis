#!/usr/bin/env python3
"""
Script de optimización automática de hiperparámetros usando Optuna + MLflow.
Encuentra automáticamente la mejor combinación de hiperparámetros.
"""

import json
import subprocess
import sys
import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
from pathlib import Path


def objective(trial):
    """
    Función objetivo para Optuna.
    Define el espacio de búsqueda de hiperparámetros.
    """
    
    # Definir espacio de búsqueda
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    num_epochs = trial.suggest_int("num_epochs", 2, 5)
    weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)
    
    # Crear configuración
    base_config = "config_rapido.json"
    with open(base_config, 'r') as f:
        config = json.load(f)
    
    # Modificar configuración
    config["training"]["learning_rate"] = learning_rate
    config["training"]["per_device_train_batch_size"] = batch_size
    config["training"]["per_device_eval_batch_size"] = batch_size * 2
    config["training"]["num_train_epochs"] = num_epochs
    config["training"]["weight_decay"] = weight_decay
    
    # Guardar configuración temporal
    Path("./experiments/optuna").mkdir(parents=True, exist_ok=True)
    config_path = f"./experiments/optuna/config_trial_{trial.number}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Ejecutar entrenamiento
    run_name = f"optuna-trial-{trial.number}"
    output_dir = f"./experiments/optuna/trial_{trial.number}"
    
    cmd = [
        sys.executable, "-m", "src.train",
        "--config", config_path,
        "--output_dir", output_dir,
        "--experiment-name", "optuna-optimization",
        "--run-name", run_name
    ]
    
    print(f"\n{'='*60}")
    print(f"🔬 Trial {trial.number}")
    print(f"  Learning Rate: {learning_rate:.2e}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Weight Decay: {weight_decay:.4f}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Trial {trial.number} falló")
        return 0.0  # Retornar valor bajo si falla
    
    # Extraer accuracy del output (esto es una simplificación)
    # En producción, deberías leer el resultado desde MLflow o un archivo
    try:
        # Buscar la línea con test_accuracy en el output
        for line in result.stdout.split('\n'):
            if 'eval_accuracy' in line.lower():
                # Extraer el valor (ajustar según tu formato de output)
                accuracy = float(line.split(':')[-1].strip())
                print(f"✅ Trial {trial.number} - Accuracy: {accuracy:.4f}")
                return accuracy
    except:
        pass
    
    # Si no se puede extraer, retornar valor por defecto
    return 0.5


def run_optuna_study(n_trials=10):
    """Ejecuta estudio de Optuna."""
    
    print("\n" + "="*70)
    print("🤖 OPTIMIZACIÓN AUTOMÁTICA DE HIPERPARÁMETROS CON OPTUNA")
    print("="*70)
    print(f"\nNúmero de trials: {n_trials}")
    print("Esto ejecutará múltiples experimentos para encontrar la mejor configuración\n")
    
    confirm = input("¿Continuar? (s/n): ")
    if confirm.lower() != 's':
        print("❌ Cancelado")
        return
    
    # Crear estudio de Optuna
    study = optuna.create_study(
        study_name="sentiment-hyperparameter-tuning",
        direction="maximize",  # Maximizar accuracy
        pruner=optuna.pruners.MedianPruner()  # Detener trials malos temprano
    )
    
    # Configurar callback de MLflow
    mlflc = MLflowCallback(
        tracking_uri="./mlruns",
        metric_name="accuracy"
    )
    
    # Ejecutar optimización
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflc])
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("📊 RESULTADOS DE LA OPTIMIZACIÓN")
    print("="*70)
    
    print("\n🏆 Mejores hiperparámetros:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n📈 Mejor accuracy: {study.best_value:.4f}")
    
    print("\n📋 Top 5 configuraciones:")
    trials_df = study.trials_dataframe().sort_values('value', ascending=False).head(5)
    print(trials_df[['number', 'value', 'params_learning_rate', 'params_batch_size', 
                     'params_num_epochs', 'params_weight_decay']])
    
    # Guardar mejores parámetros
    best_config_path = "./experiments/optuna/best_config.json"
    base_config = "config_rapido.json"
    with open(base_config, 'r') as f:
        config = json.load(f)
    
    config["training"]["learning_rate"] = study.best_params["learning_rate"]
    config["training"]["per_device_train_batch_size"] = study.best_params["batch_size"]
    config["training"]["per_device_eval_batch_size"] = study.best_params["batch_size"] * 2
    config["training"]["num_train_epochs"] = study.best_params["num_epochs"]
    config["training"]["weight_decay"] = study.best_params["weight_decay"]
    
    with open(best_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Mejor configuración guardada en: {best_config_path}")
    
    # Visualización (opcional)
    try:
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.savefig("./experiments/optuna/optimization_history.png")
        print("📊 Gráfico de optimización guardado en: ./experiments/optuna/optimization_history.png")
        
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.savefig("./experiments/optuna/param_importances.png")
        print("📊 Importancia de parámetros guardada en: ./experiments/optuna/param_importances.png")
    except:
        print("⚠️  No se pudieron generar visualizaciones (instala matplotlib)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimización de hiperparámetros con Optuna")
    parser.add_argument("--n-trials", type=int, default=10, help="Número de trials a ejecutar")
    
    args = parser.parse_args()
    
    run_optuna_study(args.n_trials)
    
    print("\n" + "="*70)
    print("📊 Para ver los resultados en MLflow:")
    print("   mlflow ui")
    print("   http://localhost:5000")
    print("="*70)
