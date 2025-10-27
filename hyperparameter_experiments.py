#!/usr/bin/env python3
"""
Script to run experiments with different hyperparameters.
Allows testing multiple configurations and comparing results in MLflow.
"""

import json
import subprocess
import sys
from itertools import product
from pathlib import Path


# ==================== CONFIGURATIONS TO TEST ====================

# 1. Learning Rates
LEARNING_RATES = [
    2e-5,   # Default value
    3e-5,   # Higher
    5e-5,   # Even higher
    1e-5,   # Lower
]

# 2. Batch Sizes
BATCH_SIZES = [
    8,      # Default
    16,     # Larger (faster but more memory)
    4,      # Smaller (slower but less memory)
]

# 3. Number of epochs
NUM_EPOCHS = [
    2,      # Fast
    3,      # Default
    5,      # More training
]

# 4. Weight Decay (regularization)
WEIGHT_DECAYS = [
    0.01,   # Default
    0.001,  # Less regularization
    0.1,    # More regularization
]

# 5. Different models
MODELS = [
    "distilbert-base-uncased",              # Faster, fewer parameters
    "bert-base-uncased",                    # Larger, more accurate
    "roberta-base",  ]                       # BERT variant
    # "albert-base-v2",                     # Lighter


# ==================== FUNCTIONS ====================

def create_config(base_config_path, modifications, output_path):
    """Creates a modified configuration file."""
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # Apply modifications
    for key_path, value in modifications.items():
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value
    
    # Save configuration
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config


def run_experiment(config_path, run_name, experiment_name="hyperparameter-tuning", output_dir=None):
    """Runs a training experiment."""
    if output_dir is None:
        output_dir = f"./experiments/{run_name}"
    
    cmd = [
        sys.executable, "-m", "src.train",
        "--config", config_path,
        "--output_dir", output_dir,
        "--experiment-name", experiment_name,
        "--run-name", run_name
    ]
    
    print(f"\n{'='*60}")
    print(f"🚀 Running: {run_name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"✅ Experiment {run_name} completed successfully")
    else:
        print(f"❌ Experiment {run_name} failed")
    
    return result.returncode


# ==================== EXPERIMENTATION STRATEGIES ====================

def strategy_1_learning_rates():
    """Strategy 1: Compare different learning rates."""
    print("\n" + "="*70)
    print("📊 STRATEGY 1: Compare Learning Rates")
    print("="*70)
    
    base_config = "config_rapido.json"
    experiment_name = "learning-rate-comparison"
    
    for lr in LEARNING_RATES:
        run_name = f"lr-{lr:.0e}"
        config_path = f"./experiments/configs/config_{run_name}.json"
        
        # Create directory if it doesn't exist
        Path("./experiments/configs").mkdir(parents=True, exist_ok=True)
        
        # Create configuration
        modifications = {"training.learning_rate": lr}
        create_config(base_config, modifications, config_path)
        
        # Run experiment
        run_experiment(config_path, run_name, experiment_name)


def strategy_2_batch_sizes():
    """Strategy 2: Compare different batch sizes."""
    print("\n" + "="*70)
    print("📊 STRATEGY 2: Compare Batch Sizes")
    print("="*70)
    
    base_config = "config_rapido.json"
    experiment_name = "batch-size-comparison"
    
    for bs in BATCH_SIZES:
        run_name = f"bs-{bs}"
        config_path = f"./experiments/configs/config_{run_name}.json"
        
        Path("./experiments/configs").mkdir(parents=True, exist_ok=True)
        
        modifications = {
            "training.per_device_train_batch_size": bs,
            "training.per_device_eval_batch_size": bs * 2  # Eval can be larger
        }
        create_config(base_config, modifications, config_path)
        
        run_experiment(config_path, run_name, experiment_name)


def strategy_3_epochs():
    """Estrategia 3: Comparar diferentes números de épocas."""
    print("\n" + "="*70)
    print("📊 ESTRATEGIA 3: Comparar Número de Épocas")
    print("="*70)
    
    base_config = "config_rapido.json"
    experiment_name = "epochs-comparison"
    
    for epochs in NUM_EPOCHS:
        run_name = f"epochs-{epochs}"
        config_path = f"./experiments/configs/config_{run_name}.json"
        
        Path("./experiments/configs").mkdir(parents=True, exist_ok=True)
        
        modifications = {"training.num_train_epochs": epochs}
        create_config(base_config, modifications, config_path)
        
        run_experiment(config_path, run_name, experiment_name)


def strategy_4_regularization():
    """Estrategia 4: Comparar diferentes niveles de regularización."""
    print("\n" + "="*70)
    print("📊 ESTRATEGIA 4: Comparar Weight Decay (Regularización)")
    print("="*70)
    
    base_config = "config_rapido.json"
    experiment_name = "regularization-comparison"
    
    for wd in WEIGHT_DECAYS:
        run_name = f"wd-{wd}"
        config_path = f"./experiments/configs/config_{run_name}.json"
        
        Path("./experiments/configs").mkdir(parents=True, exist_ok=True)
        
        modifications = {"training.weight_decay": wd}
        create_config(base_config, modifications, config_path)
        
        run_experiment(config_path, run_name, experiment_name)


def strategy_5_models():
    """Estrategia 5: Comparar diferentes modelos."""
    print("\n" + "="*70)
    print("📊 ESTRATEGIA 5: Comparar Diferentes Modelos")
    print("="*70)
    print("⚠️  ADVERTENCIA: Esto puede tardar mucho tiempo")
    
    base_config = "config_rapido.json"
    experiment_name = "model-comparison"
    
    for model in MODELS:
        model_short = model.split('/')[-1].replace('-', '_')
        run_name = f"model-{model_short}"
        config_path = f"./experiments/configs/config_{run_name}.json"
        
        Path("./experiments/configs").mkdir(parents=True, exist_ok=True)
        
        modifications = {"model.name": model}
        create_config(base_config, modifications, config_path)
        
        run_experiment(config_path, run_name, experiment_name)


def strategy_6_grid_search():
    """Estrategia 6: Grid Search (combinar múltiples parámetros)."""
    print("\n" + "="*70)
    print("📊 ESTRATEGIA 6: Grid Search (Combinación de Parámetros)")
    print("="*70)
    print("⚠️  ADVERTENCIA: Esto ejecutará MUCHOS experimentos")
    
    # Usar subconjunto más pequeño para grid search
    lr_subset = [2e-5, 5e-5]
    bs_subset = [8, 16]
    
    total_experiments = len(lr_subset) * len(bs_subset)
    print(f"📈 Total de experimentos: {total_experiments}")
    
    confirm = input("\n¿Continuar? (s/n): ")
    if confirm.lower() != 's':
        print("❌ Cancelado")
        return
    
    base_config = "config_rapido.json"
    experiment_name = "grid-search"
    
    for lr, bs in product(lr_subset, bs_subset):
        run_name = f"lr-{lr:.0e}_bs-{bs}"
        config_path = f"./experiments/configs/config_{run_name}.json"
        
        Path("./experiments/configs").mkdir(parents=True, exist_ok=True)
        
        modifications = {
            "training.learning_rate": lr,
            "training.per_device_train_batch_size": bs,
            "training.per_device_eval_batch_size": bs * 2
        }
        create_config(base_config, modifications, config_path)
        
        run_experiment(config_path, run_name, experiment_name)


def strategy_7_best_practices():
    """Estrategia 7: Configuraciones basadas en mejores prácticas."""
    print("\n" + "="*70)
    print("📊 ESTRATEGIA 7: Configuraciones de Mejores Prácticas")
    print("="*70)
    
    base_config = "config_rapido.json"
    experiment_name = "best-practices"
    
    # Configuración conservadora (más estable)
    conservative_mods = {
        "training.learning_rate": 1e-5,
        "training.per_device_train_batch_size": 8,
        "training.weight_decay": 0.1,
        "training.num_train_epochs": 5
    }
    
    # Configuración agresiva (más rápido)
    aggressive_mods = {
        "training.learning_rate": 5e-5,
        "training.per_device_train_batch_size": 16,
        "training.weight_decay": 0.001,
        "training.num_train_epochs": 2
    }
    
    # Configuración balanceada
    balanced_mods = {
        "training.learning_rate": 2e-5,
        "training.per_device_train_batch_size": 8,
        "training.weight_decay": 0.01,
        "training.num_train_epochs": 3
    }
    
    configs = [
        ("conservative", conservative_mods),
        ("aggressive", aggressive_mods),
        ("balanced", balanced_mods)
    ]
    
    for name, mods in configs:
        run_name = f"strategy-{name}"
        config_path = f"./experiments/configs/config_{run_name}.json"
        
        Path("./experiments/configs").mkdir(parents=True, exist_ok=True)
        
        create_config(base_config, mods, config_path)
        run_experiment(config_path, run_name, experiment_name)


# ==================== MENÚ PRINCIPAL ====================

def show_menu():
    """Muestra el menú de opciones."""
    print("\n" + "="*70)
    print("🔬 EXPERIMENTACIÓN CON HIPERPARÁMETROS - MLflow")
    print("="*70)
    print("\nEstrategias disponibles:")
    print("  1. Comparar Learning Rates (4 experimentos)")
    print("  2. Comparar Batch Sizes (3 experimentos)")
    print("  3. Comparar Número de Épocas (3 experimentos)")
    print("  4. Comparar Weight Decay (3 experimentos)")
    print("  5. Comparar Modelos Diferentes (depende de MODELS)")
    print("  6. Grid Search - Combinaciones (⚠️  MUCHOS experimentos)")
    print("  7. Mejores Prácticas - 3 configuraciones (3 experimentos)")
    print("  8. Ejecutar TODO (⚠️  MUCHO TIEMPO)")
    print("  0. Salir")
    print("\nDespués de ejecutar, usa: mlflow ui")
    print("="*70)


def main():
    """Función principal."""
    strategies = {
        "1": strategy_1_learning_rates,
        "2": strategy_2_batch_sizes,
        "3": strategy_3_epochs,
        "4": strategy_4_regularization,
        "5": strategy_5_models,
        "6": strategy_6_grid_search,
        "7": strategy_7_best_practices,
    }
    
    while True:
        show_menu()
        choice = input("\nSelecciona una estrategia (0-8): ").strip()
        
        if choice == "0":
            print("\n👋 ¡Hasta luego!")
            break
        
        elif choice == "8":
            confirm = input("\n⚠️  Esto ejecutará TODOS los experimentos. ¿Continuar? (s/n): ")
            if confirm.lower() == 's':
                for strategy_func in strategies.values():
                    strategy_func()
        
        elif choice in strategies:
            strategies[choice]()
        
        else:
            print("❌ Opción inválida")
        
        input("\n✅ Presiona Enter para continuar...")
    
    print("\n" + "="*70)
    print("📊 Para ver los resultados:")
    print("   1. Ejecuta: mlflow ui")
    print("   2. Abre: http://localhost:5000")
    print("   3. Compara experimentos seleccionando múltiples runs")
    print("="*70)


if __name__ == "__main__":
    main()
