# Comandos para diferentes datasets

# Dataset Amazon (productos)
"/Users/martinrodrigomorales/Desktop/Proyectos Banca/Transformer/.venv/bin/python" -m src.train --config config_amazon.json --output_dir ./modelo_amazon

# Dataset SST-2 (rápido)
echo '{
  "model": {"name": "distilbert-base-uncased", "num_labels": 2, "max_length": 128},
  "training": {"output_dir": "./results", "learning_rate": 3e-5, "per_device_train_batch_size": 16, "num_train_epochs": 1, "eval_strategy": "epoch", "save_strategy": "epoch"},
  "data": {"dataset_name": "sst2", "train_size": 1000, "eval_size": 200, "test_size": 100}
}' > config_sst2.json

"/Users/martinrodrigomorales/Desktop/Proyectos Banca/Transformer/.venv/bin/python" -m src.train --config config_sst2.json --output_dir ./modelo_sst2

# Dataset personalizado (tu propio CSV)
echo '{
  "model": {"name": "distilbert-base-uncased", "num_labels": 2, "max_length": 256},
  "data": {"dataset_name": "csv", "data_files": {"train": "mi_dataset.csv"}, "train_size": 1000}
}' > config_custom.json