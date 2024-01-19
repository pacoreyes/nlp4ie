import random
import os
from pprint import pprint

import numpy as np
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer
from typing import Dict, Any, Union
from optuna import Trial

from lib.utils import load_jsonl_file

# Initialize label map and class names
LABEL_MAP = {"support": 0, "oppose": 1, "neutral": 2}

# Initialize constants
SEED = 42


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


def set_seed(seed_value):
  """Set seed for reproducibility."""
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  os.environ['PYTHONHASHSEED'] = str(seed_value)


def model_init(params: Dict[str, Any]) -> SetFitModel:
  params = params or {}
  max_iter = params.get("max_iter", 100)
  solver = params.get("solver", "liblinear")
  params = {
    "head_params": {
      "max_iter": max_iter,
      "solver": solver,
    }
  }
  model = SetFitModel.from_pretrained(model_id, **params)
  model.to(device)
  return model


def hp_space(trial: Trial) -> Dict[str, Union[float, int, str]]:
  return {
    "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-6, 1e-3, log=True),
    "num_epochs": trial.suggest_int("num_epochs", 1, 3),
    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    "seed": trial.suggest_int("seed", 1, 40),
    "max_iter": trial.suggest_int("max_iter", 50, 300),
    "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
  }


# Set seed for reproducibility
set_seed(SEED)

# Set device to CUDA, MPS, or CPU
device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")

# Initialize model
# model_id = "sentence-transformers/all-mpnet-base-v2"
model_id = "sentence-transformers/paraphrase-mpnet-base-v2"
# model_id = "BAAI/bge-small-en-v1.5"


# Load datasets
dataset_training_route = "shared_data/dataset_3_4_training_anonym.jsonl"
dataset_validation_route = "shared_data/dataset_3_5_validation_anonym.jsonl"
dataset_test_route = "shared_data/dataset_3_6_test_anonym.jsonl"
dataset_training = load_jsonl_file(dataset_training_route)
dataset_validation = load_jsonl_file(dataset_validation_route)
dataset_test = load_jsonl_file(dataset_test_route)

# Reverse label map from string to integer
dataset_training = [{"label": LABEL_MAP[datapoint["completion"]], "text": datapoint["prompt"]}
                    for datapoint in dataset_training]
dataset_validation = [{"label": LABEL_MAP[datapoint["completion"]], "text": datapoint["prompt"]}
                      for datapoint in dataset_validation]
dataset_test = [{"label": LABEL_MAP[datapoint["completion"]], "text": datapoint["prompt"]}
                for datapoint in dataset_test]

# Count and print class distribution
print("\nClass distribution:")
print(f"- support: {len(dataset_training)}")
print(f"- oppose: {len(dataset_validation)}")
print(f"- neutral: {len(dataset_test)}")

# Convert training data into a Dataset object
train_columns = {key: [dic[key] for dic in dataset_training] for key in dataset_training[0]}
train_dataset = Dataset.from_dict(train_columns)

# Convert validation data into a Dataset object
val_columns = {key: [dic[key] for dic in dataset_validation] for key in dataset_validation[0]}
validation_dataset = Dataset.from_dict(val_columns)

# Convert test data into Dataset object
test_columns = {key: [dic[key] for dic in dataset_test] for key in dataset_test[0]}
test_dataset = Dataset.from_dict(test_columns)

# print(len(train_dataset))

trainer = Trainer(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    model_init=model_init,
)
best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=20)

# Print best run
pprint(f"\nBest run: {best_run}")

trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
trainer.train()

# Evaluate model
metrics = trainer.evaluate()
pprint(f"\nMetrics: {metrics}")

# Save model
trainer.model.save_pretrained("models/3")

