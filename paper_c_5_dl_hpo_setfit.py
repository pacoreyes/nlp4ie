import random
import os
from pprint import pprint
from collections import Counter
import json

import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer

from lib.utils import load_jsonl_file

# Set the max_split_size_mb parameter
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch


# Initialize label map and class names
LABEL_MAP = {"support": 0, "oppose": 1}

# Initialize constants
SEED = 42


def set_seed(seed_value):
  """Set seed for reproducibility."""
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  os.environ['PYTHONHASHSEED'] = str(seed_value)


# Set seed for reproducibility
set_seed(SEED)


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


def model_init(params):
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


def hp_space(trial):
  return {
    "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-6, 1e-3, log=True),
    "num_epochs": trial.suggest_int("num_epochs", 1, 3),
    "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
    "seed": trial.suggest_int("seed", 1, 40),
    "max_iter": trial.suggest_int("max_iter", 50, 100),
    "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
  }


"""def load_data_lazy(file_path):
  # Lazy loading of data from a JSONL file.
  with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
      yield json.loads(line)


def process_data_lazy(data_generator):
  # Process data in a memory-efficient manner.
  for data in data_generator:
    if data["completion"] != "neutral":
      yield {"label": LABEL_MAP[data["completion"]], "text": data["prompt"]}"""


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


""" ###############################################################
Remove the "neutral" class from the three datasets
############################################################### """
dataset_training = [datapoint for datapoint in dataset_training if datapoint["completion"] != "neutral"]
dataset_validation = [datapoint for datapoint in dataset_validation if datapoint["completion"] != "neutral"]
dataset_test = [datapoint for datapoint in dataset_test if datapoint["completion"] != "neutral"]

dataset_test = dataset_test + dataset_validation
""" ############################################################ """

# Reverse label map from string to integer
dataset_training = [{"label": LABEL_MAP[datapoint["completion"]], "text": datapoint["prompt"]}
                    for datapoint in dataset_training]
dataset_validation = [{"label": LABEL_MAP[datapoint["completion"]], "text": datapoint["prompt"]}
                      for datapoint in dataset_validation]
dataset_test = [{"label": LABEL_MAP[datapoint["completion"]], "text": datapoint["prompt"]}
                for datapoint in dataset_test]

# Count and print class distribution
train_label_counts = Counter([datapoint['label'] for datapoint in dataset_training])
val_label_counts = Counter([datapoint['label'] for datapoint in dataset_validation])
test_label_counts = Counter([datapoint['label'] for datapoint in dataset_test])

print("\nClass distribution in training dataset:", train_label_counts)
print("Class distribution in validation dataset:", val_label_counts)
print("Class distribution in test dataset:", test_label_counts)

"""
# Replace your training dataset loading with lazy loading
dataset_training_generator = load_data_lazy(dataset_training_route)
dataset_training_processed = process_data_lazy(dataset_training_generator)

# Convert the generator to a list before creating a Dataset object for the training data
dataset_training_list = list(dataset_training_processed)
train_columns = {key: [dic[key] for dic in dataset_training_list] for key in dataset_training_list[0]}
train_dataset = Dataset.from_dict(train_columns)

# Replace your validation dataset loading with lazy loading
dataset_validation_generator = load_data_lazy(dataset_validation_route)
dataset_validation_processed = process_data_lazy(dataset_validation_generator)

# Convert the generator to a list before creating a Dataset object for the validation data
dataset_validation_list = list(dataset_validation_processed)
val_columns = {key: [dic[key] for dic in dataset_validation_list] for key in dataset_validation_list[0]}
validation_dataset = Dataset.from_dict(val_columns)

# Replace your test dataset loading with lazy loading
dataset_test_generator = load_data_lazy(dataset_test_route)
dataset_test_processed = process_data_lazy(dataset_test_generator)

# Convert the generator to a list before creating a Dataset object for the test data
dataset_test_list = list(dataset_test_processed)
test_columns = {key: [dic[key] for dic in dataset_test_list] for key in dataset_test_list[0]}
test_dataset = Dataset.from_dict(test_columns)
"""

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

# Run on the terminal before running this script:
# Mac
# export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
#
# set CUDA_LAUNCH_BLOCKING=1

