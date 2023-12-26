import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import (precision_recall_fscore_support,
                             accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix)
from sklearn.model_selection import train_test_split

from db import spreadsheet_4
from lib.utils import read_from_google_sheet

# Initialize label map and class names
LABEL_MAP = {"support": 0, "oppose": 1}
class_names = list(LABEL_MAP.keys())

# Reversed Label Map
REVERSED_LABEL_MAP = {0: "support", 1: "oppose"}

# Initialize constants
SEED = 42

# Hyperparameters
BODY_LEARNING_RATE = 2.5e-5
HEAD_LEARNING_RATE = 2.5e-3
BATCH_SIZE = 16
NUM_EPOCHS = 3
L2_WEIGHT = 0.01


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
  # np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def compute_metrics(y_pred, y_test):
  accuracy = accuracy_score(y_test, y_pred)
  precision, recall, f1, _ = precision_recall_fscore_support(y_pred, y_test, average='binary')
  auc = roc_auc_score(y_pred, y_test)
  mcc = matthews_corrcoef(y_pred, y_test)
  cm = confusion_matrix(y_pred, y_test)

  return {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'auc': auc,
    'mcc': mcc,
    'confusion_matrix': cm
  }


# Set device to CUDA, MPS, or CPU
device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")

# model_id = "sentence-transformers/all-mpnet-base-v2"
model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# Move model to device
model.to(device)

# Set seed for reproducibility
set_seed(SEED)

dataset = read_from_google_sheet(spreadsheet_4, "dataset_3")

# Reverse label map from string to integer
dataset = [{"label": LABEL_MAP[datapoint["label"]], "text": datapoint["text"]} for datapoint in dataset]

counter = Counter([item['label'] for item in dataset])

print("\nClass distribution:")
print(f"- support: {counter[0]}")
print(f"- oppose: {counter[1]}")

# sentences = [entry["text"] for entry in dataset]
labels = [entry["label"] for entry in dataset]

# First split: 80% training, 20% temporary test
train_data, temp_test_data, train_labels, temp_test_labels = train_test_split(
    dataset, labels, test_size=0.2, random_state=SEED, stratify=labels)

# Second split of the temporary test data into validation and test sets (each 10% of the original)
validation_data, test_data, validation_labels, test_labels = train_test_split(
  temp_test_data, temp_test_labels, test_size=0.5, random_state=SEED, stratify=temp_test_labels)

# Convert training data into a Dataset object
train_columns = {key: [dic[key] for dic in train_data] for key in train_data[0]}
train_dataset = Dataset.from_dict(train_columns)

# Convert validation data into a Dataset object
val_columns = {key: [dic[key] for dic in validation_data] for key in validation_data[0]}
validation_dataset = Dataset.from_dict(val_columns)

# Convert test data into Dataset object
test_columns = {key: [dic[key] for dic in test_data] for key in test_data[0]}
test_dataset = Dataset.from_dict(test_columns)

args = TrainingArguments(
  batch_size=BATCH_SIZE,
  num_epochs=NUM_EPOCHS,
  end_to_end=True,
  body_learning_rate=BODY_LEARNING_RATE,
  head_learning_rate=HEAD_LEARNING_RATE,
  l2_weight=L2_WEIGHT,
  # evaluation_strategy="epoch",
  # load_best_model_at_end=True,
  evaluation_strategy="steps",
  eval_steps=100
)

# Training Loop
trainer = Trainer(
  model=model,
  args=args,
  train_dataset=train_dataset,
  eval_dataset=validation_dataset,
  metric=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate(test_dataset)
print(metrics)

df_cm = pd.DataFrame(metrics['confusion_matrix'], index=class_names, columns=class_names)

print("\nModel: SetFit\n")
print(f"- Accuracy: {metrics['accuracy']:.3f}")
print(f"- Precision: {np.mean(metrics['precision']):.3f}")
print(f"- Recall: {np.mean(metrics['recall']):.3f}")
print(f"- F1 Score: {np.mean(metrics['f1']):.3f}")
print(f"- AUC-ROC: {metrics['auc']:.3f}")
print(f"- Matthews Correlation Coefficient (MCC): {metrics['mcc']:.3f}")
print(f"- Confusion Matrix:")
print(df_cm)
print()

# Performing inference on the test set returning the scores
predictions = model.predict([
  test_data[0]["text"],
  test_data[1]["text"],
  test_data[2]["text"],
])

"""# Assuming predictions return probabilities for each class
all_predictions = model.predict([example["text"] for example in test_data])

# Collect scores - assuming binary classification and interest in the probability of class 1 ("oppose")
scores = [pred[1] for pred in all_predictions]  # This assumes the second entry is the probability of class 'oppose'

# Print or process the scores
for i, score in enumerate(scores):
    print(f"Example {i} - Score: {score}")"""
