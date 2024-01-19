import random
import os
# from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.manifold import TSNE
from sklearn.metrics import (precision_recall_fscore_support,
                             accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix)
# from sklearn.preprocessing import label_binarize
# from sklearn.model_selection import train_test_split
from transformers import TrainerCallback

from lib.utils import load_jsonl_file

# Initialize label map and class names
LABEL_MAP = {"support": 0, "oppose": 1, "neutral": 2}
class_names = list(LABEL_MAP.keys())

# Initialize constants
SEED = 42
MODEL_SEED = 42

# Hyperparameters
BODY_LEARNING_RATE = 0.00010214746757835319
HEAD_LEARNING_RATE = 0.005
BATCH_SIZE = 32
NUM_EPOCHS = 2
MAX_ITER = 191
L2_WEIGHT = 0.01
SOLVER = "liblinear"


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


def compute_metrics(y_pred, y_test):
  accuracy = accuracy_score(y_test, y_pred)
  precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
  # auc = roc_auc_score(y_test, y_pred, multi_class='ovo')
  mcc = matthews_corrcoef(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)

  return {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    # 'auc': auc,
    'mcc': mcc,
    'confusion_matrix': cm
  }


def model_init():
  params = {
    "head_params": {
      "max_iter": MAX_ITER,
      "solver": SOLVER,
    }
  }
  _model = SetFitModel.from_pretrained(model_id, **params)
  _model.to(device)
  return _model


# Set device to CUDA, MPS, or CPU
device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")

model_id = "sentence-transformers/all-mpnet-base-v2"
# model_id = "sentence-transformers/paraphrase-mpnet-base-v2"
# model_id = "BAAI/bge-small-en-v1.5"
model = SetFitModel.from_pretrained(
  model_id,
)

# Move model to device
model.to(device)

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
# counter = Counter([item['label'] for item in dataset])
print("\nDataset:")
print(f"- training: {len(dataset_training)}")
print(f"- validation: {len(dataset_validation)}")
print(f"- test: {len(dataset_test)}")

# sentences = [entry["text"] for entry in dataset]
# labels = [entry["label"] for entry in dataset]

"""# First split: 80% training, 20% temporary test
train_data, temp_test_data, train_labels, temp_test_labels = train_test_split(
  dataset, labels, test_size=0.2, random_state=SEED, stratify=labels)

# Second split of the temporary test data into validation and test sets (each 10% of the original)
validation_data, test_data, validation_labels, test_labels = train_test_split(
  temp_test_data, temp_test_labels, test_size=0.5, random_state=SEED, stratify=temp_test_labels)
"""

num_support_test = len([item for item in dataset_test if item["label"] == 0])
num_oppose_test = len([item for item in dataset_test if item["label"] == 1])
num_neutral_test = len([item for item in dataset_test if item["label"] == 2])

print(f"\nTest set class distribution:")
print(f"- support: {num_support_test}")
print(f"- oppose: {num_oppose_test}")
print(f"- neutral: {num_neutral_test}")

# Convert training data into a Dataset object
train_columns = {key: [dic[key] for dic in dataset_training] for key in dataset_training[0]}
train_dataset = Dataset.from_dict(train_columns)

# Convert validation data into a Dataset object
val_columns = {key: [dic[key] for dic in dataset_validation] for key in dataset_validation[0]}
validation_dataset = Dataset.from_dict(val_columns)

# Convert test data into Dataset object
test_columns = {key: [dic[key] for dic in dataset_test] for key in dataset_test[0]}
test_dataset = Dataset.from_dict(test_columns)


class LossPlotCallback(TrainerCallback):
  """Callback that records and prints loss after each epoch for plotting later."""

  def __init__(self):
    self.train_losses = []
    self.eval_losses = []

  def on_epoch_end(self, args, state, control, **kwargs):
    # Compute average training loss over the epoch
    train_loss = np.mean(state.log_history[-1]['embedding_loss'])
    self.train_losses.append(train_loss)
    print(f"Average Training Loss for Epoch {state.epoch}: {train_loss}")

  def on_evaluate(self, args, state, control, **kwargs):
    # Assuming eval loss is recorded at the end of each evaluation in the log_history
    eval_loss = state.log_history[-1]['eval_embedding_loss']
    self.eval_losses.append(eval_loss)
    print(f"Average Validation Loss after Epoch {state.epoch}: {eval_loss}")


# class EmbeddingPlotCallback(TrainerCallback):
  # Simple embedding plotting callback that plots the tSNE of the training and evaluation datasets
  # throughout training."""

"""  def on_evaluate(self, args, state, control, **kwargs):
    train_embeddings = model.encode(train_dataset["text"])
    eval_embeddings = model.encode(validation_dataset["text"])

    # Determine dataset size to adjust perplexity
    eval_dataset_size = len(validation_dataset["text"])
    perplexity_value = min(30, max(5, int(eval_dataset_size / 2)))

    fig, (train_ax, eval_ax) = plt.subplots(ncols=2)

    train_x = TSNE(n_components=2, perplexity=perplexity_value).fit_transform(train_embeddings)
    train_ax.scatter(*train_x.T, c=train_dataset["label"], label=train_dataset["label"])
    train_ax.set_title("Training embeddings")

    eval_x = TSNE(n_components=2, perplexity=perplexity_value).fit_transform(eval_embeddings)
    eval_ax.scatter(*eval_x.T, c=validation_dataset["label"], label=validation_dataset["label"])
    eval_ax.set_title("Evaluation embeddings")

    fig.suptitle(f"tSNE of training and evaluation embeddings at step {state.global_step} of {state.max_steps}.")
    fig.savefig(f"images/step_{state.global_step}.png")"""


class EmbeddingPlotCallback(TrainerCallback):
  """Simple embedding plotting callback that plots the tSNE of the training and evaluation datasets
  # throughout training."""

  def on_evaluate(self, args, state, control, **kwargs):
    train_embeddings = model.encode(train_dataset["text"])
    eval_embeddings = model.encode(validation_dataset["text"])

    # Determine dataset size to adjust perplexity
    eval_dataset_size = len(validation_dataset["text"])
    perplexity_value = min(30, max(5, int(eval_dataset_size / 2)))

    # Define a color map for different classes
    color_map = {0: 'red', 1: 'blue', 2: 'green'}  # Update this with your class colors

    # Create subplots
    fig, (train_ax, eval_ax) = plt.subplots(ncols=2)

    # Plot training embeddings
    train_x = TSNE(n_components=2, perplexity=perplexity_value).fit_transform(train_embeddings)
    scatter = train_ax.scatter(*train_x.T, c=[color_map[label] for label in train_dataset["label"]])
    train_ax.set_title("Training embeddings")
    train_ax.set_xlabel("Component 1")
    train_ax.set_ylabel("Component 2")

    # Plot evaluation embeddings
    eval_x = TSNE(n_components=2, perplexity=perplexity_value).fit_transform(eval_embeddings)
    scatter = eval_ax.scatter(*eval_x.T, c=[color_map[label] for label in validation_dataset["label"]])
    eval_ax.set_title("Evaluation embeddings")
    eval_ax.set_xlabel("Component 1")
    eval_ax.set_ylabel("Component 2")

    # Add a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                          markerfacecolor=color, markersize=10) for class_name, color in color_map.items()]
    fig.legend(handles=handles, loc='upper right')

    # Add overall title and save the figure
    fig.suptitle(f"tSNE of training and evaluation embeddings at step {state.global_step} of {state.max_steps}.")
    fig.savefig(f"images/step_{state.global_step}.png")


arguments = TrainingArguments(
  batch_size=BATCH_SIZE,
  num_epochs=NUM_EPOCHS,
  end_to_end=True,
  body_learning_rate=BODY_LEARNING_RATE,
  # head_learning_rate=HEAD_LEARNING_RATE,
  # l2_weight=L2_WEIGHT,
  evaluation_strategy="epoch",
  eval_steps=20,
  # num_iterations=20,
  seed=MODEL_SEED,
)

# Initialize callbacks
embedding_plot_callback = EmbeddingPlotCallback()
loss_plot_callback = LossPlotCallback()


# Training Loop
trainer = Trainer(
  # model=model,
  model_init=model_init,
  args=arguments,
  train_dataset=train_dataset,
  eval_dataset=validation_dataset,
  callbacks=[embedding_plot_callback, loss_plot_callback],
  metric=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate(test_dataset)
# print(metrics)

df_cm = pd.DataFrame(metrics['confusion_matrix'], index=class_names, columns=class_names)

print(f"\nModel: SetFit ({model_id})\n")
print(f"- Accuracy: {metrics['accuracy']:.3f}")
print(f"- Precision: {np.mean(metrics['precision']):.3f}")
print(f"- Recall: {np.mean(metrics['recall']):.3f}")
print(f"- F1 Score: {np.mean(metrics['f1']):.3f}")
# print(f"- AUC-ROC: {metrics['auc']:.3f}")
print(f"- Matthews Correlation Coefficient (MCC): {metrics['mcc']:.3f}")
print(f"- Confusion Matrix:")
print(df_cm)
print()

# Make visualization for training and validation losses
plt.figure()
plt.plot(range(1, NUM_EPOCHS + 1), loss_plot_callback.train_losses, label="Training Loss", color="green")
plt.plot(range(1, NUM_EPOCHS + 1), loss_plot_callback.eval_losses, label="Validation Loss", color="black")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("images/paper_c_losses.png")
plt.close()

print()
print("Hyperparameters:")
print(f"- Body Learning Rate: {BODY_LEARNING_RATE}")
# print(f"- Head Learning Rate: {HEAD_LEARNING_RATE}")
print(f"- Batch Size: {BATCH_SIZE}")
print(f"- Number of Epochs: {NUM_EPOCHS}")
# print(f"- L2 Weight: {L2_WEIGHT}")
print(f"- Max Iterations: {MAX_ITER}")
print(f"- Solver: {SOLVER}")
print("---")
print(f"- Seed: {SEED}")
print(f"- Model Seed: {MODEL_SEED}")
print()


"""
------------- Jan 18, 24 -------------

Model: SetFit (sentence-transformers/all-mpnet-base-v2)

- Accuracy: 0.944
- Precision: 0.944
- Recall: 0.944
- F1 Score: 0.944
- Matthews Correlation Coefficient (MCC): 0.921
- Confusion Matrix:
         support  oppose  neutral
support        6       0        0
oppose         0       6        0
neutral        0       1        5


Hyperparameters:
- Body Learning Rate: 0.00010214746757835319
- Batch Size: 32
- Number of Epochs: 2
---
- Seed: 42
"""
