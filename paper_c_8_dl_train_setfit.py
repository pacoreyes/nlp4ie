import random
import os
# from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from transformers import TrainerCallback, EarlyStoppingCallback
from sklearn.manifold import TSNE
from sklearn.metrics import (precision_recall_fscore_support,
                             accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix)

from lib.utils import load_jsonl_file
from lib.visualizations import plot_confusion_matrix


""" NOTE: Set this environment variable to avoid CUDA out of memory errors.
Set them always before importing torch"""
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = "0.0"

import torch

# Initialize label map and class names
LABEL_MAP = {"support": 0, "oppose": 1}
class_names = list(LABEL_MAP.keys())


# Hyperparameters
BODY_LEARNING_RATE = 1.1372066559539598e-06
NUM_EPOCHS = 1
BATCH_SIZE = 32
SEED = 9
MAX_ITER = 297
SOLVER = "lbfgs"


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


def compute_metrics(y_pred, y_test):
  accuracy = accuracy_score(y_test, y_pred)
  precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
  auc = roc_auc_score(y_test, y_pred)
  mcc = matthews_corrcoef(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)

  # Plot confusion matrix
  plot_confusion_matrix(y_test,
                        y_pred,
                        ["support", "oppose"],
                        "paper_c_1_dl_setfit_confusion_matrix.png",
                        "Confusion Matrix for SetFit model",
                        values_fontsize=22
                        )
  return {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'auc': auc,
    'mcc': mcc,
    'confusion_matrix': cm
  }


class EmbeddingPlotCallback(TrainerCallback):
  """ Simple embedding plotting callback that plots the tSNE of the training and evaluation datasets
  # throughout training. """

  def on_evaluate(self, args, state, control, **kwargs):
    train_embeddings = trainer.model.encode(train_dataset["text"])
    eval_embeddings = trainer.model.encode(validation_dataset["text"])

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
    fig.savefig(f"images/paper_c_setfit_step_{state.global_step}.png")
    plt.close(fig)


"""class EmbeddingPlotCallback(TrainerCallback):
  #Simple embedding plotting callback that plots the tSNE of the training and evaluation datasets
  # throughout training.

  def on_evaluate(self, args, state, control, **kwargs):
    train_embeddings = trainer.model.encode(train_dataset["text"])
    eval_embeddings = trainer.model.encode(validation_dataset["text"])

    # Determine dataset size to adjust perplexity
    eval_dataset_size = len(validation_dataset["text"])
    perplexity_value = min(30, max(5, int(eval_dataset_size / 2)))

    # Enlarge the canvas
    fig, (train_ax, eval_ax) = plt.subplots(ncols=2, figsize=(15, 7))

    # Custom colors: green for 'support', orange for 'oppose'
    custom_colors = ['green', 'orange']
    color_map = [custom_colors[label] for label in train_dataset["label"]]

    train_x = TSNE(n_components=2, perplexity=perplexity_value).fit_transform(train_embeddings)
    train_ax.scatter(*train_x.T, c=color_map, label=class_names)
    train_ax.set_title("Training embeddings")

    eval_color_map = [custom_colors[label] for label in validation_dataset["label"]]
    eval_x = TSNE(n_components=2, perplexity=perplexity_value).fit_transform(eval_embeddings)
    eval_ax.scatter(*eval_x.T, c=eval_color_map, label=class_names)
    eval_ax.set_title("Evaluation embeddings")

    # Create a shared legend and place it at the bottom of the figure
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i],
                          markerfacecolor=custom_colors[i], markersize=12) for i in range(len(class_names))]
    fig.legend(handles=handles, title="Stance classes", loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=len(class_names), fontsize='medium')

    # Set the super title for the figure
    fig.suptitle(f"tSNE of training and evaluation embeddings at step {state.global_step} of {state.max_steps}.",
                 fontsize=16)

    # Save the figure and close the plot to free memory
    fig.savefig(f"images/setfit_step_{state.global_step}.png", bbox_inches='tight')
    plt.close(fig)"""


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
print("\nDataset:")
print(f"- training: {len(dataset_training)}")
print(f"- validation: {len(dataset_validation)}")
print(f"- test: {len(dataset_test)}")


num_support_test = len([item for item in dataset_test if item["label"] == 0])
num_oppose_test = len([item for item in dataset_test if item["label"] == 1])

print(f"\nTest set class distribution:")
print(f"- support: {num_support_test}")
print(f"- oppose: {num_oppose_test}\n")

# Convert training data into a Dataset object
train_columns = {key: [dic[key] for dic in dataset_training] for key in dataset_training[0]}
train_dataset = Dataset.from_dict(train_columns)

# Convert validation data into a Dataset object
val_columns = {key: [dic[key] for dic in dataset_validation] for key in dataset_validation[0]}
validation_dataset = Dataset.from_dict(val_columns)

# Convert test data into Dataset object
test_columns = {key: [dic[key] for dic in dataset_test] for key in dataset_test[0]}
test_dataset = Dataset.from_dict(test_columns)

# Initialize callback for embedding plots
embedding_plot_callback = EmbeddingPlotCallback()
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

# Initialize model parameters
arguments = TrainingArguments(
  body_learning_rate=BODY_LEARNING_RATE,
  num_epochs=NUM_EPOCHS,
  batch_size=BATCH_SIZE,
  seed=SEED,
  evaluation_strategy="steps",
  num_iterations=69,  # Note from the lecturer: I don't know why this number is 69, but it changes the number of steps
  eval_steps=20,
  end_to_end=True,
  load_best_model_at_end=True
)

# Training Loop
trainer = Trainer(
  model_init=model_init,
  args=arguments,
  train_dataset=train_dataset,
  eval_dataset=validation_dataset,
  callbacks=[embedding_plot_callback, early_stopping_callback],
  metric=compute_metrics,
)

# Train model
trainer.train()

# Evaluate model on test dataset
metrics = trainer.evaluate(test_dataset)

# Create confusion matrix
df_cm = pd.DataFrame(metrics['confusion_matrix'], index=class_names, columns=class_names)

# Print metrics and Hyperparameters
print(f"\nModel: SetFit ({model_id})\n")
print(f"- Accuracy: {metrics['accuracy']:.3f}")
print(f"- Precision: {np.mean(metrics['precision']):.3f}")
print(f"- Recall: {np.mean(metrics['recall']):.3f}")
print(f"- F1 Score: {np.mean(metrics['f1']):.3f}")
print(f"- AUC-ROC: {metrics['auc']:.3f}")
print(f"- Matthews Correlation Coefficient (MCC): {metrics['mcc']:.3f}")
print(f"- Confusion Matrix:")
print(df_cm)
print()
print("Hyperparameters:")
print(f"- Body Learning Rate: {BODY_LEARNING_RATE}")
print(f"- Batch Size: {BATCH_SIZE}")
print(f"- Number of Epochs: {NUM_EPOCHS}")
print(f"- Max Iterations: {MAX_ITER}")
print(f"- Solver: {SOLVER}")
print("---")
print(f"- Seed: {SEED}")
print()

# Save model
trainer.model.save_pretrained("models/3")
print("\nModel saved successfully!\n")
