import random
import os
# from pprint import pprint
from collections import Counter

# from optuna import Trial
import optuna
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from transformers import TrainerCallback, EarlyStoppingCallback
from sklearn.manifold import TSNE
from sklearn.metrics import (precision_recall_fscore_support,
                             accuracy_score, roc_auc_score, matthews_corrcoef)

from lib.utils import load_jsonl_file

import torch

# optuna.logging.set_verbosity(optuna.logging.INFO)

# Initialize label map and class names
LABEL_MAP = {"support": 0, "oppose": 1}
class_names = list(LABEL_MAP.keys())

# Initialize constants
SEED = 42
NUM_TRIALS = 2


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


# Set device to CUDA, MPS, or CPU
device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")


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
    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    "seed": trial.suggest_int("seed", 1, 40),
    "max_iter": trial.suggest_int("max_iter", 50, 300),
    "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
    # "end_to_end": True,
  }


def compute_metrics(y_pred, y_test):
  accuracy = accuracy_score(y_test, y_pred)
  precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
  auc = roc_auc_score(y_test, y_pred)
  mcc = matthews_corrcoef(y_test, y_pred)
  # cm = confusion_matrix(y_test, y_pred)

  """# Plot confusion matrix
  plot_confusion_matrix(y_test,
                        y_pred,
                        ["support", "oppose"],
                        "paper_c_1_dl_setfit_confusion_matrix.png",
                        "Confusion Matrix for SetFit model",
                        values_fontsize=22
                        )"""
  return {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'auc': auc,
    'mcc': mcc,
    # 'confusion_matrix': cm
  }


class EmbeddingPlotCallback(TrainerCallback):
  """Simple embedding plotting callback that plots the tSNE of the training and evaluation datasets
  # throughout training. LARGE GREEN/ORANGE"""

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
                          markerfacecolor=custom_colors[i], markersize=10) for i in range(len(class_names))]
    fig.legend(handles=handles, title="Stance classes", loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=len(class_names), fontsize='medium')

    # Set the super title for the figure
    fig.suptitle(f"tSNE of training and evaluation embeddings at step {state.global_step} of {state.max_steps}.",
                 fontsize=16)

    # Save the figure and close the plot to free memory
    fig.savefig(f"images/setfit_step_{state.global_step}.png", bbox_inches='tight')
    plt.close(fig)


# Initialize model
model_id = "sentence-transformers/all-mpnet-base-v2"
# model_id = "sentence-transformers/paraphrase-mpnet-base-v2"
# model_id = "BAAI/bge-small-en-v1.5"


# Load datasets
dataset_training_route = "datasets/3/bootstrap_0_seed/dataset_3_1_training.jsonl"
dataset_validation_route = "datasets/3/bootstrap_0_seed/dataset_3_2_validation.jsonl"
dataset_test_route = "datasets/3/bootstrap_0_seed/dataset_3_3_test.jsonl"
dataset_training = load_jsonl_file(dataset_training_route)
dataset_validation = load_jsonl_file(dataset_validation_route)
dataset_test = load_jsonl_file(dataset_test_route)

# Reverse label map from string to integer
dataset_training = [{"label": LABEL_MAP[datapoint["label"]], "text": datapoint["text"]}
                    for datapoint in dataset_training]
dataset_validation = [{"label": LABEL_MAP[datapoint["label"]], "text": datapoint["text"]}
                      for datapoint in dataset_validation]
dataset_test = [{"label": LABEL_MAP[datapoint["label"]], "text": datapoint["text"]}
                for datapoint in dataset_test]

dataset_training = dataset_training + dataset_test

dataset_test = load_jsonl_file("datasets/3/bootstrap_0_seed/dataset_3_4_test_master.jsonl")

dataset_test = [{"label": LABEL_MAP[datapoint["label"]], "text": datapoint["text"]}
                for datapoint in dataset_test]

# Count and print class distribution
train_label_counts = Counter([datapoint['label'] for datapoint in dataset_training])
val_label_counts = Counter([datapoint['label'] for datapoint in dataset_validation])
test_label_counts = Counter([datapoint['label'] for datapoint in dataset_test])

print(f"\nClass distribution in training dataset: {train_label_counts}")
print(f"Class distribution in validation dataset: {val_label_counts}")
print(f"Class distribution in test dataset: {test_label_counts}\n")


# Convert training data into a Dataset object
train_columns = {key: [dic[key] for dic in dataset_training] for key in dataset_training[0]}
train_dataset = Dataset.from_dict(train_columns)

# Convert validation data into a Dataset object
val_columns = {key: [dic[key] for dic in dataset_validation] for key in dataset_validation[0]}
validation_dataset = Dataset.from_dict(val_columns)

# Convert test data into Dataset object
test_columns = {key: [dic[key] for dic in dataset_test] for key in dataset_test[0]}
test_dataset = Dataset.from_dict(test_columns)

args = TrainingArguments(
  output_dir="models/3_2024",
  evaluation_strategy="epoch",
  save_strategy="epoch",
  # logging_steps=50,
  eval_steps=1,
  save_steps=1,
  load_best_model_at_end=True,
  metric_for_best_model="accuracy",
  greater_is_better=False,
  seed=SEED,
)

# Initialize trainer
trainer = Trainer(
  train_dataset=train_dataset,  # training dataset
  eval_dataset=validation_dataset,  # validation dataset
  model_init=model_init,  # model initialization function
  metric=compute_metrics,  # metrics computation function
  args=args,  # training arguments
  callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # callbacks
)

# Perform hyperparameter search
best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=NUM_TRIALS)

# Print best run information
print(f"\nBest run: {best_run}")


"""# Apply the best hyperparameters to the best model
trainer.apply_hyperparameters(best_run.hyperparameters, final_model=False)

# Initialize callbacks for embedding plots
embedding_plot_callback = EmbeddingPlotCallback()

# Add callbacks to trainer
trainer.callback_handler.add_callback(embedding_plot_callback)

# Train best model
trainer.train()

# Evaluate best model using the test dataset
metrics = trainer.evaluate(test_dataset, "test")
print(f"\nMetrics: {metrics}")

# Save best model
trainer.model.save_pretrained("models/9")
print("\nModel saved successfully!\n")"""
