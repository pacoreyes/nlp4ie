# from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from transformers import TrainerCallback
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import (precision_recall_fscore_support,
                             accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix)

from lib.utils import load_jsonl_file

# Initialize label map and class names
LABEL_MAP = {"support": 0, "oppose": 1}
class_names = list(LABEL_MAP.keys())


# Hyperparameters
BODY_LEARNING_RATE = 6.125082727886936e-05
BATCH_SIZE = 32
NUM_EPOCHS = 2
MAX_ITER = 249
MODEL_SEED = 19
SOLVER = "liblinear"


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
  # Simple embedding plotting callback that plots the tSNE of the training and evaluation datasets
  # throughout training."""

  def on_evaluate(self, args, state, control, **kwargs):
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
    fig.savefig(f"images/step_{state.global_step}.png")


"""class EmbeddingPlotCallback(TrainerCallback):
  # Simple embedding plotting callback that plots the 3D tSNE of the training and evaluation datasets
  # throughout training.

  def on_evaluate(self, args, state, control, **kwargs):
    _model = model_init()
    train_embeddings = _model.encode(train_dataset["text"])
    eval_embeddings = _model.encode(validation_dataset["text"])

    # Determine dataset size to adjust perplexity for 3D
    eval_dataset_size = len(validation_dataset["text"])
    perplexity_value = min(30, max(5, int(eval_dataset_size / 2)))

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # 3D t-SNE for training embeddings
    train_x = TSNE(n_components=3, perplexity=perplexity_value).fit_transform(train_embeddings)
    ax1.scatter(train_x[:, 0], train_x[:, 1], train_x[:, 2], c=train_dataset["label"])
    ax1.set_title("Training embeddings")

    # 3D t-SNE for evaluation embeddings
    eval_x = TSNE(n_components=3, perplexity=perplexity_value).fit_transform(eval_embeddings)
    ax2.scatter(eval_x[:, 0], eval_x[:, 1], eval_x[:, 2], c=validation_dataset["label"])
    ax2.set_title("Evaluation embeddings")

    fig.suptitle(f"3D tSNE of training and evaluation embeddings at step {state.global_step} of {state.max_steps}.")
    # plt.show()
    fig.savefig(f"images/3D_step_{state.global_step}.png")"""


# Set device to CUDA, MPS, or CPU
device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")

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
print("\nDataset:")
print(f"- training: {len(dataset_training)}")
print(f"- validation: {len(dataset_validation)}")
print(f"- test: {len(dataset_test)}")


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

# Initialize callbacks
embedding_plot_callback = EmbeddingPlotCallback()

# Initialize model parameters
arguments = TrainingArguments(
  body_learning_rate=BODY_LEARNING_RATE,
  num_epochs=NUM_EPOCHS,
  batch_size=BATCH_SIZE,
  seed=MODEL_SEED,
  evaluation_strategy="steps",
  num_iterations=20,
  eval_steps=20,
  end_to_end=True,
)

# Training Loop
trainer = Trainer(
  model_init=model_init,
  args=arguments,
  train_dataset=train_dataset,
  # eval_dataset=test_dataset,
  metric=compute_metrics,
  # callbacks=[embedding_plot_callback],
)

trainer.train()

metrics = trainer.evaluate(test_dataset)

df_cm = pd.DataFrame(metrics['confusion_matrix'], index=class_names, columns=class_names)

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
print(f"- Seed: {MODEL_SEED}")
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
