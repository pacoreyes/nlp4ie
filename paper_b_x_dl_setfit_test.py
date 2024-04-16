import os
import random
# from pprint import pprint

import numpy as np
import pandas as pd
import torch
from setfit import SetFitModel
from sklearn.metrics import (precision_recall_fscore_support,
                             accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix)

from db import spreadsheet_4
from lib.utils import load_jsonl_file, read_from_google_sheet
from lib.visualizations import plot_confusion_matrix


# Load model
model_setfit_path = "models/3"
model = SetFitModel.from_pretrained(model_setfit_path, local_files_only=True)

# Set model id
model_id = "sentence-transformers/paraphrase-mpnet-base-v2"

# Load test dataset
test_dataset = load_jsonl_file("shared_data/dataset_3_7_test_master.jsonl")
# test_dataset = read_from_google_sheet(spreadsheet_4, "dataset_3_it2")

# Initialize label map and class names
LABEL_MAP = {"support": 0, "oppose": 1}
# Create list of class names
class_names = ["support", "oppose"]

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

  # Class-wise metrics
  precision_classwise, recall_classwise, f1_classwise, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

  # Average metrics
  precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

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
    'precision_classwise': precision_classwise,
    'recall_classwise': recall_classwise,
    'f1_classwise': f1_classwise,
    'precision_avg': precision_avg,
    'recall_avg': recall_avg,
    'f1_avg': f1_avg,
    'auc': auc,
    'mcc': mcc,
    'confusion_matrix': cm
  }


# Set seed for reproducibility
set_seed(SEED)

# Move model to appropriate device
model.to(get_device())

# Get sentences from test dataset
sentences = [datapoint["text"] for datapoint in test_dataset]

# Infer labels
pred_values = model.predict(sentences)

# Get labels from test dataset
true_values = [datapoint["label"] for datapoint in test_dataset]

# Convert labels to integers
true_values = [LABEL_MAP[label] for label in true_values]

# Compute metrics
metrics = compute_metrics(pred_values, true_values)

# Create dataframe for confusion matrix
df_cm = pd.DataFrame(metrics['confusion_matrix'], index=class_names, columns=class_names)

# Print metrics
print(f"\nModel: Setfit ({model_id})\n")
print(f"- Accuracy: {metrics['accuracy']:.3f}")
print(f"- Average Precision: {metrics['precision_avg']:.3f}")
print(f"- Average Recall: {metrics['recall_avg']:.3f}")
print(f"- Average F1: {metrics['f1_avg']:.3f}")
print(f"- AUC: {metrics['auc']:.3f}")
print(f"- MCC: {metrics['mcc']:.3f}")
print("- Confusion Matrix:")
print(df_cm)

# Print class-wise metrics
for i, class_name in enumerate(class_names):
    print(f"\nClass: {class_name}")
    print(f"- Precision: {metrics['precision_classwise'][i]:.3f}")
    print(f"- Recall: {metrics['recall_classwise'][i]:.3f}")
    print(f"- F1: {metrics['f1_classwise'][i]:.3f}")


"""
Model: Setfit (sentence-transformers/paraphrase-mpnet-base-v2)

- Accuracy: 1.000
- Precision: 1.000
- Recall: 1.000
- F1: 1.000
- AUC: 1.000
- MCC: 1.000
- Confusion Matrix:
         support  oppose
support        9       0
oppose         0       9

"""
