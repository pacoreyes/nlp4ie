import random

import shap
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score, \
  roc_curve, auc
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

from lib.utils import load_jsonl_file
# empty_json_file, save_row_to_jsonl_file)
# from lib.visualizations import plot_confusion_matrix


# Initialize constants
SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 16


# Initialize label map and class names
LABEL_MAP = {"continue": 0, "not_continue": 1}
CLASS_NAMES = list(LABEL_MAP.keys())
REVERSED_LABEL_MAP = {0: "continue", 1: "not_continue"}

# CLASS_NAMES = ['continue', 'not_continue']


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")
  # return torch.device("cpu")


def set_seed(seed_value):
  """Set seed for reproducibility."""
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


# Set seed for reproducibility
set_seed(SEED)

# Initialize BERT model
BERT_MODEL = 'bert-base-uncased'

# Assuming you've saved your best model to a path during training
MODEL_PATH = "models/3/TopicContinuityBERT.pth"

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# Load Pre-trained Model
model = BertForSequenceClassification.from_pretrained(BERT_MODEL)

# Move Model to Device
device = get_device()
model.to(device)

# Load Saved Weights
print("• Loading Saved Weights...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Load test set
# test_set = load_jsonl_file("shared_data/topic_boundary_test.jsonl")

model.eval()


def preprocess_pairs(text_pairs, tokenizer, device, max_length=MAX_LENGTH):
  """Tokenize and preprocess text pairs."""
  input_ids = []
  attention_masks = []

  for text in text_pairs:
    sentence1, sentence2 = text
    encoded_input = tokenizer.encode_plus(
      sentence1.strip(),
      sentence2.strip(),
      add_special_tokens=True,
      max_length=max_length,
      truncation=True,
      padding='max_length',
      return_tensors='pt'
    )
    input_ids.append(encoded_input['input_ids'])
    attention_masks.append(encoded_input['attention_mask'])

  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)

  return input_ids.to(device), attention_masks.to(device)


def predict(text_pairs):
  input_ids, attention_masks = preprocess_pairs(text_pairs, tokenizer, device)
  with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_masks)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
  return probs.cpu().numpy()


# Create SHAP explainer
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict, masker)

# Example text pairs for explanation
text_pairs = [
  ("Sentence one of pair 1", "Sentence two of pair 1"),
  ("Sentence one of pair 2", "Sentence two of pair 2")
]

# Explain the model's predictions
shap_values = explainer(text_pairs)
