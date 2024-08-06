# from pprint import pprint

import scipy as sp
import shap
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt

from lib.utils import load_jsonl_file

CLASS_NAMES = ['monologic', 'dialogic']

# Load dataset
DATASET = load_jsonl_file("shared_data/dataset_1_6_1b_test_anonym.jsonl")

# Split the dataset into speech and interview classes
speech_class = [data for data in DATASET if data["label"] == "monologic"]
interview_class = [data for data in DATASET if data["label"] == "dialogic"]


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


# Set device
device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")

# Initialize constants
BERT_MODEL = 'bert-base-uncased'
# MODEL_PATH = 'models/1/paper_a_x_dl_bert_train_hop_bert.pth'
MODEL_PATH = 'models/1/paper_a_bert_solo_anonym.pth'

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=len(CLASS_NAMES),
                                                      hidden_dropout_prob=0.1)

# Move the model to the device
model = model.to(device)
# Load the model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# Set the model to evaluation mode
model.eval()


def f(x):
  tv = torch.tensor([tokenizer.encode(v, padding="max_length", max_length=500, truncation=True)
                     for v in x], device=device)
  outputs = model(tv)[0].detach().cpu().numpy()
  scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
  val = sp.special.logit(scores[:, 1])
  return val


# Initialize the SHAP explainer
explainer = shap.Explainer(f, tokenizer)
# explainer = shap.Explainer(f, tokenizer, output_names=["monologic", "dialogic"])

# texts = [data["text"] for data in speech_class]
texts = [data["text"] for data in interview_class]

# Compute SHAP values for the selected samples
shap_values = explainer(texts, fixed_context=1)

# Visualize the SHAP values
# shap.plots.text(shap_values)

shap.plots.bar(shap_values.abs.mean(0), max_display=100, show=False)
# shap.plots.bar(shap_values.abs.mean(0))

summed_shap_values = shap_values.abs.mean(0).values
summed_shap_features = shap_values.abs.mean(0).feature_names

# Convert to DataFrame for easy CSV saving
df_shap_values = pd.DataFrame(summed_shap_values, columns=["Value"])

# Add a column for the feature names if available
df_shap_values["Feature"] = summed_shap_features
df_shap_values = df_shap_values.sort_values(by="Value", ascending=False)

# Save to CSV
df_shap_values.to_csv("shared_data/dataset_1_9_interview_anonym_shap_features_plot_bar.csv", index=False)

# Save the plot PNG
plt.savefig("images/paper_a_20_interview_anonym_shap_features_plot_bar.png")
