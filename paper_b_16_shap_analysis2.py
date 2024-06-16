import torch
import shap
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from setfit import SetFitModel
from transformers import AutoTokenizer

from lib.utils import load_jsonl_file

CLASS_NAMES = ['support', 'oppose']

# Set model id
model_id = "sentence-transformers/paraphrase-mpnet-base-v2"

# Load dataset
DATASET = load_jsonl_file("shared_data/dataset_2_test.jsonl")

support_class = [data for data in DATASET if data["label"] == "support"]
oppose_class = [data for data in DATASET if data["label"] == "oppose"]


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


# Load model
model_setfit_path = "models/9"
model = SetFitModel.from_pretrained(model_setfit_path, local_files_only=True)

# Get best device
device = get_device()

# Move the model to the chosen device
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_setfit_path, model_max_length=512)

# Select texts from the support class
texts = [data["text"] for data in support_class]

masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(model, masker=masker)
shap_values = explainer(texts[1], fixed_context=1)

print(shap_values)

# shap.plots.bar(shap_values.abs.mean(0), max_display=100, show=False)

# shap.plots.bar(shap_values[0, :, "support"], show=False)

figure = plt.figure()

shap.plots.bar(shap_values.abs.sum(0))

"""------------------------------------------------"""

"""# define a prediction function
def f(x):
  tv = torch.tensor(
      [
        tokenizer.encode(v, padding="max_length", max_length=500, truncation=True)
        for v in x
      ]
  ).to(dtype=torch.long, device=device)
  outputs = model(tv)[0].detach().cpu().numpy()
  scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
  val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
  return val


# build an explainer using a token masker
explainer = shap.Explainer(f, tokenizer)

shap_values = explainer(texts[0], fixed_context=1)"""

"""------------------------------------------------"""

"""# Wrapper function for SHAP
def model_predict(texts):
    # Tokenize and encode the texts using the model's tokenizer
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Get model predictions
    with torch.no_grad():
        logits = model(input_ids, attention_mask).logits
    return logits.cpu().numpy()


# Custom masker function to convert texts to embeddings
def masker(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    return encodings["input_ids"].cpu().numpy()


# Create a SHAP explainer with the custom masker
explainer = shap.Explainer(model_predict, masker)

# Compute SHAP values for the selected samples
shap_values = explainer(texts)"""

