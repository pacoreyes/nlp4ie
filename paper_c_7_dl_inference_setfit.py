import random
import os

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from setfit import SetFitModel

from lib.utils import load_jsonl_file


# Load model
model_setfit_path = "models/3"
model = SetFitModel.from_pretrained(model_setfit_path, local_files_only=True)
# Set model id
model_id = "sentence-transformers/paraphrase-mpnet-base-v2"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load test dataset
dataset = load_jsonl_file("shared_data/dataset_3_5_validation_anonym.jsonl")

# Initialize label map and class names
LABEL_MAP = {"support": 0, "oppose": 1}
# Create reverse label map
REVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}
#REVERSE_LABEL_MAP = {0: "support", 1: "oppose"}

SEED = 1234


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


preds = model.predict([
    "I support democracy.",
    "We oppose migration.",
])
print(preds)

"""
# Split examples by class
support_sents = [datapoint["prompt"] for datapoint in dataset if datapoint["completion"] == "support"]
oppose_sents = [datapoint["prompt"] for datapoint in dataset if datapoint["completion"] == "oppose"]

print(support_sents)

preds = model(support_sents)

for i in range(len(support_sents)):

  print(preds[i])
  print(support_sents[i])
  print(REVERSE_LABEL_MAP[preds[i]])
  print('\n')"""


"""def predict(_text, _model, _tokenizer, _device):
  # Encode the input text
  inputs = _tokenizer(_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
  inputs = {key: value.to(_device) for key, value in inputs.items()}

  # Perform prediction
  with torch.no_grad():
    outputs = _model(**inputs)
    logits = outputs.logits
    print(logits.shape)

  # Apply softmax to get probabilities
  _probabilities = softmax(logits, dim=1)
  return _probabilities


# Set seed for reproducibility
set_seed(SEED)

device = get_device()
model.to(device)
model.eval()

for datapoint in dataset:
  print("--------------------------------")
  print(datapoint["prompt"])
  print(f"True class: {datapoint['completion']}")

  text = datapoint["prompt"]

  probabilities = predict(text, model, tokenizer, device)

  # Get the predicted class
  predicted_class = torch.argmax(probabilities, dim=1).item()
  print(f"Predicted class: {REVERSE_LABEL_MAP.get(predicted_class)}")
  print(f"Probabilities: {probabilities}")
  print()"""
