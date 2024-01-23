import os
import random
from pprint import pprint

import numpy as np
import torch
from setfit import SetFitModel
from transformers import AutoTokenizer

from db import spreadsheet_4
from lib.utils import load_jsonl_file, read_from_google_sheet
from lib.utils2 import remove_examples_in_dataset

# Load dataset
dataset = read_from_google_sheet(spreadsheet_4, "dataset_3_")

dataset = dataset[1001:2000]

# Load model
model_setfit_path = "models/3"
model = SetFitModel.from_pretrained(model_setfit_path, local_files_only=True)
# Set model id
model_id = "sentence-transformers/paraphrase-mpnet-base-v2"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load test dataset
dataset3 = load_jsonl_file("shared_data/dataset_3_6_test_anonym.jsonl")

# Initialize label map and class names
LABEL_MAP = {"support": 0, "oppose": 1}
# Create reverse label map
REVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}
# REVERSE_LABEL_MAP = {0: "support", 1: "oppose"}

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


# dataset = [datapoint for datapoint in dataset if datapoint["class"] not in ["support", "oppose"]]
# Remove datapoints that are present in the training set
dataset = remove_examples_in_dataset(dataset, )


sentences = [datapoint["text"] for datapoint in dataset]

# pprint(sentences)

"""sentences = [
  "The terrorists and insurgents are violently opposed to democracy and will continue to attack it.",
  "Education is my top priority, and by supporting this budget, you'll make it yours, as well..",
  "A few days after the war began, I received a telegram from Joanne Speicher, the wife of the first pilot killed in the Gulf, Lieutenant Commander Scott Speicher.",
  "The Mac, short for Macintosh (its official name until 1999), is a family of personal computers designed and marketed by Apple Inc.",
  "Iraq's capacity to sustain war is being destroyed.",
  "If we value work, we can't justify a system that makes welfare more attractive than work if people are worried about losing their health care.",
  "We will reward good governance, work to reduce corruption, and support the rights of all Afghans, men and women alike."
]"""

preds = model.predict_proba(sentences)

if preds.is_cuda:
    predictions = preds.cpu()

predictions_list = preds.numpy().tolist()
# print(predictions_list)

# validity_threshold = 0.9925

for idx, p in enumerate(predictions_list):
  pred_class = None
  pred_score = 0
  if p[0] > p[1] and p[0] > 0.9930:  # 0.9931
    pred_class = "support"
    pred_score = p[0]
  elif p[0] < p[1] and p[1] > 0.9927:  # 0.9930
    pred_class = "oppose"
    pred_score = p[1]
  else:
    pred_class = "undefined"
  if pred_class in ["oppose"]:
    print(sentences[idx])
    print(pred_class)
    print(pred_score)
    print("-------")

