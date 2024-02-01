# from pprint import pprint

import torch
import tqdm
from setfit import SetFitModel

from db import spreadsheet_4
from lib.utils import load_jsonl_file, write_to_google_sheet, read_from_google_sheet
from lib.utils2 import remove_duplicated_datapoints, remove_examples_in_dataset


# Initialize constants
GOOGLE_SHEET = "dataset3"

# Load dataset
# dataset = load_jsonl_file("shared_data/dataset_3_7_unlabeled_sentences_1.jsonl")
# dataset = load_jsonl_file("shared_data/dataset_3_9_unseen_unlabeled_sentences.jsonl")
dataset = load_jsonl_file("shared_data/_123/argumentative_sentences.jsonl")

# Load dataset3
dataset_3 = read_from_google_sheet(spreadsheet_4, "dataset_3_it2")
dataset = remove_examples_in_dataset(dataset, dataset_3)

print("##############################################")
dataset = remove_duplicated_datapoints(dataset)
print("##############################################")

dataset = dataset[:2000]


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


# Load model
model_setfit_path = "models/3"
model = SetFitModel.from_pretrained(model_setfit_path, local_files_only=True)

# Get best device
device = get_device()

# Move the model to the chosen device
model.to(device)

# Create list of sentences
sentences = [datapoint["text"] for datapoint in dataset]
# pprint(sentences)

# Make predictions using the model
print("Making predictions...")
predictions = model.predict_proba(sentences)

# Move predictions to CPU
if predictions.is_cuda:
  predictions = predictions.cpu()

# Convert predictions to list
predictions_list = predictions.numpy().tolist()

# Filter predictions by class and generate inference
predictions = []
for idx, p in tqdm.tqdm(enumerate(predictions_list, start=0), desc=f"Processing {len(predictions_list)} predictions"):
  pred_class = None
  pred_score = 0
  if p[0] > p[1] and p[0] > 0:  # 0.9947 is the threshold
    pred_class = "support"
    pred_score = p[0]
  elif p[0] < p[1] and p[1] > 0:  # 0.9947 is the threshold
    pred_class = "oppose"
    pred_score = p[1]
  else:
    pred_class = "undefined"
  if pred_class in ["oppose", "support"]:
    predictions.append(
      [
        dataset[idx]['id'],
        sentences[idx],
        # dataset[idx]['target'],
        "",
        pred_class,
        pred_score
      ]
    )

# Sort predictions by score
predictions.sort(key=lambda x: x[4], reverse=True)
# Take top 50 predictions
# predictions = predictions[:50]

# Write predictions to Google Sheet
print("Writing predictions to Google Sheet...")
write_to_google_sheet(spreadsheet_4, GOOGLE_SHEET, predictions)
