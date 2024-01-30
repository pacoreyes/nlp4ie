# from pprint import pprint
import random

import torch
import tqdm
from setfit import SetFitModel

from db import spreadsheet_4
from lib.utils import load_jsonl_file, write_to_google_sheet, read_from_google_sheet
from lib.utils2 import remove_duplicated_datapoints, set_seed

# Set constants
SEED = 42

# Set seed
set_seed(SEED)

# Load dataset
# dataset1 = load_jsonl_file("shared_data/dataset_3_7_unlabeled_sentences_1.jsonl")

dataset1 = read_from_google_sheet(spreadsheet_4, "dataset_3_")
dataset2 = load_jsonl_file("shared_data/dataset_3_8_unlabeled_sentences_2.jsonl")

# Remove duplicated datapoints
dataset1 = remove_duplicated_datapoints(dataset1)
dataset2 = remove_duplicated_datapoints(dataset2)

# Remove skipped datapoints
dataset1 = [item for item in dataset1 if item["class"] == "skipped"]

# Merge datasets
dataset = dataset1 + dataset2

# dataset = dataset[:2000]  # use it to test the code


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
    row = {
      "id": dataset[idx]['id'],
      "text": sentences[idx],
      "target": dataset[idx]['target'],
      "label": pred_class,
      "score": pred_score
    }
    predictions.append(row)

# pprint(predictions)

# Create a representative sample of predictions
print("Creating a representative sample of predictions...")

# Extract the extremes
extreme_low = min(predictions, key=lambda x: x['score'])
extreme_high = max(predictions, key=lambda x: x['score'])

# Remove the extremes from the main dataset to avoid re-selection
filtered_predictions = [item for item in predictions if item not in [extreme_low, extreme_high]]

# Create buckets for different score ranges
buckets = {}
for item in filtered_predictions:
    bucket_key = int(item['score'] * 5) / 5  # Adjust this for different bucket sizes
    if bucket_key not in buckets:
        buckets[bucket_key] = []
    buckets[bucket_key].append(item)

# Sample from each bucket
sample_size = 300 - 2  # Reserving 2 spots for the extremes
samples_per_bucket = sample_size // len(buckets)
sampled_data = [extreme_low, extreme_high]

for bucket in buckets.values():
    if len(bucket) <= samples_per_bucket:
        sampled_data.extend(bucket)
    else:
        sampled_data.extend(random.sample(bucket, samples_per_bucket))

# Ensure the total sample size is 300
while len(sampled_data) < 500:
    additional_samples = random.choice(list(buckets.values()))
    sampled_data.extend(random.sample(additional_samples, 1))

# Deduplicate the final sample
sampled_data = [dict(t) for t in {tuple(d.items()) for d in sampled_data}]

# Format data for Google Sheet
predictions = []
for item in sampled_data:
  predictions.append([
    item['id'],
    item['text'],
    item['target'],
    item['label'],
    item['score']
  ])

# Write predictions to Google Sheet
print("Writing predictions to Google Sheet...")
write_to_google_sheet(spreadsheet_4, "error_analysis_0", predictions)
