import random

import spacy
from tqdm import tqdm
# from pprint import pprint

from db import spreadsheet_4
from lib.utils2 import anonymize_text
from lib.utils import (read_from_google_sheet, write_to_google_sheet, save_row_to_jsonl_file, empty_json_file)
from lib.utils2 import remove_duplicated_datapoints


""" This script prepares the dataset for training, validation, and testing. 
    It also creates a parallel anonymized dataset. 
    The dataset is saved in a Google Sheet and in JSONL files. """

SEED = 1234
ANONYMIZE_TARGET = False

# Set seed for reproducibility
random.seed(SEED)

# Load spacy model
nlp_trf = spacy.load("en_core_web_trf")

# Load dataset from Google Sheets
dataset3 = read_from_google_sheet(spreadsheet_4, "dataset_3")
# dataset3_2 = read_from_google_sheet(spreadsheet_4, "dataset_3_")

# Join dataset and dataset3_1
# dataset3 = dataset3_2 + dataset3_1

# Remove those datapoints that are not support or oppose
print("Combine datasets...")
dataset = []
accepted_classes = ["support", "oppose"]
for datapoint in tqdm(dataset3, desc=f"Processing {len(dataset)} datapoints"):
  if datapoint["class"] in accepted_classes:
    dataset.append(datapoint)

# Label mapping
LABEL_MAP = {
  "0": "support",
  "1": "oppose",
}

LABEL_CLASS_1 = LABEL_MAP["0"]
LABEL_CLASS_2 = LABEL_MAP["1"]

# Initialize path and name of output JSON-L files and Google Sheets
output_spreadsheet = "dataset_3"

output_dataset_training = "shared_data/dataset_3_1_training.jsonl"
output_dataset_validation = "shared_data/dataset_3_2_validation.jsonl"
output_dataset_test = "shared_data/dataset_3_3_test.jsonl"
output_dataset_training_anonym = "shared_data/dataset_3_4_training_anonym.jsonl"
output_dataset_validation_anonym = "shared_data/dataset_3_5_validation_anonym.jsonl"
output_dataset_test_anonym = "shared_data/dataset_3_6_test_anonym.jsonl"


# Empty JSONL files
empty_json_file(output_dataset_training)
empty_json_file(output_dataset_validation)
empty_json_file(output_dataset_test)
empty_json_file(output_dataset_training_anonym)
empty_json_file(output_dataset_validation_anonym)
empty_json_file(output_dataset_test_anonym)

""" #############################################
Step 2: Remove duplicated datapoints
############################################# """

print("\nRemoving duplicated datapoints...")

dataset = remove_duplicated_datapoints(dataset, verbose=True)

""" #############################################
Step 3: Shuffle dataset
############################################# """

random.shuffle(dataset)

""" #############################################
Write dataset to Google Sheets
############################################# """

dataset3 = []
for datapoint in dataset:
  row = [
    datapoint["id"],
    datapoint["text"],
    datapoint["target"],
    datapoint["class"],
    # datapoint["type"],
  ]
  dataset3.append(row)

# Order dataset3 by class
dataset3 = sorted(dataset3, key=lambda x: x[3])

# Save dataset to Google Sheets
write_to_google_sheet(spreadsheet_4, output_spreadsheet, dataset3)

""" #############################################
Step 4: Filter datapoints by label
############################################# """

# dataset_remapped = []

print("Filter datapoints...")

"""for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  if datapoint["class"] == "0" or datapoint["class"] == "1" or datapoint["class"] == "2":
    dataset_remapped.append(datapoint)"""

# Split dataset by label
class_0 = [item for item in dataset if item["class"] == "support"]
class_1 = [item for item in dataset if item["class"] == "oppose"]

"""trim_length = 76
class_0 = class_0[:trim_length]
class_1 = class_1[:trim_length]"""

print(f"• Class 0: {len(class_0)}")
print(f"• Class 1: {len(class_1)}")

# Shuffle datasets
random.shuffle(class_0)
random.shuffle(class_1)

class_0_training = class_0[:int(len(class_0) * 0.8)]  # 80%
class_0_validation = class_0[int(len(class_0) * 0.8):int(len(class_0) * 0.9)]  # 10%
class_0_test = class_0[int(len(class_0) * 0.9):]  # 10%

class_1_training = class_1[:int(len(class_1) * 0.8)]  # 80%
class_1_validation = class_1[int(len(class_1) * 0.8):int(len(class_1) * 0.9)]  # 10%
class_1_test = class_1[int(len(class_1) * 0.9):]  # 10%

# Merge datasets
dataset_training = class_0_training + class_1_training
dataset_validation = class_0_validation + class_1_validation
dataset_test = class_0_test + class_1_test

# Create parallel anonymized datasets
dataset_training_anonym = []
dataset_validation_anonym = []
dataset_test_anonym = []


# Create dataset for training
print("\nCreating dataset for training...")
for datapoint in tqdm(dataset_training, desc=f"Processing {len(dataset_training)} datapoints"):
  # Anonymize target
  if ANONYMIZE_TARGET:
    target = datapoint["target"]
    datapoint["text"] = datapoint["text"].replace(target, "[TARGET]")
  row = {
    "id": datapoint["id"],
    "text": datapoint["text"],
    "label": datapoint["class"]
  }
  row_anonym = {
    "id": datapoint["id"],
    "text": anonymize_text(datapoint["text"], nlp_trf),
    "label": datapoint["class"]
  }

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_training)
  save_row_to_jsonl_file(row_anonym, output_dataset_training_anonym)

# Create dataset for validation
print("\nCreating dataset for validation...")
for datapoint in tqdm(dataset_validation, desc=f"Processing {len(dataset_validation)} datapoints"):
  # Anonymize target
  if ANONYMIZE_TARGET:
    target = datapoint["target"]
    datapoint["text"] = datapoint["text"].replace(target, "[TARGET]")
  row = {
    "id": datapoint["id"],
    "text": datapoint["text"],
    "label": datapoint["class"]
  }
  row_anonym = {
    "id": datapoint["id"],
    "text": anonymize_text(datapoint["text"], nlp_trf),
    "label": datapoint["class"]
  }

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_validation)
  save_row_to_jsonl_file(row_anonym, output_dataset_validation_anonym)

# Create dataset for test
print("\nCreating dataset for test...")
for datapoint in tqdm(dataset_test, desc=f"Processing {len(dataset_test)} datapoints"):
  # Anonymize target
  if ANONYMIZE_TARGET:
    target = datapoint["target"]
    datapoint["text"] = datapoint["text"].replace(target, "[TARGET]")
  row = {
    "id": datapoint["id"],
    "text": datapoint["text"],
    "label": datapoint["class"]
  }
  row_anonym = {
    "id": datapoint["id"],
    "text": anonymize_text(datapoint["text"], nlp_trf),
    "label": datapoint["class"]
  }

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_test)
  save_row_to_jsonl_file(row_anonym, output_dataset_test_anonym)
