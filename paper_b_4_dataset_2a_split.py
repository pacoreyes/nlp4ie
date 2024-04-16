import random

import spacy
from tqdm import tqdm
# from pprint import pprint

from db import spreadsheet_4
from lib.utils import (read_from_google_sheet, save_row_to_jsonl_file, empty_json_file)
from lib.utils2 import remove_duplicated_datapoints


""" This script prepares the dataset for training, validation, and testing. 
    It also creates a parallel anonymized dataset. 
    The dataset is saved in a Google Sheet and in JSONL files. """

SEED = 42

# Set seed for reproducibility
random.seed(SEED)

# Load spacy model
nlp_trf = spacy.load("en_core_web_trf")

# Load dataset from Google Sheets
dataset3 = read_from_google_sheet(spreadsheet_4, "dataset_3*")

# Remove those datapoints that are not support or oppose
print("Combine datasets...")
dataset = []
accepted_classes = ["support", "oppose"]
for datapoint in tqdm(dataset3, desc=f"Processing {len(dataset)} datapoints"):
  if datapoint["label"] in accepted_classes:
    dataset.append(datapoint)

# Label mapping
LABEL_MAP = {
  "0": "support",
  "1": "oppose",
}

LABEL_CLASS_1 = LABEL_MAP["0"]
LABEL_CLASS_2 = LABEL_MAP["1"]

# Initialize path and name of output JSON-L files and Google Sheets
# output_spreadsheet = "dataset_3"

output_dataset_training = "datasets/2/seed/dataset_2_train.jsonl"
output_dataset_validation = "datasets/2/seed/dataset_2_validation.jsonl"
output_dataset_test = "datasets/2//seed/dataset_2_test.jsonl"

# Empty JSONL files
empty_json_file(output_dataset_training)
empty_json_file(output_dataset_validation)
empty_json_file(output_dataset_test)

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
Step 4: Filter datapoints by label
############################################# """

print("Filter datapoints...")

# Split dataset by label
class_0 = [item for item in dataset if item["label"] == "support"]
class_1 = [item for item in dataset if item["label"] == "oppose"]

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


# Create dataset for training
print("\nCreating dataset for training...")
for datapoint in tqdm(dataset_training, desc=f"Processing {len(dataset_training)} datapoints"):
  row = {
    "id": datapoint["id"],
    "text": datapoint["text"],
    "label": datapoint["label"]
  }

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_training)

# Create dataset for validation
print("\nCreating dataset for validation...")
for datapoint in tqdm(dataset_validation, desc=f"Processing {len(dataset_validation)} datapoints"):
  row = {
    "id": datapoint["id"],
    "text": datapoint["text"],
    "label": datapoint["label"]
  }

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_validation)

# Create dataset for test
print("\nCreating dataset for test...")
for datapoint in tqdm(dataset_test, desc=f"Processing {len(dataset_test)} datapoints"):
  row = {
    "id": datapoint["id"],
    "text": datapoint["text"],
    "label": datapoint["label"]
  }

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_test)
