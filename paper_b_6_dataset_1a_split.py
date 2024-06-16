import random

import spacy
from tqdm import tqdm
# from pprint import pprint

from db import spreadsheet_4
from lib.utils import (read_from_google_sheet, save_row_to_jsonl_file, empty_json_file)
# from lib.utils2 import remove_duplicated_datapoints
# from lib.text_utils import preprocess_text


""" This script prepares the dataset for training, validation, and testing. 
    The dataset is saved in JSONL files. """

if spacy.prefer_gpu():
  print("spaCy is using GPU!")
else:
  print("GPU not available, spaCy is using CPU instead.")

# load spaCy's Transformer model
nlp = spacy.load("en_core_web_trf")

SEED = 2024

# Set seed for reproducibility
random.seed(SEED)

# Load spacy model
nlp_trf = spacy.load("en_core_web_trf")

# Load dataset from Google Sheets
dataset = read_from_google_sheet(spreadsheet_4, "dataset_3_final")

# Label mapping
LABEL_MAP = {
  "0": "support",
  "1": "oppose",
}

LABEL_CLASS_1 = LABEL_MAP["0"]
LABEL_CLASS_2 = LABEL_MAP["1"]

# Initialize path and name of output JSON-L files and Google Sheets
# output_spreadsheet = "dataset_3"

output_dataset_training = "shared_data/dataset_2_train.jsonl"
output_dataset_validation = "shared_data/dataset_2_validation.jsonl"
output_dataset_test = "shared_data/dataset_2_test.jsonl"

# Empty JSONL files
empty_json_file(output_dataset_training)
empty_json_file(output_dataset_validation)
empty_json_file(output_dataset_test)

""" #############################################
Step 2: Remove duplicated datapoints
############################################# """

print("\nRemoving duplicated datapoints...")

# dataset = remove_duplicated_datapoints(dataset, verbose=True)

""" #############################################
Step 3: Shuffle dataset
############################################# """

# random.shuffle(dataset)

""" #############################################
Step 4: Process metadata
############################################# """

# Convert metadata into dictionary
"""print("Processing metadata...")
for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  if datapoint["metadata"]:
    datapoint["metadata"] = eval(datapoint["metadata"])

    metadata_title = preprocess_text(datapoint["metadata"]["title"], nlp,
                                     with_remove_known_unuseful_strings=True,
                                     with_remove_parentheses_and_brackets=False,
                                     with_remove_text_inside_parentheses=False,
                                     with_remove_leading_patterns=False,
                                     with_remove_timestamps=False,
                                     with_replace_unicode_characters=True,
                                     with_expand_contractions=False,
                                     with_remove_links_from_text=False,
                                     with_put_placeholders=False,
                                     with_final_cleanup=True)

    datapoint["metadata"]["title"] = metadata_title

# Remove duplicates from semantic frames in metadata
for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  semantic_frames = datapoint["metadata"]["semantic_frames"]
  datapoint["metadata"]["semantic_frames"] = list(set(semantic_frames))"""

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

# Create test datasets
class_0_test = class_0[:100]
class_1_test = class_1[:100]

# Drop the test samples from the main classes
class_0 = class_0[100:]
class_1 = class_1[100:]

# Create training and validation sets
class_0_training = class_0[:int(len(class_0) * 0.9)]  # 90%
class_0_validation = class_0[int(len(class_0) * 0.9):]  # 10%

class_1_training = class_1[:int(len(class_1) * 0.9)]  # 90%
class_1_validation = class_1[int(len(class_1) * 0.9):]  # 10%

# Merge datasets
dataset_training = class_0_training + class_1_training
dataset_validation = class_0_validation + class_1_validation
dataset_test = class_0_test + class_1_test

id_num = 0

# Create dataset for training
print("\nCreating dataset for training...")
for datapoint in tqdm(dataset_training, desc=f"Processing {len(dataset_training)} datapoints"):
  id_num += 1
  row = {
    "id": id_num,
    "text": datapoint["text"],
    "label": datapoint["label"],
  }
  if datapoint["metadata"]:
    row["metadata"] = datapoint["metadata"]

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_training)

# Create dataset for validation
print("\nCreating dataset for validation...")
for datapoint in tqdm(dataset_validation, desc=f"Processing {len(dataset_validation)} datapoints"):
  id_num += 1
  row = {
    "id": id_num,
    "text": datapoint["text"],
    "label": datapoint["label"],
  }
  if datapoint["metadata"]:
    row["metadata"] = datapoint["metadata"]

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_validation)

# Create dataset for test
print("\nCreating dataset for test...")
for datapoint in tqdm(dataset_test, desc=f"Processing {len(dataset_test)} datapoints"):
  id_num += 1
  row = {
    "id": id_num,
    "text": datapoint["text"],
    "label": datapoint["label"],
  }
  if datapoint["metadata"]:
    row["metadata"] = datapoint["metadata"]

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_test)
