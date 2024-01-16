import random

import spacy
from tqdm import tqdm
from pprint import pprint

from db import spreadsheet_4
# from lib.ner_processing import custom_anonymize_text
from lib.utils2 import anonymize_text
from lib.utils import (read_from_google_sheet, save_row_to_jsonl_file, empty_json_file)
from lib.utils2 import remove_duplicated_datapoints

SEED = 42
FOR_OPEN_AI = True

# Set seed for reproducibility
random.seed(SEED)

# Load spacy model
nlp_trf = spacy.load("en_core_web_trf")

# Load dataset from Google Sheets
dataset3_1 = read_from_google_sheet(spreadsheet_4, "dataset_3")
dataset3_2 = read_from_google_sheet(spreadsheet_4, "dataset_3_")

print("Combine datasets...")
dataset = []
accepted_classes = ["support", "oppose", "neutral"]
for datapoint in tqdm(dataset3_2, desc=f"Processing {len(dataset)} datapoints"):
  if datapoint["class"] in accepted_classes:
    dataset.append(datapoint)

# Join dataset and dataset3_1
dataset = dataset + dataset3_1

# Label mapping
LABEL_MAP = {
  "0": "support",
  "1": "oppose",
  "2": "neutral"
}

#WITH_BALANCE = True
LABEL_CLASS_1 = LABEL_MAP["0"]
LABEL_CLASS_2 = LABEL_MAP["1"]
LABEL_CLASS_3 = LABEL_MAP["2"]

# Initialize path and name of output JSON-L files and Google Sheets
# output_spreadsheet = "dataset_3"
# output_anonym_spreadsheet = "dataset_3_anonym"
# output_dataset = "shared_data/dataset_3_1.jsonl"
# output_anonym_dataset = "shared_data/dataset_3_2_anonym.jsonl"
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
class_2 = [item for item in dataset if item["class"] == "neutral"]

trim_length = 60
class_0 = class_0[:trim_length]
class_1 = class_1[:trim_length]
class_2 = class_2[:trim_length]

print(f"• Class 0: {len(class_0)}")
print(f"• Class 1: {len(class_1)}")
print(f"• Class 2: {len(class_2)}")

# Shuffle datasets
random.shuffle(class_0)
random.shuffle(class_1)
random.shuffle(class_2)

class_0_training = class_0[:int(len(class_0) * 0.8)]  # 80%
class_0_validation = class_0[int(len(class_0) * 0.8):int(len(class_0) * 0.9)]  # 10%
class_0_test = class_0[int(len(class_0) * 0.9):]  # 10%

class_1_training = class_1[:int(len(class_1) * 0.8)]  # 80%
class_1_validation = class_1[int(len(class_1) * 0.8):int(len(class_1) * 0.9)]  # 10%
class_1_test = class_1[int(len(class_1) * 0.9):]  # 10%

class_2_training = class_2[:int(len(class_2) * 0.8)]  # 80%
class_2_validation = class_2[int(len(class_2) * 0.8):int(len(class_2) * 0.9)]  # 10%
class_2_test = class_2[int(len(class_2) * 0.9):]  # 10%

# Merge datasets
dataset_training = class_0_training + class_1_training + class_2_training
dataset_validation = class_0_validation + class_1_validation + class_2_validation
dataset_test = class_0_test + class_1_test + class_2_test

# Create parallel anonymized datasets
dataset_training_anonym = []
dataset_validation_anonym = []
dataset_test_anonym = []


# Create dataset for training
print("\nCreating dataset for training...")
for datapoint in tqdm(dataset_training, desc=f"Processing {len(dataset_training)} datapoints"):
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
  if FOR_OPEN_AI:
    row = {"prompt": row["text"], "completion": row["label"]}
    row_anonym = {"prompt": row_anonym["text"], "completion": row_anonym["label"]}

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_training)
  save_row_to_jsonl_file(row_anonym, output_dataset_training_anonym)

# Create dataset for validation
print("\nCreating dataset for validation...")
for datapoint in tqdm(dataset_validation, desc=f"Processing {len(dataset_validation)} datapoints"):
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
  if FOR_OPEN_AI:
    row = {"prompt": row["text"], "completion": row["label"]}
    row_anonym = {"prompt": row_anonym["text"], "completion": row_anonym["label"]}

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_validation)
  save_row_to_jsonl_file(row_anonym, output_dataset_validation_anonym)

# Create dataset for test
print("\nCreating dataset for test...")
for datapoint in tqdm(dataset_test, desc=f"Processing {len(dataset_test)} datapoints"):
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
  if FOR_OPEN_AI:
    row = {"prompt": row["text"], "completion": row["label"]}
    row_anonym = {"prompt": row_anonym["text"], "completion": row_anonym["label"]}

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_test)
  save_row_to_jsonl_file(row_anonym, output_dataset_test_anonym)


"""# Save class 2 to test dataset
for datapoint in tqdm(class_2, desc=f"Processing {len(class_2)} datapoints"):
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
  if FOR_OPEN_AI:
    row = {"prompt": row["text"], "completion": row["label"]}
    row_anonym = {"prompt": row_anonym["text"], "completion": row_anonym["label"]}

  # Save datapoint to JSONL files
  save_row_to_jsonl_file(row, output_dataset_test)
  save_row_to_jsonl_file(row_anonym, output_dataset_test_anonym)
"""

"""# Order dataset by label
dataset = sorted(dataset_remapped, key=lambda k: k['class'])"""

# print("• Filtered datapoints")

""" #############################################
Step 3: Shuffle dataset
############################################# """

# random.shuffle(dataset)

""" #############################################
Step 4: Create dataset in JSONL file and Google Sheets
############################################# """

"""print("\nCreating dataset...")

dataset_no_anonym = []

for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  row_for_json = {
    "id": datapoint["id"],
    "text": datapoint["text"],
    "label": LABEL_MAP[datapoint["class"]]
  }
  row_for_gsheets = [
    datapoint["id"],
    datapoint["text"],
    LABEL_MAP[datapoint["class"]]
  ]
  # Save datapoint to JSONL file
  save_row_to_jsonl_file(row_for_json, output_dataset)
  # Add datapoint to Google Sheets dataset
  dataset_no_anonym.append(row_for_gsheets)

# Save dataset to Google Sheets
write_to_google_sheet(spreadsheet_4, output_spreadsheet, dataset_no_anonym)
print("• Saved non-anonymized dataset to spreadsheet")

dataset_no_anonym = load_jsonl_file(output_dataset)

# Balance classes
if WITH_BALANCE:
  dataset_no_anonym = balance_classes_in_dataset(dataset_no_anonym, LABEL_CLASS_1, LABEL_CLASS_2, "label", SEED)

save_jsonl_file(dataset_no_anonym, output_dataset)"""

# print("• Saved dataset to JSONL file")

""" #############################################
Step 5: Create anonymized version of the dataset, in JSONL file and Google Sheets
############################################# """

# print("\nCreating anonymized version of the dataset...")

"""dataset_anonym = []

for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  datapoint["text"] = anonymize_text(datapoint["text"], nlp_trf)
  row_for_json = {
    "id": datapoint["id"],
    "text": datapoint["text"],
    "label": LABEL_MAP[datapoint["class"]]
  }
  row_for_gsheets = [
    datapoint["id"],
    datapoint["text"],
    LABEL_MAP[datapoint["class"]]
  ]
  # Save anonymized datapoint to JSONL file
  save_row_to_jsonl_file(row_for_json, output_anonym_dataset)
  # Add anonymized datapoint to Google Sheets dataset
  dataset_anonym.append(row_for_gsheets)

# Save anonymized dataset to Google Sheets
write_to_google_sheet(spreadsheet_4, output_anonym_spreadsheet, dataset_anonym)
print("• Saved anonymized dataset to spreadsheet")

dataset_anonym = load_jsonl_file(output_anonym_dataset)

# Balance classes
if WITH_BALANCE:
  dataset_anonym = balance_classes_in_dataset(dataset_anonym, LABEL_CLASS_1, LABEL_CLASS_2, "label", SEED)

save_jsonl_file(dataset_anonym, output_anonym_dataset)"""

# print("• Saved anonymized dataset to JSONL file")

""" #############################################
Step 6: Create dataset for Open AI in JSONL file
############################################# """

# print("\nCreating dataset for Open AI...")

"""dataset = load_jsonl_file(output_dataset)

# Balance classes
if WITH_BALANCE:
  dataset = balance_classes_in_dataset(dataset, LABEL_CLASS_1, LABEL_CLASS_2, "label", SEED)

class1_length = len([item for item in dataset if item["label"] == LABEL_CLASS_1])
class2_length = len([item for item in dataset if item["label"] == LABEL_CLASS_2])

# remap attribute names output_openai_dataset
dataset = [{"prompt": item["text"], "completion": item["label"]} for item in dataset]
save_jsonl_file(dataset, output_openai_dataset)

print("• Saved dataset for Open AI to JSONL file")
print(f"• Class 1: {class1_length}")
print(f"• Class 2: {class2_length}")

# Create anonymized version of the dataset for Open AI
for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  row_for_json = {
    "prompt": anonymize_text(datapoint["prompt"], nlp_trf),
    "completion": datapoint["completion"]
  }
  # Save datapoint to JSONL file
  save_row_to_jsonl_file(row_for_json, output_anonym_openai_dataset)"""
