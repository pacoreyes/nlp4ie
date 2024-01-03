import random

import spacy
from tqdm import tqdm

from db import spreadsheet_4
# from lib.ner_processing import custom_anonymize_text
from lib.utils2 import anonymize_text

from lib.utils import (read_from_google_sheet, write_to_google_sheet, save_row_to_jsonl_file, save_jsonl_file,
                       load_jsonl_file, empty_json_file)
from lib.utils2 import remove_duplicated_datapoints, balance_classes_in_dataset


# Load spacy model
nlp_trf = spacy.load("en_core_web_trf")

# Load dataset from Google Sheets
dataset = read_from_google_sheet(spreadsheet_4, "analize_frames")

# Label mapping
LABEL_MAP = {
  "0": "support",
  "1": "oppose"
}

SEED = 42
WITH_BALANCE = True
LABEL_CLASS_1 = LABEL_MAP["0"]
LABEL_CLASS_2 = LABEL_MAP["1"]

# Initialize path and name of output JSON-L files and Google Sheets
output_spreadsheet = "dataset_3"
output_anonym_spreadsheet = "dataset_3_anonym"
output_dataset = "shared_data/dataset_3_1.jsonl"
output_anonym_dataset = "shared_data/dataset_3_2_anonym.jsonl"
output_openai_dataset = "shared_data/dataset_3_3_openai.jsonl"
output_anonym_openai_dataset = "shared_data/dataset_3_4_openai_anonym.jsonl"


# Empty JSONL files
empty_json_file(output_dataset)
empty_json_file(output_anonym_dataset)
empty_json_file(output_openai_dataset)
empty_json_file(output_anonym_openai_dataset)

""" #############################################
Step 1: Remove duplicated datapoints
############################################# """

print("\nRemoving duplicated datapoints...")

dataset = remove_duplicated_datapoints(dataset)

""" #############################################
Step 2: Filter datapoints by label
############################################# """

dataset_remapped = []

print("Filter datapoints...")

for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  if datapoint["class"] == "0" or datapoint["class"] == "1":
    dataset_remapped.append(datapoint)

# Order dataset by label
dataset = sorted(dataset_remapped, key=lambda k: k['class'])

print("• Filtered datapoints")

""" #############################################
Step 3: Shuffle dataset
############################################# """

random.shuffle(dataset)

""" #############################################
Step 4: Create dataset in JSONL file and Google Sheets
############################################# """

print("\nCreating dataset...")

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

save_jsonl_file(dataset_no_anonym, output_dataset)

print("• Saved dataset to JSONL file")

""" #############################################
Step 5: Create anonymized version of the dataset, in JSONL file and Google Sheets
############################################# """

print("\nCreating anonymized version of the dataset...")

dataset_anonym = []

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

save_jsonl_file(dataset_anonym, output_anonym_dataset)

print("• Saved anonymized dataset to JSONL file")

""" #############################################
Step 6: Create dataset for Open AI in JSONL file
############################################# """

print("\nCreating dataset for Open AI...")

dataset = load_jsonl_file(output_dataset)

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
  save_row_to_jsonl_file(row_for_json, output_anonym_openai_dataset)
