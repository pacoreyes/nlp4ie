import spacy
from tqdm import tqdm

from db import spreadsheet_4
from lib.utils import load_jsonl_file, write_to_google_sheet, save_row_to_jsonl_file, empty_json_file
from lib.utils2 import remove_examples_in_dataset, remove_duplicated_datapoints

# Set spaCy to use GPU if available
if spacy.prefer_gpu():
  print("spaCy is using GPU!")
else:
  print("GPU not available, spaCy is using CPU instead.")

# Load spacy model
nlp = spacy.load("en_core_web_trf")

# Initialize constants
GOOGLE_SHEET = "dataset_3_final"

first_person_pronouns = ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"]

second_person_pronouns = ["you", "your", "yours", "yourself", "yourselves"]

third_person_pronouns = ["he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                          "they", "them", "their", "theirs", "themselves"]

output_dataset = "datasets/2/seed/dataset_2_unlabeled_cleaned.jsonl"

# Empty JSONL files
empty_json_file(output_dataset)

# Load datasets
dataset_training = load_jsonl_file("datasets/2/seed/dataset_2_train.jsonl")
dataset_validation = load_jsonl_file("datasets/2/seed/dataset_2_validation.jsonl")
dataset_test = load_jsonl_file("datasets/2//seed/dataset_2_test.jsonl")

dataset_3 = dataset_training + dataset_validation + dataset_test

# Load training all pools
pool_0 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch0.jsonl")
pool_1 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch1.jsonl")
pool_2 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch2.jsonl")
pool_3 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch3.jsonl")

# Combine all pools
dataset = pool_0 + pool_1 + pool_2 + pool_3

dataset = remove_examples_in_dataset(dataset, dataset_3)
dataset = remove_duplicated_datapoints(dataset)

# dataset = dataset[:2000]

# Process dataset
data = []
for rec in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  doc = nlp(rec["text"])

  # Check if number of tokens (excluding punctuation) is less or equal to 15
  if len([token for token in doc if token.is_alpha]) <= 15:
    continue

  # Check if any of the tokens in the doc is in first_person_pronouns
  is_first_person = False
  is_second_person_or_third_person = False

  for token in doc:
    if token.text.lower() in first_person_pronouns:
      is_first_person = True
      break

  if is_first_person:

    # Check if any of the tokens in the doc is in second_person_pronouns
    for token in doc:
      if token.text.lower() in second_person_pronouns or token.text.lower() in third_person_pronouns:
        is_second_person_or_third_person = True
        break

    if not is_second_person_or_third_person:
      row = [
        rec["id"],
        rec["text"],
        rec["target"],
      ]
      data.append(row)
      save_row_to_jsonl_file(rec, output_dataset)

# Write data to Google Sheet
write_to_google_sheet(spreadsheet_4, GOOGLE_SHEET, data)
