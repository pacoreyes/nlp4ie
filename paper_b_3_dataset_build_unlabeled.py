import spacy

from db import spreadsheet_4
from lib.utils import load_jsonl_file, write_to_google_sheet
from lib.utils2 import remove_duplicated_datapoints

nlp = spacy.load("en_core_web_trf")

# Load training all pools
pool_0 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch0.jsonl")
pool_1 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch1.jsonl")
pool_2 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch2.jsonl")
pool_3 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch3.jsonl")

# Combine all pools
pool_all = pool_0 + pool_1 + pool_2 + pool_3

# Load datasets iteration 1
dataset_1_training = load_jsonl_file("datasets/3/bootstrap_1/dataset_3_1_training.jsonl")
dataset_1_validation = load_jsonl_file("datasets/3/bootstrap_1/dataset_3_2_validation.jsonl")
dataset_1_test = load_jsonl_file("datasets/3/bootstrap_1/dataset_3_3_test.jsonl")
dataset_1_test_master = load_jsonl_file("datasets/3/bootstrap_1/dataset_3_4_test_master.jsonl")

# Combine all datasets iteration 1
dataset_1_all = dataset_1_training + dataset_1_validation + dataset_1_test + dataset_1_test_master

first_person_pronouns = ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"]

# Write to Google Sheets
data = []
for rec in dataset_1_all:
  doc = nlp(rec["text"])

  # Check if any of the tokens in the doc is in first_person_pronouns
  is_first_person = False
  for token in doc:
    if token.text.lower() in first_person_pronouns:
      is_first_person = True
      break

  if is_first_person:
    row = [
      rec["id"],
      rec["text"],
      rec["label"],
    ]
    data.append(row)

# Write data to Google Sheet
write_to_google_sheet(spreadsheet_4, "dataset_3*", data)
