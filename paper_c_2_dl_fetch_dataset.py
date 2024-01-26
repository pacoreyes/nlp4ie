# from pprint import pprint

from google.api_core.retry import Retry

from db import firestore_db, spreadsheet_4
from lib.utils import write_to_google_sheet, save_jsonl_file, load_jsonl_file
from lib.utils2 import remove_examples_in_dataset

TO_GSHEETS = False
PAGE_SIZE = 20000  # 2K is the max page size
START_AT = 0

dataset3_col_ref = firestore_db.collection("sentences")

# Create query with pagination and limit
query_start_at = dataset3_col_ref.order_by("id").start_at({"id": START_AT}).limit(PAGE_SIZE)

# retrieve documents from the collection using pagination
print("Retrieving documents from Firestore...")
docs = query_start_at.stream(retry=Retry())

# Convert documents to list of dicts
dataset = [doc.to_dict() for doc in docs]

# Load training all datasets
dataset_training = load_jsonl_file("shared_data/dataset_3_1_training.jsonl")
dataset_validation = load_jsonl_file("shared_data/dataset_3_2_validation.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_3_3_test.jsonl")

# Combine all datasets
dataset_used = dataset_training + dataset_validation + dataset_test

# Remove datapoints that are already present in the datasets used for training, validation, and testing
print("Removing datapoints that are already present in the datasets used for training, validation, and testing...")
dataset = remove_examples_in_dataset(dataset, dataset_used)

# Save documents in a Google Sheet
if TO_GSHEETS:
  print("Saving documents in a Google Sheet...")
  data = []
  for rec in dataset:
    row = [
      rec["id"],
      rec["text"],
      rec["issue"],
      # rec["main_frame"],
      # ', '.join(rec["semantic_frames"]),
    ]
    data.append(row)

  # Write data to Google Sheet
  write_to_google_sheet(spreadsheet_4, "dataset_3_", data)

# Remove unnecessary fields
new_dataset = []
for datapoint in dataset:
  new_datapoint = {
    "id": datapoint["id"],
    "text": datapoint["text"],
    "target": datapoint["issue"],
  }
  new_dataset.append(new_datapoint)


# Save documents in a JSONL file
print("Saving unlabeled sentences in a JSONL file...")
save_jsonl_file(new_dataset, "shared_data/dataset_3_7_unlabeled_sentences_1.jsonl")
