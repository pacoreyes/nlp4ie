# from pprint import pprint

from google.api_core.retry import Retry
from tqdm import tqdm

from db import firestore_db
from lib.utils import save_jsonl_file, load_jsonl_file, empty_json_file
from lib.utils2 import remove_examples_in_dataset, remove_duplicated_datapoints

TO_GSHEETS = False
PAGE_SIZE = 10

collections = ["sentences", "sentences2", "sentences3", "sentences4"]

"""output_batch1 = "shared_data/dataset_2_unlabeled_batch1.jsonl"
output_batch2 = "shared_data/dataset_2_unlabeled_batch2.jsonl"
output_batch3 = "shared_data/dataset_2_unlabeled_batch3.jsonl"
output_batch4 = "shared_data/dataset_2_unlabeled_batch4.jsonl"

# Empty JSONL files
empty_json_file(output_batch1)
empty_json_file(output_batch2)
empty_json_file(output_batch3)
empty_json_file(output_batch4)"""

# Load training all datasets
dataset_training = load_jsonl_file("shared_data/dataset_3_1_training.jsonl")
dataset_validation = load_jsonl_file("shared_data/dataset_3_2_validation.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_3_3_test.jsonl")

# Combine all datasets
dataset_used = dataset_training + dataset_validation + dataset_test

for idx, collection in enumerate(collections):
  print("Processing collection:", collection)
  dataset3_col_ref = firestore_db.collection(collection)  # .limit(PAGE_SIZE)

  # retrieve documents from the collection using pagination
  print("Retrieving documents from Firestore...")
  docs = dataset3_col_ref.stream(retry=Retry())

  # Convert documents to list of dicts
  temp_sentences = [doc.to_dict() for doc in docs]

  print(f"Retrieved {len(temp_sentences)} documents from collection {collection}.")

  # Remove datapoints that are already present in the datasets used for training, validation, and testing
  print("Removing datapoints that are already present in the datasets used for training, validation, and testing...")
  dataset_deduplicated1 = remove_examples_in_dataset(temp_sentences, dataset_used)
  dataset_deduplicated2 = remove_examples_in_dataset(dataset_deduplicated1, dataset_used)
  dataset_deduplicated3 = remove_examples_in_dataset(dataset_deduplicated2, dataset_used)
  dataset_deduplicated4 = remove_examples_in_dataset(dataset_deduplicated3, dataset_used)
  dataset_deduplicated = remove_examples_in_dataset(dataset_deduplicated4, dataset_used)

  print("Removing duplicated datapoints...")
  dataset_clean = remove_duplicated_datapoints(dataset_deduplicated)

  # Remove unnecessary fields
  new_dataset = []
  for datapoint in tqdm(dataset_clean, desc=f"Processing {len(dataset_clean)} datapoints"):
    new_datapoint = {
      "id": datapoint["id"],
      "text": datapoint["text"],
      "target": datapoint["issue"],
    }
    new_dataset.append(new_datapoint)

  save_jsonl_file(new_dataset, f"shared_data/dataset_2_unlabeled_batch{idx + 1}.jsonl")

"""# Save documents in a Google Sheet
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
  write_to_google_sheet(spreadsheet_4, "delete", data)

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
save_jsonl_file(new_dataset, "shared_data/dataset_3_8_unlabeled_sentences_2.jsonl")
"""