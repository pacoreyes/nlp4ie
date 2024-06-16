from pprint import pprint

from db import spreadsheet_4
from lib.utils import load_jsonl_file, write_to_google_sheet, read_from_google_sheet
# from lib.utils2 import remove_duplicated_datapoints

# Load training all pools
pool_0 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch0.jsonl")
pool_1 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch1.jsonl")
pool_2 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch2.jsonl")
pool_3 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch3.jsonl")

# Combine all pools
pool_all = pool_0 + pool_1 + pool_2 + pool_3

output_spreadsheet = "master3"

dataset = load_jsonl_file("shared_data/dataset_2_test_master.jsonl")

completed_recs = []

for datapoint in pool_all:
  for rec in dataset:
    if rec["text"] == datapoint["text"]:
      completed_recs.append({
        "id": rec["id"],
        "text": rec["text"],
        "label": rec["label"],
      })

write_to_google_sheet(spreadsheet_4, output_spreadsheet, completed_recs)


"""# Load used datasets
dataset_1_training = load_jsonl_file("datasets/3/bootstrap_1/dataset_3_1_training.jsonl")
dataset_1_validation = load_jsonl_file("datasets/3/bootstrap_1/dataset_3_2_validation.jsonl")
dataset_1_test = load_jsonl_file("datasets/3/bootstrap_1/dataset_3_3_test.jsonl")
dataset_1_test_master = load_jsonl_file("datasets/3/bootstrap_1/dataset_3_4_test_master.jsonl")

# Combine all used datasets
used_dataset = dataset_1_training + dataset_1_validation + dataset_1_test + dataset_1_test_master

# Load dataset from Google Sheets
dataset = read_from_google_sheet(spreadsheet_4, "dataset_3_final")
dataset2_ids = [rec["id"] for rec in dataset]
print(dataset2_ids)

data = []
for rec in used_dataset:
  if rec["id"] not in dataset2_ids:
    row = [
      rec["id"],
      rec["text"],
      rec["label"],
    ]
    data.append(row)

# Write data to Google Sheet
write_to_google_sheet(spreadsheet_4, output_spreadsheet, data)"""
