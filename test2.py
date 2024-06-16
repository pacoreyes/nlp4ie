from pprint import pprint

from db import spreadsheet_4
from lib.utils import load_jsonl_file, write_to_google_sheet, read_from_google_sheet, load_json_file
from lib.utils2 import remove_duplicated_datapoints, remove_examples_in_dataset

# Load training all pools
pool_0 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch0.jsonl")
pool_1 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch1.jsonl")
pool_2 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch2.jsonl")
pool_3 = load_jsonl_file("shared_data/dataset_2_unlabeled_batch3.jsonl")

# Combine all pools
pool_all = pool_0 + pool_1 + pool_2 + pool_3

output_spreadsheet = "master2"

# dataset = load_jsonl_file("shared_data/dataset_2_test_master.jsonl")
# dataset = load_jsonl_file("shared_data/best_100_datapoints.jsonl")

dataset = read_from_google_sheet(spreadsheet_4, "master3")

completed_recs = []

for datapoint in pool_all:
  for rec in dataset:
    if rec["text"] == datapoint["text"]:
      completed_recs.append({
        "id": datapoint["id"],
        "text": datapoint["text"],
        "label": rec["label"],
      })

completed_recs = remove_duplicated_datapoints(completed_recs)

dataset_2 = read_from_google_sheet(spreadsheet_4, "dataset_3_final")
dataset_master = read_from_google_sheet(spreadsheet_4, "master")

completed_recs = remove_examples_in_dataset(completed_recs, dataset_2)
completed_recs = remove_examples_in_dataset(completed_recs, dataset_master)

completed_recs2 = []

for datapoint in completed_recs:
  row = [
    datapoint["id"],
    datapoint["text"],
    datapoint["label"],
  ]
  completed_recs2.append(row)

write_to_google_sheet(spreadsheet_4, output_spreadsheet, completed_recs2)
