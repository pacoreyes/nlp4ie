from db import spreadsheet_4
from lib.utils import load_jsonl_file, write_to_google_sheet, read_from_google_sheet, load_json_file
from lib.utils2 import remove_duplicated_datapoints, remove_examples_in_dataset
from lib.text_utils import expand_contractions


dataset3 = read_from_google_sheet(spreadsheet_4, "dataset_3_final")
dataset_master = read_from_google_sheet(spreadsheet_4, "master")

dataset_master = remove_duplicated_datapoints(dataset3)

"""dataset_no_contractions = []
for rec in dataset_master:
  rec["text"] = expand_contractions(rec["text"])
  dataset_no_contractions.append(rec)

dataset3 = remove_examples_in_dataset(dataset_no_contractions, dataset3)

cleaned_dataset = []
for rec in dataset3:
  row = [
    rec["id"],
    rec["text"],
    rec["label"],
  ]
  cleaned_dataset.append(row)

write_to_google_sheet(spreadsheet_4, "master2", cleaned_dataset)"""
