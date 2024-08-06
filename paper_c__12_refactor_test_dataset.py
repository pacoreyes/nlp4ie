from db import spreadsheet_7
from lib.utils import load_jsonl_file, read_from_google_sheet, empty_json_file, save_row_to_jsonl_file

output_file = "shared_data/topic_boundary_test2.jsonl"
empty_json_file(output_file)
test_dataset = read_from_google_sheet(spreadsheet_7, "_test_dataset")  # true
"""continue_dataset = read_from_google_sheet(spreadsheet_7, "continue")
not_continue_dataset = read_from_google_sheet(spreadsheet_7, "not_continue")"""

counter_continue = 0
counter_not_continue = 0

new_dataset = []

for idx, datapoint in enumerate(test_dataset, start=1):
  new_datapoint = {
    "id": idx,
    "text": datapoint["text"],
    "label": datapoint["label"],
  }
  new_dataset.append(new_datapoint)
  if new_datapoint["label"] == "continue":
    counter_continue += 1
  else:
    counter_not_continue += 1

print(counter_continue)
print(counter_not_continue)

for datapoint in new_dataset:
  save_row_to_jsonl_file(datapoint, output_file)

"""counter = 0

continue_class = []
not_continue_class = []

for datapoint in test_dataset:
  for continue_datapoint in continue_dataset:
    if datapoint["id"] == continue_datapoint["id"] and datapoint["passage_id"] == continue_datapoint["passage_id"]:
      counter += 1
      continue_class.append({
        "text": datapoint["text"],
        "label": datapoint["label"],
      })

print(counter)

for datapoint in test_dataset:
  for not_continue_datapoint in not_continue_dataset:
    if datapoint["id"] == not_continue_datapoint["id"] and datapoint["passage_id"] == not_continue_datapoint["passage_id"]:
      counter += 1

print(counter)"""
