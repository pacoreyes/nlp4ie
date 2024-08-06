import random
# from pprint import pprint

import numpy as np
from sklearn.model_selection import train_test_split

from db import spreadsheet_7
from lib.utils import read_from_google_sheet, write_to_google_sheet, save_jsonl_file

SEED = 42


def set_seed(seed_value):
  """Set seed for reproducibility."""
  random.seed(seed_value)
  np.random.seed(seed_value)


# Set seed for reproducibility
set_seed(SEED)

# Load the datasets
not_continue_dataset = read_from_google_sheet(spreadsheet_7, "not_continue")
continue_dataset = read_from_google_sheet(spreadsheet_7, "continue")

random.shuffle(not_continue_dataset)
random.shuffle(continue_dataset)

# Trim the continue dataset to match the size of the not_continue dataset
continue_dataset = continue_dataset[:len(not_continue_dataset)]

# Split the datasets: 80% train, 10% validation, 10% test
not_continue_train, not_continue_test = train_test_split(not_continue_dataset, test_size=0.2, random_state=SEED)
not_continue_val, not_continue_test = train_test_split(not_continue_test, test_size=0.5, random_state=SEED)

continue_train, continue_test = train_test_split(continue_dataset, test_size=0.2, random_state=SEED)
continue_val, continue_test = train_test_split(continue_test, test_size=0.5, random_state=SEED)

# Merge the datasets
train_dataset = not_continue_train + continue_train
val_dataset = not_continue_val + continue_val
test_dataset = not_continue_test + continue_test

print(f"Train: {len(train_dataset)}")
print(f"Valid: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")

# Save the datasets
data_test = []
for datapoint in test_dataset:
  row = [
    datapoint["id"],
    datapoint["passage_id"],
    datapoint["text"],
    datapoint["label"],
    datapoint["metadata"],
    datapoint["notes"]
  ]
  data_test.append(row)

data_valid = []
for datapoint in val_dataset:
  row = [
    datapoint["id"],
    datapoint["passage_id"],
    datapoint["text"],
    datapoint["label"],
    datapoint["metadata"],
    datapoint["notes"]
  ]
  data_valid.append(row)

data_train = []
for datapoint in train_dataset:
  row = [
    datapoint["id"],
    datapoint["passage_id"],
    datapoint["text"],
    datapoint["label"],
    datapoint["metadata"],
    datapoint["notes"]
  ]
  data_train.append(row)

write_to_google_sheet(spreadsheet_7, "_test_dataset", data_test)
write_to_google_sheet(spreadsheet_7, "_valid_dataset", data_valid)
write_to_google_sheet(spreadsheet_7, "_train_dataset", data_train)

# Process datasets: drop the notes and passage_id, add id and add index
index = None
for idx, datapoint in enumerate(test_dataset, start=1):
  datapoint["id"] = idx
  # Drop the passage_id and notes
  del datapoint["passage_id"]
  del datapoint["notes"]
  del datapoint["metadata"]
  index = idx

for idx, datapoint in enumerate(val_dataset, start=index+1):
  datapoint["id"] = idx
  # Drop the passage_id and notes
  del datapoint["passage_id"]
  del datapoint["notes"]
  del datapoint["metadata"]
  index = idx

for idx, datapoint in enumerate(train_dataset, start=index+1):
  datapoint["id"] = idx
  # Drop the passage_id and notes
  # if "passage_id" in datapoint:
  del datapoint["passage_id"]
  # if "notes" in datapoint:
  del datapoint["notes"]
  # if "metadata" in datapoint:
  del datapoint["metadata"]
  # index = idx

save_jsonl_file(test_dataset, "shared_data/topic_continuity_test.jsonl")
save_jsonl_file(val_dataset, "shared_data/topic_continuity_valid.jsonl")
save_jsonl_file(train_dataset, "shared_data/topic_continuity_train.jsonl")
