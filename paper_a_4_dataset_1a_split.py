from collections import Counter
import random

import numpy as np
import torch

from lib.utils import load_jsonl_file, save_jsonl_file, empty_json_file
from lib.utils2 import split_stratify_dataset

SEED = 42


def set_seed(seed_value):
  """Set seed for reproducibility."""
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


# Set seed for reproducibility
set_seed(SEED)

# Set the path and name of the output JSON-L files
output_dataset_train = "shared_data/dataset_1_4_1a_train.jsonl"
output_dataset_test = "shared_data/dataset_1_4_1a_test.jsonl"

# Empty the output JSONL files
empty_json_file(output_dataset_train)
empty_json_file(output_dataset_test)

# Load the JSON file
dataset = load_jsonl_file("shared_data/dataset_1_2_1a_preprocessed.jsonl")
# Shuffle the dataset
random.shuffle(dataset)

# Count the number of monologic and dialogic texts using Counter
counter = Counter([datapoint["label"] for datapoint in dataset])
print(counter)

# Split the class for training, validation, and test sets
train_set, validation_set, test_set = split_stratify_dataset(dataset)

test_set = validation_set + test_set

# Sort the datasets by label
train_set = sorted(train_set, key=lambda x: x["label"])
test_set = sorted(test_set, key=lambda x: x["label"])

counter_train = Counter([datapoint["label"] for datapoint in train_set])
counter_test = Counter([datapoint["label"] for datapoint in test_set])

print("Train set:", counter_train)
print("Test set:", counter_test)

# Save the datasets
save_jsonl_file(train_set, output_dataset_train)
save_jsonl_file(test_set, output_dataset_test)
