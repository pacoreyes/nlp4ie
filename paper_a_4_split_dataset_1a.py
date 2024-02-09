from collections import Counter

import random

from lib.utils import load_jsonl_file, save_jsonl_file
from lib.utils2 import set_seed, split_stratify_dataset

SEED = 42

# Set seed for reproducibility
set_seed(SEED)

# Load the JSON file
dataset = load_jsonl_file("shared_data/dataset_1_2_preprocessed_a.jsonl")
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
save_jsonl_file(train_set, "shared_data/dataset_1_4_1a_split_train.jsonl")
save_jsonl_file(test_set, "shared_data/dataset_1_4_1a_split_test.jsonl")
