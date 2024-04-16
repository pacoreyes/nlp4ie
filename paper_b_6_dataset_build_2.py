import random
from pprint import pprint

from sklearn.model_selection import train_test_split

from db import spreadsheet_4
from lib.utils import read_from_google_sheet, save_jsonl_file, empty_json_file, load_jsonl_file

GOOGLE_SHEET = "dataset_3_final"

SEED = 42

output_train = "shared_data/dataset_2_train.jsonl"
output_val = "shared_data/dataset_2_validation.jsonl"
output_test = "shared_data/dataset_2_test.jsonl"

# Empty JSONL file
empty_json_file(output_train)
empty_json_file(output_val)
empty_json_file(output_test)

# Load dataset
dataset = read_from_google_sheet(spreadsheet_4, GOOGLE_SHEET)

# Load the 3 splits of the seed dataset
# Load datasets
dataset_training_route = "datasets/2/seed/dataset_2_train.jsonl"
dataset_validation_route = "datasets/2/seed/dataset_2_validation.jsonl"
dataset_test_route = "datasets/2//seed/dataset_2_test.jsonl"

seed_training = load_jsonl_file(dataset_training_route)
seed_validation = load_jsonl_file(dataset_validation_route)
seed_test = load_jsonl_file(dataset_test_route)

seed_dataset = seed_training + seed_validation + seed_test

# split into 2 classes

seed_support = [datapoint for datapoint in seed_dataset if datapoint["label"] == "support"]
seed_oppose = [datapoint for datapoint in seed_dataset if datapoint["label"] == "oppose"]

print(f"Seed support: {len(seed_support)}")
print(f"Seed oppose: {len(seed_oppose)}")
print("---")

support = [datapoint for datapoint in dataset if datapoint["label"] == "support" and datapoint["valid"] == "TRUE"]
oppose = [datapoint for datapoint in dataset if datapoint["label"] == "oppose" and datapoint["valid"] == "TRUE"]

support = support + seed_support
oppose = oppose + seed_oppose

random.shuffle(support)
random.shuffle(oppose)

# Use sklearn train_test_split to split the support class in train 80%, validation 10%, test 10%
support_train, temp_set = train_test_split(support, test_size=0.2, random_state=42)
support_val, support_test = train_test_split(temp_set, test_size=0.5, random_state=42)

# Use sklearn train_test_split to split the oppose class in train 80%, validation 10%, test 10%
oppose_train, temp_set = train_test_split(oppose, test_size=0.2, random_state=42)
oppose_val, oppose_test = train_test_split(temp_set, test_size=0.5, random_state=42)

train_dataset = support_train + oppose_train
val_dataset = support_val + oppose_val
test_dataset = support_test + oppose_test

print(f"Train: {len(train_dataset)}")
print(f"Val: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")
print("---")
print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")

# Save datasets
save_jsonl_file(train_dataset, output_train)
save_jsonl_file(val_dataset, output_val)
save_jsonl_file(test_dataset, output_test)
