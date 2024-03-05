# import functools

# import requests
# import spacy

from collections import Counter
from itertools import combinations

from db import firestore_db, spreadsheet_4
# from issues_matcher import match_issues
# from linguistic_utils import check_minimal_meaning
# from text_preprocessing import preprocess_text
from lib.utils import read_from_google_sheet, save_row_to_jsonl_file, empty_json_file, load_jsonl_file

from pprint import pprint

# Load dataset 3
# dataset3 = read_from_google_sheet(spreadsheet_4, "analize_frames")
dataset_training = load_jsonl_file("shared_data/datasets/3/bootstrap_1/dataset_3_4_training_anonym.jsonl")
dataset_validation = load_jsonl_file("shared_data/datasets/3/bootstrap_1/dataset_3_5_validation_anonym.jsonl")
dataset_testing = load_jsonl_file("shared_data/datasets/3/bootstrap_1/dataset_3_6_test_anonym.jsonl")
dataset_test = load_jsonl_file("shared_data/datasets/3/bootstrap_1/dataset_3_7_test_anonym.jsonl")

dataset = dataset_training + dataset_validation + dataset_testing + dataset_test

""" #######################################
1. Convert frames datatype, from string to list
########################################"""

new_dataset3 = []
for datapoint in dataset3:
  found_frames = datapoint["found_frames"].split(',')
  found_frames = [s.strip() for s in found_frames]
  datapoint["found_frames"] = found_frames
  # for matched_frames
  matched_frames = datapoint["matched_frames"].split(',')
  matched_frames = [s.strip() for s in matched_frames]
  datapoint["matched_frames"] = matched_frames
  new_dataset3.append(datapoint)


""" #######################################
2. Display classes distribution
########################################"""

# collect datapoints of class 0
dataset_3_class_0 = [row for row in new_dataset3 if row["class"] == "0"]
# collect datapoints of class 1
dataset_3_class_1 = [row for row in new_dataset3 if row["class"] == "1"]

print()
print("Dataset 3 distribution:")
pprint(f"Class SUPPORT (0): {len(dataset_3_class_0)}")
pprint(f"Class OPPOSE (1): {len(dataset_3_class_1)}")
print()


""" #######################################
3. Create new Dataset 3 JSON file and display concurrent features of each class
########################################"""

empty_json_file("shared_data/dataset3_analysis.jsonl")
dataset_3_jsonl = "shared_data/dataset3_analysis.jsonl"

# Initialize a Counter to count the occurrences of each feature
feature_counter_class_0 = Counter()
feature_counter_class_1 = Counter()

# Store rows while counting the occurrences of each feature
for datapoint in new_dataset3:
  if datapoint["class"] == "0":
    feature_counter_class_0.update(datapoint["found_frames"])
  elif datapoint["class"] == "1":
    feature_counter_class_1.update(datapoint["found_frames"])
  save_row_to_jsonl_file(datapoint, dataset_3_jsonl)

print("---------------------------------------")

print("Occurrences of each feature in class 0:")
class0 = feature_counter_class_0.most_common()
pprint(class0[:10])
print()

print("Occurrences of each feature in class 1:")
class1 = feature_counter_class_1.most_common()
pprint(class1[:10])
print()


""" #######################################
4. Display most concurrent frames
########################################"""

print("---------------------------------------")

# Count the co-occurrence of triples of frames for class 0
triple_co_occurrence_counts = Counter()
quad_co_occurrence_counts = Counter()

for datapoint in dataset_3_class_0:
  for combo in combinations(datapoint['found_frames'], 3):
    triple_co_occurrence_counts[combo] += 1
  for combo in combinations(datapoint['found_frames'], 4):
    triple_co_occurrence_counts[combo] += 1

# Sort the co-occurrences by their frequency
sorted_triple_co_occurrences = sorted(triple_co_occurrence_counts.items(), key=lambda x: x[1], reverse=True)
sorted_quad_co_occurrences = sorted(quad_co_occurrence_counts.items(), key=lambda x: x[1], reverse=True)

# Show the top co-occurring triples for class 0
print("Class 0: triple co-occurrence counts")
pprint(sorted_triple_co_occurrences[:10])
print()
print("Class 0: quad co-occurrence counts")
pprint(sorted_quad_co_occurrences[:10])
print("---")

# Count the co-occurrence of triples of frames for class 1
triple_co_occurrence_counts = Counter()
quad_co_occurrence_counts = Counter()

for datapoint in dataset_3_class_1:
  for combo in combinations(datapoint['found_frames'], 3):
    triple_co_occurrence_counts[combo] += 1
  for combo in combinations(datapoint['found_frames'], 4):
    triple_co_occurrence_counts[combo] += 1

# Sort the co-occurrences by their frequency
sorted_triple_co_occurrences = sorted(triple_co_occurrence_counts.items(), key=lambda x: x[1], reverse=True)
sorted_quad_co_occurrences = sorted(quad_co_occurrence_counts.items(), key=lambda x: x[1], reverse=True)

# Show the top co-occurring triples for class 1
print("Class 1: triple co-occurrence counts")
pprint(sorted_triple_co_occurrences[:10])
print()
print("Class 1: quad co-occurrence counts")
pprint(sorted_quad_co_occurrences[:10])

""" #######################################
5. Display most concurrent frames
########################################"""

print("---------------------------------------")

# remove already annotated datapoints
dataset3_final = [row for row in new_dataset3
                  if row["class"] != "N/A" and row["class"] != "1" and row["class"] != "0"]

# pprint(new_dataset_3_class_0)

dataset3_final = [row for row in dataset3_final
                  if "Performers_and_roles" in row["matched_frames"]
                  and "Leadership" in row["matched_frames"]
                  and "Law" in row["matched_frames"]]

for row in dataset3_final:
  pprint(row["id"])
