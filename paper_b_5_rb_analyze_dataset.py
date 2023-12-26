import ast
from collections import Counter

from db import spreadsheet_6
from lib.utils import read_from_google_sheet

# Load the dataset from the JSONL file
dataset = read_from_google_sheet(spreadsheet_6, "dataset_2_reclass")
# remove the first row
dataset = dataset[1:]

""" Describe the dataset """

""" A: Count the number of continuity types """

continuity_counter = Counter()

for datapoint in dataset:
  if datapoint["continuity"] != "[]":
    continuity_list = ast.literal_eval(datapoint["continuity"])
    for continuity in continuity_list:
      # Increment count for each type of continuity
      continuity_counter[list(continuity.keys())[0]] += 1

# order the counter by value
continuity_counter = dict(sorted(continuity_counter.items(), key=lambda item: item[1], reverse=True))

print("Continuity counter:")
for continuity, count in continuity_counter.items():
  print(f"• {continuity}: {count}")

""" B: Count the number of concurrent continuity types """

concurrent_continuity_counter = Counter()

for datapoint in dataset:
  if datapoint["continuity"] != "[]":
    continuity_list = ast.literal_eval(datapoint["continuity"])
    # Increment count for each type of continuity
    concurrent_continuity_counter[len(continuity_list)] += 1

# order the counter by value
concurrent_continuity_counter = dict(
  sorted(concurrent_continuity_counter.items(), key=lambda item: item[1], reverse=True))

print("\nConcurrent continuity counter:")
for continuity, count in concurrent_continuity_counter.items():
  print(f"• {continuity}: {count}")

""" C: Contingency Table displaying the frequency of each combination of features (continuity types) """

# Initialize a counter for the combinations
combinations_counter = Counter()

# Define the feature names
feature_names = ['lexical_continuity', 'syntactic_continuity', 'semantic_continuity',
                 'transition_markers_continuity', 'coreference']

for datapoint in dataset:
  if datapoint["continuity"] != "[]":
    # Convert the string representation of the list into a list
    continuity_list = ast.literal_eval(datapoint["continuity"])

    # Initialize a key to represent the combination of features in this datapoint
    combination_key = []

    """
    for continuity_dict in continuity_list:
      # Check for each feature's presence and add to the key accordingly
      for feature in feature_names:
        if feature in continuity_dict:
          combination_key.append(feature)
    """

    for continuity_dict in continuity_list:
      # Check for each feature's presence and add to the key accordingly
      for feature in feature_names:
        if feature in continuity_dict:
          if feature == "transition_markers_continuity":
            if any(m.get('type') == 'continue' and m.get('location') == 'flexible'
                   for m in continuity_dict["transition_markers_continuity"]):
              print(f"{datapoint['id']} :: {continuity_dict['transition_markers_continuity']}")
              combination_key.append(feature)
              continue
            """else:
              continue"""
          combination_key.append(feature)

    # Convert the combination into a string to count as a key
    combination_key_str = ' + '.join(
      sorted(set(combination_key)))

    # Increment the count for this combination in the counter
    combinations_counter[combination_key_str] += 1

# Print the results
print("Contingency Table of Feature Combinations:")
for combination, count in combinations_counter.items():
  print(f"{combination}: {count}")
