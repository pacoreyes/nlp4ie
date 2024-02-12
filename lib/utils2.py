import re
import random
import numpy as np
# from pprint import pprint

from tqdm import tqdm
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from lib.ner_processing import custom_anonymize_text
# from lib.utils import save_jsonl_file, load_jsonl_file, empty_json_file


def set_seed(seed_value):
  """Set seed for reproducibility."""
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def remove_duplicated_datapoints(dataset, verbose=False):
  unique_texts = set()
  processed_dataset = []
  
  for item in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints", total=len(dataset)):
    if item['text'] not in unique_texts:
      unique_texts.add(item['text'])
      processed_dataset.append(item)
    else:
      if verbose:
        print(f"Duplicate found: {item['id']}")
  
  print(
    f"Dataset after duplicate removal: {len(dataset) - len(processed_dataset)}  = {len(processed_dataset)}\n")

  return processed_dataset


def anonymize_text(text, nlp):

  more_entities = [
    "COVID-19",
    "COVID",
    "Army",
    "WeCanDoThis.HHS.gov",
    "HIV",
    "AIDS"
  ]

  entity_labels = [
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "LAW",
    "DATE",
    "TIME",
    "MONEY",
    "QUANTITY"
  ]

  # Anonymize text
  text = custom_anonymize_text(text, nlp, entity_labels)

  # Create a regular expression pattern that matches the entities with case-insensitivity
  pattern = re.compile('|'.join(re.escape(entity) for entity in more_entities), re.IGNORECASE)

  # Replace matched entities with "[ENTITY]"
  text = pattern.sub('[ENTITY]', text)

  return text


def balance_classes_in_dataset(dataset, label1, label2, label_name, seed=False):
  """
  Balance the dataset by truncating the majority class to the same size as the minority class.
  :param dataset: list of dictionaries containing the dataset.
  :param label1: str, label of the first class.
  :param label2: str, label of the second class.
  :param label_name: str, name of the label key in the dictionary.
  :param seed: optional, int, seed for reproducibility.
  :return:
  """
  if seed:
    set_seed(seed)
    # shuffle dataset
    # random.shuffle(dataset)

  # Balance dataset
  class1 = [item for item in dataset if item[label_name] == label1]
  class2 = [item for item in dataset if item[label_name] == label2]

  # Determine which class is the minority
  if len(class1) < len(class2):
    minority_class = class1
  else:
    minority_class = class2

  # Determine the maximum number of datapoints to keep
  max_num_datapoints = len(minority_class)

  # Truncate class1 class to max_num_datapoints
  class1 = class1[:max_num_datapoints]
  # Truncate class2 class to max_num_datapoints
  class2 = class2[:max_num_datapoints]

  # Concatenate class1 and class2 classes
  balanced_dataset = class1 + class2

  print(f"Balanced dataset: {len(balanced_dataset)} datapoints")
  print(f"• {label1}: {len(class1)} datapoints")
  print(f"• {label2}: {len(class2)} datapoints")

  return balanced_dataset


def remove_examples_in_dataset(source_list, filter_list):
  """
  Removes any elements from source_list that are present in filter_list.

  :param source_list: List to be cleaned.
  :param filter_list: List containing elements to remove from source_list.
  :return: A new list which is a cleaned version of source_list.
  """
  # Create a set of texts from filter_list for faster look-up
  filter_texts = {dp["text"] for dp in filter_list}

  # Use list comprehension to create a new list excluding filtered elements
  cleaned_list = [dp for dp in source_list if dp["text"] not in filter_texts]

  return cleaned_list


def split_stratify_dataset(dataset, stratify_key='label'):
  """
  Splits the dataset into training, validation, and test sets in a stratified manner,
  maintaining the original order. The split proportions are 80% training, 10% validation, and 10% test.

  Parameters:
  - dataset: List of dictionaries, where each dictionary represents a data point.
  - stratify_key: The key in the dictionaries used for stratifying the split.

  Returns:
  - train_data: Training dataset (80% of the original dataset).
  - validation_data: Validation dataset (10% of the original dataset).
  - test_data: Test dataset (10% of the original dataset).
  """
  # Group data by the stratify key
  from collections import defaultdict
  grouped_data = defaultdict(list)
  for item in dataset:
    key = item[stratify_key]
    grouped_data[key].append(item)

  # Initialize the splits
  train_data, validation_data, test_data = [], [], []

  # Determine split indices for each group and distribute the data accordingly
  for _, items in grouped_data.items():
    n = len(items)
    train_end = int(n * 0.8)
    validation_end = train_end + int(n * 0.1)

    train_data.extend(items[:train_end])
    validation_data.extend(items[train_end:validation_end])
    test_data.extend(items[validation_end:])

  return train_data, validation_data, test_data
