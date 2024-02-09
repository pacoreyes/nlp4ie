import re
import random
# from pprint import pprint

from tqdm import tqdm
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from lib.ner_processing import custom_anonymize_text
from lib.utils import save_jsonl_file, load_jsonl_file, empty_json_file


def set_seed(seed_value):
  """Set seed for reproducibility."""
  random.seed(seed_value)
  # np.random.seed(seed_value)
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


def balance_classes_in_dataset(dataset, label1, label2, label_name, seed):
  set_seed(seed)

  # shuffle dataset
  random.shuffle(dataset)

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


def split_stratify_dataset(json_objects, dataset_name):
  """
  Split a dataset into train, validation, and test sets, and save them as JSONL files.
  :param json_objects: dictionary containing the dataset
  :param dataset_name: root name of the dataset
  :return:
  """
  df = pd.DataFrame(json_objects)

  # Perform the stratified split
  train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
  # Split temp into validation and test
  validation_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

  # Convert DataFrames back into lists of JSON objects
  train_data = train_df.to_dict(orient='records')
  validation_data = validation_df.to_dict(orient='records')
  test_data = test_df.to_dict(orient='records')

  # Define file names based on the dataset name
  train_file_name = f"{dataset_name}_train.jsonl"
  validation_file_name = f"{dataset_name}_validation.jsonl"
  test_file_name = f"{dataset_name}_test.jsonl"

  # Save the datasets to JSONL files
  save_jsonl_file(train_data, f"shared_data/{train_file_name}")
  save_jsonl_file(validation_data, f"shared_data/{validation_file_name}")
  save_jsonl_file(test_data, f"shared_data/{test_file_name}")

  print(f"Dataset split and saved as {train_file_name}, {validation_file_name}, and {test_file_name}")
