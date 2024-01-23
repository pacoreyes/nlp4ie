import re
import random
# from pprint import pprint

from tqdm import tqdm
import torch

from lib.ner_processing import custom_anonymize_text


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
    return [element for element in source_list if element not in filter_list]
