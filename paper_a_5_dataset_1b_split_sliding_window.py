from collections import Counter
import os
import random
# from pprint import pprint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import spacy
from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np
import torch

from lib.utils import load_jsonl_file, empty_json_file, save_row_to_jsonl_file
from lib.utils2 import split_stratify_dataset, balance_classes_in_dataset
from lib.ner_processing import anonymize_text

# Set spaCy to use GPU if available
if spacy.prefer_gpu():
    print("spaCy is using GPU!")
else:
    print("GPU not available, spaCy is using CPU instead.")

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

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_trf")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Load the JSON file
dataset = load_jsonl_file("shared_data/dataset_1_3_1b_preprocessed.jsonl")

temp = "shared_data/_temp.jsonl"
temp_anonym = "shared_data/_temp_anonym.jsonl"

output_dataset_train = "shared_data/dataset_1_6_1b_train.jsonl"
output_dataset_validation = "shared_data/dataset_1_6_1b_validation.jsonl"
output_dataset_test = "shared_data/dataset_1_6_1b_test.jsonl"

output_dataset_train_anonym = "shared_data/dataset_1_6_1b_train_anonym.jsonl"
output_dataset_validation_anonym = "shared_data/dataset_1_6_1b_validation_anonym.jsonl"
output_dataset_test_anonym = "shared_data/dataset_1_6_1b_test_anonym.jsonl"

# Empty the output JSONL files
output_files = [output_dataset_train, output_dataset_validation, output_dataset_test, output_dataset_train_anonym,
                output_dataset_validation_anonym, output_dataset_test_anonym, temp, temp_anonym]
for file in output_files:
  empty_json_file(file)

datapoint_id = 0  # Initialize a counter for the unique sequential id
monologic_counter = 0
dialogic_counter = 0

# Process each text
for idx, datapoint in tqdm(enumerate(dataset), desc=f"Processing {len(dataset)} datapoints", total=len(dataset)):
  idx = int(idx)
  doc = nlp(datapoint['text'])
  sentences = [sent.text for sent in doc.sents]

  # Create windows with overlap
  window = []
  for i in range(len(sentences)):
    # Add the sentence to the current window
    window.append(sentences[i])
    # Check if the total number of tokens in the current window exceeds the BERT token limit
    if len(tokenizer.tokenize(" ".join(window))) > 510:  # Saving 2 tokens for padding
      # If it does, remove the last sentence from the window
      window.pop()
      # Check the number of tokens, skip if less than 300
      if len(tokenizer.tokenize(" ".join(window))) < 400:
        window = [sentences[i]]
        continue
      # Join the sentences in the current window to form a single text, and add it to the new dataset
      new_text = " ".join(window)
      datapoint_id += 1  # Increment the unique id for each new datapoint
      row = {
        'id': datapoint_id,
        'text': new_text,
        'label': datapoint['label'],
        'metadata': datapoint['metadata']
      }
      save_row_to_jsonl_file(row, temp)
      # Create a new row with the anonymized text
      row["text"] = anonymize_text(row["text"], nlp)
      # Save anonymized row
      save_row_to_jsonl_file(row, temp_anonym)

      if row['label'] == 0:
        monologic_counter += 1
      else:
        dialogic_counter += 1

      # Start a new window with the last sentence of the previous window
      window = [sentences[i]]
  # Add the remaining sentences in the last window to the new dataset
  if window and len(tokenizer.tokenize(" ".join(window))) >= 400:
    new_text = " ".join(window)
    datapoint_id += 1  # Increment the unique id for each new datapoint
    row = {
      'id': datapoint_id,
      'text': new_text,
      'label': datapoint['label'],
      'metadata': datapoint['metadata']
    }
    save_row_to_jsonl_file(row, temp)
    # Create a new row with the anonymized text
    row["text"] = anonymize_text(row["text"], nlp)
    # Save anonymized row
    save_row_to_jsonl_file(row, temp_anonym)

    if row['label'] == 0:
      monologic_counter += 1
    else:
      dialogic_counter += 1


temp_dataset = load_jsonl_file(temp)
temp_dataset_anonym = load_jsonl_file(temp_anonym)

temp_dataset = balance_classes_in_dataset(temp_dataset, "monologic", "dialogic", "label")
temp_dataset_anonym = balance_classes_in_dataset(temp_dataset_anonym, "monologic", "dialogic", "label")

# Count the number of monologic and dialogic datapoints
counter = Counter([item['label'] for item in temp_dataset])
monologic_percentage = counter["monologic"] / (counter["monologic"] + counter["dialogic"]) * 100
dialogic_percentage = counter["dialogic"] / (counter["monologic"] + counter["dialogic"]) * 100

counter_a = Counter([item['label'] for item in temp_dataset_anonym])
monologic_percentage_a = counter_a["monologic"] / (counter_a["monologic"] + counter_a["dialogic"]) * 100
dialogic_percentage_a = counter_a["dialogic"] / (counter_a["monologic"] + counter_a["dialogic"]) * 100

print()
print("Dataset distribution:")
print(f"• Monologic: {counter['monologic']} ({monologic_percentage:.2f})")
print(f"• Dialogic: {counter['dialogic']} ({dialogic_percentage:.2f})")
print(f"• Monologic anonym: {counter_a['monologic']} ({monologic_percentage_a:.2f})")
print(f"• Dialogic anonym: {counter_a['dialogic']} ({dialogic_percentage_a:.2f})\n")

# Stratify and split the datasets
train_set, validation_set, test_set = split_stratify_dataset(temp_dataset)
train_set_anonym, validation_set_anonym, test_set_anonym = split_stratify_dataset(temp_dataset_anonym)

# Save the split datasets
for idx, row in enumerate(train_set):
  save_row_to_jsonl_file(row, output_dataset_train)
for idx, row in enumerate(validation_set):
  save_row_to_jsonl_file(row, output_dataset_validation)
for idx, row in enumerate(test_set):
  save_row_to_jsonl_file(row, output_dataset_test)

# Save the split anonym datasets
for idx, row in enumerate(train_set_anonym):
  save_row_to_jsonl_file(row, output_dataset_train_anonym)
for idx, row in enumerate(validation_set_anonym):
  save_row_to_jsonl_file(row, output_dataset_validation_anonym)
for idx, row in enumerate(test_set_anonym):
  save_row_to_jsonl_file(row, output_dataset_test_anonym)


# Remove temporary files
os.remove(temp)
os.remove(temp_anonym)

print("\nSplit dataset into train, validation and test sets:")
print(f"• Train: {len(train_set)}")
print(f"• Validation: {len(validation_set)}")
print(f"• Test: {len(test_set)}")
print()
print(f"• Train anonym: {len(train_set_anonym)}")
print(f"• Validation anonym: {len(validation_set_anonym)}")
print(f"• Test anonym: {len(test_set_anonym)}\n")

print("Dataset distribution:")
counter_train = Counter([item['label'] for item in train_set])
counter_validation = Counter([item['label'] for item in validation_set])
counter_test = Counter([item['label'] for item in test_set])
print(f"• Train: {counter_train}")
print(f"• Validation: {counter_validation}")
print(f"• Test: {counter_test}")
print()
