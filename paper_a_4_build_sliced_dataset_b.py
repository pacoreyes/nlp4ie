import random
from collections import Counter
import os

import spacy
import tqdm
from transformers import BertTokenizer

from lib.utils import load_jsonl_file, empty_json_file, save_row_to_jsonl_file
from lib.utils2 import balance_classes_in_dataset, set_seed, split_stratify_dataset

SEED = 42

# Set seed for reproducibility
set_seed(SEED)

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_trf")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Load the JSON file
dataset = load_jsonl_file("shared_data/dataset_1_3_preprocessed_b.jsonl")
dataset_anonym = load_jsonl_file("shared_data/dataset_1_3_preprocessed_b_anonym.jsonl")

temp = "shared_data/_temp.jsonl"
temp_anonym = "shared_data/_temp_anonym.jsonl"

random.shuffle(dataset)
dataset = dataset[:20]

output_dataset_train = "shared_data/dataset_1_4_train.jsonl"
output_dataset_validation = "shared_data/dataset_1_4_validation.jsonl"
output_dataset_test = "shared_data/dataset_1_4_test.jsonl"

output_dataset_train_anonym = "shared_data/dataset_1_5_train_anonym.jsonl"
output_dataset_validation_anonym = "shared_data/dataset_1_5_validation_anonym.jsonl"
output_dataset_test_anonym = "shared_data/dataset_1_5_test_anonym.jsonl"

# Empty the output JSONL files
output_files = [output_dataset_train, output_dataset_validation, output_dataset_test, output_dataset_train_anonym,
                output_dataset_validation_anonym, output_dataset_test_anonym, temp, temp_anonym]
for file in output_files:
  empty_json_file(file)

datapoint_id = 0  # Initialize a counter for the unique sequential id
monologic_counter = 0
dialogic_counter = 0

# Process each text
for idx, text_doc in tqdm.tqdm(enumerate(dataset), desc=f"Processing {len(dataset)} datapoints", total=len(dataset)):
  idx = int(idx)
  # print(f"Processing text {idx + 1}/{dataset}")
  doc = nlp(text_doc['text'])
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
      if len(tokenizer.tokenize(" ".join(window))) < 300:
        window = [sentences[i]]
        continue
      # Join the sentences in the current window to form a single text, and add it to the new dataset
      new_text = " ".join(window)
      datapoint_id += 1  # Increment the unique id for each new datapoint
      datapoint = {
        'id': datapoint_id,
        'text': new_text,
        'label': text_doc['label'],
        'metadata': text_doc['metadata']
      }
      save_row_to_jsonl_file(datapoint, temp)
      save_row_to_jsonl_file(dataset_anonym[idx], temp_anonym)

      if datapoint['label'] == 0:
        monologic_counter += 1
      else:
        dialogic_counter += 1

      # Start a new window with the last sentence of the previous window
      window = [sentences[i]]
  # Add the remaining sentences in the last window to the new dataset
  if window and len(tokenizer.tokenize(" ".join(window))) >= 450:
    new_text = " ".join(window)
    datapoint_id += 1  # Increment the unique id for each new datapoint
    datapoint = {
      'id': datapoint_id,
      'text': new_text,
      'label': text_doc['label'],
      'metadata': text_doc['metadata']
    }
    # save_row_to_jsonl_file(datapoint, output_file)
    save_row_to_jsonl_file(datapoint, temp)
    save_row_to_jsonl_file(dataset_anonym[idx], temp_anonym)

    if datapoint['label'] == 0:
      monologic_counter += 1
    else:
      dialogic_counter += 1

# temp_dataset = load_jsonl_file(temp)
# temp_dataset_anonym = load_jsonl_file(temp_anonym)

# dataset = balance_classes_in_dataset(temp_dataset, "monologic", "dialogic", "label", SEED)

"""# Count the number of monologic and dialogic datapoints
counter = Counter([item['label'] for item in load_jsonl_file(output_file)])
monologic_percentage = counter["monologic"] / (counter["monologic"] + counter["dialogic"]) * 100
dialogic_percentage = counter["dialogic"] / (counter["monologic"] + counter["dialogic"]) * 100

print()
print("The original dataset has been split:")
print(f"\n• Monologic: {counter['monologic']} ({monologic_percentage:.2f})")
print(f"• Dialogic: {counter['dialogic']} ({dialogic_percentage:.2f})\n")

# Split the dataset into train, dev and test sets
split_stratify_dataset(dataset, "dataset1")
# Remove output_file from storage
os.remove(output_file)

print("Split dataset into train, dev and test sets:")
print(f"• Train: {len(load_jsonl_file('shared_data/dataset1_train.jsonl'))}")
print(f"• Validation: {len(load_jsonl_file('shared_data/dataset1_validation.jsonl'))}")
print(f"• Test: {len(load_jsonl_file('shared_data/dataset1_test.jsonl'))}")"""
