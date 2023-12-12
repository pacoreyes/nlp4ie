import random
from tqdm import tqdm

import spacy
from transformers import BertTokenizer

from utils.ner_processing import custom_anonymize_text
from utils.utils import save_row_to_jsonl_file, empty_json_file, load_jsonl_file

"""
NOTE: install the following spaCy model before running this script:

python -m spacy download en_core_web_trf 
"""

# Load spacy model
nlp_trf = spacy.load("en_core_web_trf")

# Initialize path and name of output JSON-L file
output_file = "shared_data/dataset_2_2_pair_sentences.jsonl"

# Initialize a JSONL file for the dataset
empty_json_file(output_file)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

# Load all passage records from the JSON file
dataset2_raw = load_jsonl_file("shared_data/dataset_2_1_raw.jsonl")

WITH_ANONYMIZATION = True
SEED = 42

# Initialize unique ID counter for datapoints
datapoint_id = 0

# Initialize counters
exceeded_token_limit = 0
continue_class_counter = 0
not_continue_class_counter = 0

# Initialize dataset
dataset = []

# Initialize ignored annotators
ignore_list = ["IE-Reyes", "IE-Asgola"]

# Set seed for reproducibility
random.seed(SEED)

for passage in tqdm(dataset2_raw, desc=f"Processing {len(dataset2_raw)} passages"):
  if passage["metadata"]["annotator"] in ignore_list:
    continue
  # Get sentences data
  sentences = passage["text"]

  for idx, sentence in enumerate(sentences):

    # Proceed only if there's a next sentence
    if idx < len(sentences) - 1:
      # Modify role to 'inside' if it's 'beginning'
      if sentences[idx + 1]["role"] == "beginning":
        sentences[idx + 1]["role"] = "inside"
      next_sentence = sentences[idx + 1]

      # Form pair with padding and check max length
      pair = "[CLS] " + sentence["sentence"] + " [SEP] " + next_sentence["sentence"] + " [SEP]"
      if len(tokenizer.tokenize(pair)) > 512:
        exceeded_token_limit += 1
        continue

      if WITH_ANONYMIZATION:
        # Anonymize text
        more_entities = ["COVID-19", "COVID", "Army", "WeCanDoThis.HHS.gov", "HIV"]
        pair = custom_anonymize_text(pair, nlp_trf,
                                     ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "LAW", "DATE",
                                      "TIME", "MONEY", "QUANTITY"])

        for entity in more_entities:
          pair = pair.replace(entity, "[ENTITY]")

      # Assign labels based on roles
      label = None
      if sentence["role"] == "inside" and next_sentence["role"] == "inside":
        label = "continue"
        continue_class_counter += 1
      elif sentence["role"] == "outside" and next_sentence["role"] == "inside":
        label = "not_continue"
        not_continue_class_counter += 1
      elif sentence["role"] == "inside" and next_sentence["role"] == "outside":
        label = "not_continue"
        not_continue_class_counter += 1

      # Only create and save datapoint if label is assigned
      if label:
        datapoint_id += 1
        datapoint = {
          "id": datapoint_id,
          "passage_id": passage["id"],
          "text": pair,
          "label": label,
          "annotator": passage["metadata"]["annotator"],
        }
        # print(f"Processing datapoint ({datapoint_id}): {datapoint['text']}")
        dataset.append(datapoint)

# prune "continue" class for representation balance
dataset_continue = []
dataset_not_continue = []
for datapoint in tqdm(dataset, desc="Pruning 'continue' class"):
  if datapoint["label"] == "continue":
    dataset_continue.append(datapoint)
  else:
    dataset_not_continue.append(datapoint)
# Shuffle the dataset with label "continue" before pruning to create diversity
random.shuffle(dataset_continue)
# Prune 13% of the dataset with label "continue"
dataset_continue = dataset_continue[:int(len(dataset_continue) * 0.87)]

dataset = dataset_continue + dataset_not_continue

# Save and print the datapoint
for datapoint in tqdm(dataset, desc="Saving dataset"):
  save_row_to_jsonl_file(datapoint, output_file)

print("\nClass distribution:")
print(f"• Continue: {len(dataset_continue)} (before prune {continue_class_counter})")
print(f"• Not continue: {not_continue_class_counter}")

print(f"\nExceeded token limit: {exceeded_token_limit}")
print(f"Ignored annotators: {ignore_list}")
