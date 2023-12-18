import random
# from pprint import pprint
from collections import Counter

from tqdm import tqdm
import spacy
from transformers import BertTokenizer

from lib.ner_processing import custom_anonymize_text
from lib.utils import save_row_to_jsonl_file, empty_json_file, load_jsonl_file

"""
NOTE: install the following spaCy model before running this script:

python -m spacy download en_core_web_trf 
"""

WITH_BALANCE = False

# Load spacy model
nlp_trf = spacy.load("en_core_web_trf")

# Initialize path and name of output JSON-L files
output_file = "shared_data/dataset_2_3_pair_sentences.jsonl"
output_file_anonym = "shared_data/dataset_2_4_pair_sentences_anonym.jsonl"

# Initialize a JSONL file for the dataset
empty_json_file(output_file)
empty_json_file(output_file_anonym)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

# Load all passage records from the JSON file
dataset2_raw = load_jsonl_file("shared_data/dataset2_raw_dec_16.jsonl")

SEED = 42

# Initialize unique ID counter for datapoints
datapoint_id = 0

# Initialize counters
exceeded_token_limit = 0
continue_class_counter = 0
not_continue_class_counter = 0

# Initialize dataset
dataset = []
dataset_anonym = []

# Initialize ignored annotators
ignore_list = ["IE-Reyes"]

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
      pair = sentence["sentence"] + " [SEP] " + next_sentence["sentence"]
      pair_anonym = "[CLS] " + sentence["sentence"] + " [SEP] " + next_sentence["sentence"] + " [SEP]"

      if len(tokenizer.tokenize(pair)) > 512 or len(tokenizer.tokenize(pair_anonym)) > 510:
        exceeded_token_limit += 1
        continue

      # Anonymize text
      more_entities = ["COVID-19", "COVID", "Army", "WeCanDoThis.HHS.gov", "HIV", "AIDS"]
      pair_anonym = custom_anonymize_text(pair_anonym, nlp_trf,
                                          ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
                                           "LAW", "DATE", "TIME", "MONEY", "QUANTITY"])

      for entity in more_entities:
        pair_anonym = pair_anonym.replace(entity, "[ENTITY]")

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
      elif sentence["role"] == "beginning" and next_sentence["role"] == "inside":
        label = "continue"
        continue_class_counter += 1
      elif sentence["role"] == "beginning" and next_sentence["role"] == "outside":
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
          "text_id": passage["metadata"]["text_id"]
        }
        datapoint_anonym = {
          "id": datapoint_id,
          "passage_id": passage["id"],
          "text": pair_anonym,
          "label": label,
          "annotator": passage["metadata"]["annotator"],
          "text_id": passage["metadata"]["text_id"]
        }
        dataset.append(datapoint)
        dataset_anonym.append(datapoint_anonym)
        save_row_to_jsonl_file(datapoint, output_file)
        save_row_to_jsonl_file(datapoint_anonym, output_file_anonym)
      else:
        print(f"Skipped datapoint with no label: {passage['id']}")

counter = Counter([item['label'] for item in dataset])
continue_percentage = counter["continue"] / (counter["continue"] + counter["not_continue"]) * 100
not_continue_percentage = counter["not_continue"] / (counter["continue"] + counter["not_continue"]) * 100

print()
print("The original dataset has been split:")
print(f"• Continue: {counter['continue']} ({continue_percentage:.2f}%)")
print(f"• Not continue: {counter['not_continue']} ({not_continue_percentage:.2f}%)")
