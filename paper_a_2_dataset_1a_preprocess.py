"""
This script creates the preprocessed dataset 1A prepared for the rule-based model.
All political discourse texts are complete, and the speaker labels are anonymized with "ENTITY".
"""
# import spacy
import tqdm
from transformers import BertTokenizer

from lib.text_utils import remove_speaker_labels
from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file

# load spaCy's Transformer model
# nlp = spacy.load("en_core_web_trf")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# REMOVE_SPEAKER_LABELS = True

output_file = "shared_data/dataset_1_2_1a_preprocessed.jsonl"

# Load the JSONL file with all the datapoints
dataset = load_jsonl_file('shared_data/dataset_1_1_raw.jsonl')

# Empty the output JSONL
empty_json_file(output_file)

""" #######################################################################
Preprocess text
########################################################################"""

counter = 0
# Process all datapoints in Dataset 1
for idx, datapoint in tqdm.tqdm(enumerate(dataset),
                                desc=f"Processing {len(dataset)} datapoints", total=len(dataset)):
  text = datapoint["text"]
  # convert list of strings to a single string
  # text = " ".join(text)
  tokens = tokenizer.tokenize(text)
  num_tokens = len(tokens)
  if num_tokens < 450:
    continue

  # if REMOVE_SPEAKER_LABELS:
  text = remove_speaker_labels(text)
  slots = {
    "id": datapoint["id"],
    "text": text,
    "label": datapoint["label"],
    "metadata": datapoint["metadata"]
  }

  save_row_to_jsonl_file(slots, output_file)
  counter += 1

print()
print("Processed", counter, "datapoints")
