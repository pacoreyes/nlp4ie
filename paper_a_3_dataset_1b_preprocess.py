"""
This script preprocesses the dataset 1B for the BERT model using the sliding window approach.
The output are two JSONL files:
1. Dataset with sliced texts with limit of 510 tokens (dataset_1_3_preprocessed_b.jsonl)
2. Dataset with anonymized texts with limit of 510 tokens (dataset_1_3_preprocessed_b_anonym.jsonl)
"""
import spacy
import tqdm
from transformers import BertTokenizer

from lib.linguistic_utils import check_minimal_meaning
from lib.ner_processing import anonymize_text
from lib.text_utils import preprocess_text, remove_speaker_labels
from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file

PREPROCESS_TEXT = True
REMOVE_SPEAKER_LABELS = True

# load spaCy's Transformer model
nlp = spacy.load("en_core_web_trf")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# load dataset
output_file = "shared_data/dataset_1_3_1b_preprocessed.jsonl"
# output_file_anonym = "shared_data/dataset_1_3_1b_preprocessed_anonym.jsonl"

# Load the JSONL file with all the datapoints
dataset = load_jsonl_file('shared_data/dataset_1_1_raw.jsonl')

# Empty the output JSONL
empty_json_file(output_file)
# empty_json_file(output_file_anonym)

# Initialize a list of entities not anonymized by spaCy
custom_entities = [
  "COVID-19",
  "COVID",
  "Army",
  "WeCanDoThis.HHS.gov",
  "HIV",
  "AIDS"
]

""" #######################################################################
Preprocess text
########################################################################"""

# Process all datapoints in Dataset 1

for idx, datapoint in tqdm.tqdm(enumerate(dataset),
                                desc=f"Processing {len(dataset)} datapoints", total=len(dataset)):
  text = datapoint["text"]

  tokens = tokenizer.tokenize(text)
  num_tokens = len(tokens)

  if num_tokens < 450:
    continue

  # convert single string to list of sentences using spaCy
  text = [sent.text for sent in nlp(text).sents]

  # Remove sentences without minimal meaning
  text = [sent for sent in text if check_minimal_meaning(nlp(sent))]

  new_text = []
  for sent in text:
    sent = preprocess_text(sent, nlp,
                           with_remove_known_unuseful_strings=False,
                           with_remove_parentheses_and_brackets=False,
                           with_remove_text_inside_parentheses=False,
                           with_remove_leading_patterns=True,
                           with_remove_timestamps=False,
                           with_replace_unicode_characters=True,
                           with_expand_contractions=False,
                           with_remove_links_from_text=False,
                           with_put_placeholders=False,
                           with_final_cleanup=False)

    if REMOVE_SPEAKER_LABELS:
      sent = remove_speaker_labels(sent)
    new_text.append(sent)

  # Text non-anonymized
  text = " ".join(new_text)

  # Text anonymized
  # text_anonym = anonymize_text(text, nlp)

  for entity in custom_entities:
    text = text.replace(entity, "ENTITY")

  row = {
    "id": datapoint["id"],
    "text": text,
    "label": datapoint["label"],
    "metadata": datapoint["metadata"]
  }
  save_row_to_jsonl_file(row, output_file)
  # row["text"] = text_anonym
  # save_row_to_jsonl_file(row, output_file_anonym)

print()
print("Process finished.")
