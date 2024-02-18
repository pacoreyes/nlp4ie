# from pprint import pprint
import random

from tqdm import tqdm
import spacy

from db import firestore_db, spreadsheet_7
from lib.utils import read_from_google_sheet, save_row_to_jsonl_file, empty_json_file, firestore_timestamp_to_string
from lib.text_utils import preprocess_text
from lib.ner_processing import custom_anonymize_text


if spacy.prefer_gpu():
  print("spaCy is using GPU")
else:
  print("Using CPU")

ref_collection_passages = firestore_db.collection("passages")
ref_collection_text = firestore_db.collection("texts2")

# Global variables
SEED = 42

dataset = read_from_google_sheet(spreadsheet_7, "dataset_2_reclass_")
# dataset = dataset[:10]

# Set seed
random.seed(SEED)
# Shuffle dataset
random.shuffle(dataset)

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

# Initialize path and name of output JSON-L files
output_file = "shared_data/dataset_2_5_pair_sentences_reclass.jsonl"
output_file_anonym = "shared_data/dataset_2_5_pair_sentences_reclass_anonym.jsonl"

# Initialize a JSONL files
empty_json_file(output_file)
empty_json_file(output_file_anonym)

# Initialize counters
counter_continue = 0
counter_not_continue = 0

for idx, datapoint in enumerate(tqdm(dataset, desc=f"Generating Dataset 2")):
  # Retrieve passage record from Firestore
  passage_doc = ref_collection_passages.document(datapoint["passage_id"]).get()
  passage_rec = passage_doc.to_dict()
  # Retrieve text record from Firestore
  text_doc = ref_collection_text.document(passage_rec['text_id']).get()
  text_rec = text_doc.to_dict()
  # Preprocess title
  title = text_rec["title"]
  title = preprocess_text(title, nlp,
                          with_remove_known_unuseful_strings=False,
                          with_remove_parentheses_and_brackets=False,
                          with_remove_text_inside_parentheses=False,
                          with_remove_leading_patterns=False,
                          with_remove_timestamps=False,
                          with_replace_unicode_characters=True,
                          with_expand_contractions=False,
                          with_remove_links_from_text=False,
                          with_put_placeholders=False,
                          with_final_cleanup=False)
  # Create row
  row = {
    "id": idx + 1,
    "text": datapoint["text"],
    "label": datapoint["label"],
    "metadata": {
      "passage_id": datapoint["passage_id"],
      'text_id': passage_rec['text_id'],
      'title': title,
      'publication_date': firestore_timestamp_to_string(passage_rec['publication_date']),
      'source': passage_rec['url'],
    }
  }
  # Increase counters
  if row["label"] == "continue":
    counter_continue += 1
  else:
    counter_not_continue += 1

  # Save non-anonymized row to JSONL file
  save_row_to_jsonl_file(row, output_file)
  # Anonymize text
  anonym_text = custom_anonymize_text(row["text"], nlp)
  row["text"] = anonym_text
  # Save anonymized row to JSONL file
  save_row_to_jsonl_file(row, output_file_anonym)

print(f"Saved {counter_continue + counter_not_continue} datapoints, anonymized and non-anonymized, to JSONL files.")
print(f"• Continue: {counter_continue}")
print(f"• Not continue: {counter_not_continue}")
