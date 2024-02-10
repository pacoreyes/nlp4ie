"""
This script creates the raw dataset 1A prepared for the rule-based model.
"""
import spacy
from tqdm import tqdm
from transformers import BertTokenizer

from db import firestore_db, spreadsheet_5
from lib.text_utils import preprocess_text
from lib.utils import (save_row_to_jsonl_file, load_jsonl_file, empty_json_file, firestore_timestamp_to_string,
                       read_from_google_sheet)
from lib.ner_processing import replace_speaker_labels

# Load spaCy's Transformer model
nlp = spacy.load("en_core_web_trf")

# Initialize Firestore DB
source_texts_ref = firestore_db.collection('texts2')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Get Dataset1 from Google Sheets
dataset = read_from_google_sheet(spreadsheet_5, "dataset_1")
# dataset = dataset[828:832]

output_file = "shared_data/dataset_1_1_raw.jsonl"

empty_json_file(output_file)

# Initialize counters
exceeded_token_limit = 0
number_of_datapoints = 0

monologic_class_counter = 0
dialogic_class_counter = 0

index = 0
for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  doc = source_texts_ref.document(datapoint["id"]).get()
  rec = doc.to_dict()

  text = " ".join(rec["dataset1_text"])

  if len(tokenizer.tokenize(text)) < 450:
    exceeded_token_limit += 1
    continue
  else:
    politician = datapoint["politician"].split(", ")
    gender = datapoint["gender"].split(", ")

    if rec.get("publication_date"):
      publication_date = firestore_timestamp_to_string(rec["publication_date"])
    else:
      publication_date = ""

    # Replace speaker labels
    text = replace_speaker_labels(text)

    # Preprocess the text
    text = preprocess_text(text, nlp,
                           with_remove_known_unuseful_strings=True,
                           with_remove_parentheses_and_brackets=True,
                           with_remove_text_inside_parentheses=True,
                           with_remove_leading_patterns=False,
                           with_remove_timestamps=False,
                           with_replace_unicode_characters=True,
                           with_expand_contractions=True,
                           with_remove_links_from_text=True,
                           with_put_placeholders=False,
                           with_final_cleanup=True)

    rec["title"] = preprocess_text(rec["title"], nlp,
                                   with_remove_known_unuseful_strings=False,
                                   with_remove_parentheses_and_brackets=False,
                                   with_remove_text_inside_parentheses=False,
                                   with_remove_leading_patterns=False,
                                   with_remove_timestamps=False,
                                   with_replace_unicode_characters=True,
                                   with_expand_contractions=False,
                                   with_remove_links_from_text=True,
                                   with_put_placeholders=False,
                                   with_final_cleanup=True)

    row = {
      "text": text,
      "metadata": {
        "text_id": rec["id"],
        "title": rec["title"],
        "source": rec["url"],
        "publication_date": publication_date,
        "crawling_date": firestore_timestamp_to_string(rec["crawling_date"]),
        "politician": politician,
        "gender": gender
      }
    }
    number_of_datapoints += 1
    if rec["dataset1_class"] == 0:
      row["label"] = "monologic"
      monologic_class_counter += 1
    else:
      row["label"] = "dialogic"
      dialogic_class_counter += 1
    save_row_to_jsonl_file(row, output_file)

  index += 1

print("Sorting the dataset by label")
dataset = load_jsonl_file(output_file)
# Empty output JSONL file
empty_json_file(output_file)
# Sort the dataset by "label"
dataset = sorted(dataset, key=lambda k: k['label'])

index = 0
for datapoint in tqdm(dataset, desc=f"Saving {len(dataset)} datapoints"):
  row = {
    "id": index,
    "text": datapoint["text"],
    "label": datapoint["label"],
    "metadata": {
      "text_id": datapoint["metadata"]["text_id"],
      "title": datapoint["metadata"]["title"],
      "source": datapoint["metadata"]["source"],
      "publication_date": datapoint["metadata"]["publication_date"],
      "crawling_date": datapoint["metadata"]["crawling_date"],
      "politician": datapoint["metadata"]["politician"],
      "gender": datapoint["metadata"]["gender"]
    }
  }
  save_row_to_jsonl_file(row, output_file)
  index += 1

print("\nSkipped datapoints due to token limit:", exceeded_token_limit)
print("Total number of datapoints:", number_of_datapoints)
print(f"• Monologic: {monologic_class_counter}")
print(f"• Dialogic: {dialogic_class_counter}")
