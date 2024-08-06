import spacy
from google.api_core.retry import Retry

from db import firestore_db, spreadsheet_7
from lib.utils import save_jsonl_file, read_from_google_sheet

# nlp = spacy.load("en_core_web_lg")

passages_collection = firestore_db.collection("passages")

recs = passages_collection.stream(retry=Retry(deadline=60))
# convert to list of dictionaries
recs = [rec.to_dict() for rec in recs]

annotated = 0

for rec in recs:
  print(rec["id"])
  if rec.get("annotator"):
    annotated += 1

print(f"Annotated: {annotated}")




"""texts_collection = firestore_db.collection("texts2")

train_set = read_from_google_sheet(spreadsheet_7, "_train_dataset")
val_set = read_from_google_sheet(spreadsheet_7, "_valid_dataset")
test_set = read_from_google_sheet(spreadsheet_7, "_test_dataset")

# Iterate the dataset and find the number of tokens of the longest and shortest sentences

counter = 0

train_dataset = []

for entry in train_set:
  passage_id = entry["passage_id"]
  print(entry["passage_id"])
  passage = passages_collection.document(passage_id).get().to_dict()

  _id = counter + 1
  _label = entry["label"]
  _text_id = passage["text_id"]
  _text = texts_collection.document(_text_id).get().to_dict()

  # metadata
  _url = passage["url"]
  _title = _text["title"]

  row = {
    "id": _id,
    "label": _label,
    "text": entry["text"],
    "metadata": {
      "title": _title,
      "url": _url,
    }
  }
  train_dataset.append(row)

save_jsonl_file(train_dataset, "shared_data/topic_boundary_train_.jsonl")

valid_dataset = []

for entry in val_set:
  print(entry["passage_id"])
  passage_id = entry["passage_id"]
  passage = passages_collection.document(passage_id).get().to_dict()

  _id = counter + 1
  _label = entry["label"]
  _text_id = passage["text_id"]
  _text = texts_collection.document(_text_id).get().to_dict()

  # metadata
  _url = passage["url"]
  _title = _text["title"]

  row = {
    "id": _id,
    "label": _label,
    "text": entry["text"],
    "metadata": {
      "title": _title,
      "url": _url,
    }
  }
  valid_dataset.append(row)

save_jsonl_file(valid_dataset, "shared_data/topic_boundary_valid_.jsonl")

test_dataset = []

for entry in test_set:
  print(entry["passage_id"])
  passage_id = entry["passage_id"]
  passage = passages_collection.document(passage_id).get().to_dict()

  _id = counter + 1
  _label = entry["label"]
  _text_id = passage["text_id"]
  _text = texts_collection.document(_text_id).get().to_dict()

  # metadata
  _url = passage["url"]
  _title = _text["title"]

  row = {
    "id": _id,
    "label": _label,
    "text": entry["text"],
    "metadata": {
      "title": _title,
      "url": _url,
    }
  }
  test_dataset.append(row)

save_jsonl_file(test_dataset, "shared_data/topic_boundary_test_.jsonl")

print("Done!")"""
