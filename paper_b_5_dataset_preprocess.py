# from pprint import pprint

import spacy
from tqdm import tqdm

from db import firestore_db, spreadsheet_4
from lib.text_utils import preprocess_text
from lib.semantic_frames import api_get_frames
from lib.utils import read_from_google_sheet, write_to_google_sheet


if spacy.prefer_gpu():
  print("spaCy is using GPU!")
else:
  print("GPU not available, spaCy is using CPU instead.")

# load spaCy's Transformer model
nlp = spacy.load("en_core_web_trf")

ref_sentences1 = firestore_db.collection("sentences")
ref_sentences2 = firestore_db.collection("sentences2")
ref_sentences3 = firestore_db.collection("sentences3")
ref_sentences4 = firestore_db.collection("sentences4")

ref_texts = firestore_db.collection("texts2")

# Load dataset from Google Sheets
dataset = read_from_google_sheet(spreadsheet_4, "master")

# dataset = dataset[:1]

output_spreadsheet = "dataset_3_test"


""" #######################################################################
Preprocess text
########################################################################"""

rows = []

# Process all datapoints in Dataset 1
for idx, datapoint in tqdm(enumerate(dataset), desc=f"Processing {len(dataset)} datapoints", total=len(dataset)):
  text = datapoint["text"]

  sent = preprocess_text(text, nlp,
                         with_remove_known_unuseful_strings=True,
                         with_remove_parentheses_and_brackets=False,
                         with_remove_text_inside_parentheses=False,
                         with_remove_leading_patterns=False,
                         with_remove_timestamps=False,
                         with_replace_unicode_characters=True,
                         with_expand_contractions=True,
                         with_remove_links_from_text=False,
                         with_put_placeholders=False,
                         with_final_cleanup=True)

  with_metadata = False

  for database in [ref_sentences1, ref_sentences2, ref_sentences3, ref_sentences4]:
    doc_ref = database.document(datapoint["id"])

    if not doc_ref.get().exists:
      continue
    else:
      rec = doc_ref.get().to_dict()

      #
      ref_doc = ref_texts.document(rec["source"])
      if ref_doc.get().exists:
        text_title = ref_doc.get().to_dict()["title"]
      else:
        text_title = ""

      metadata = {
        "title": text_title,
        "source": rec["source_url"],
      }
      # Query the semantic frames API
      present_frames_response = api_get_frames(datapoint["text"], "localhost", "5001", "all")

      if present_frames_response["frames"]["textae"]:
        semantic_frames = present_frames_response["frames"]["textae"][0]["denotations"]
      else:
        semantic_frames = []

      denotations = [frame for frame in semantic_frames]
      sf = [denotation["obj"] for denotation in denotations]

      metadata["semantic_frames"] = list(set(sf))

      with_metadata = True

  if with_metadata:
    row = [
      # idx + 1,
      datapoint["id"],
      sent,
      datapoint["label"],
      str(metadata)
    ]
  else:
    row = [
      # idx + 1,
      datapoint["id"],
      sent,
      datapoint["label"],
      str({
        "semantic_frames": metadata["semantic_frames"]
      })
    ]
  rows.append(row)

write_to_google_sheet(spreadsheet_4, output_spreadsheet, rows)
print()
print("Process finished.")
