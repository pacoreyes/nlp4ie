# import functools

# import requests
# import spacy

from db import firestore_db, spreadsheet_4
# from issues_matcher import match_issues
# from linguistic_utils import check_minimal_meaning
# from text_preprocessing import preprocess_text
from lib.utils import read_from_google_sheet, write_to_google_sheet, load_txt_file

from pprint import pprint

"""
This script extracts sentences with support/oppose stances from texts to create a corpus of sentences.
The script uses a fine-tuned BERT model to predict the the support/oppose stance of a sentence.
"""

# Load spacy models
# nlp = spacy.load("en_core_web_sm")
# Load transformer models
# nlp_trf = spacy.load("en_core_web_trf")

dataset3_col_ref = firestore_db.collection("_dataset3")

# retrieve all documents from the collection
docs = dataset3_col_ref.stream()
recs = [doc.to_dict() for doc in docs]

data = []
for rec in recs:
  row = [
    rec["id"],
    rec["text"],
    rec["issue"],
    rec["main_frame"],
    ', '.join(rec["semantic_frames"]),
  ]
  data.append(row)

# Write data to Google Sheet
write_to_google_sheet(spreadsheet_4, "dataset_3_raw", data)
