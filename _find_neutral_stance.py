import spacy
from tqdm import tqdm

from db import firestore_db, spreadsheet_4
from lib.utils import load_txt_file, read_from_google_sheet

dataset = read_from_google_sheet(spreadsheet_4, "dataset_3_")

nlp = spacy.load("en_core_web_trf")

for datapoint in tqdm(dataset, desc="Processing text"):
  sentence = datapoint["text"]
  doc = nlp(sentence)

  # Check if sentence contains an adjective
  if not any(token.pos_ == "ADJ" or token.pos_ == "ADV" for token in doc):
    print(datapoint["id"])
