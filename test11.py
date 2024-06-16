from pprint import pprint
import json

import spacy
from afinn import Afinn
from tqdm import tqdm

from lib.utils import load_jsonl_file

if spacy.prefer_gpu():
  print("spaCy is using GPU!")
else:
  print("GPU not available, spaCy is using CPU instead.")

nlp = spacy.load("en_core_web_trf")

# dataset = read_from_google_sheet(spreadsheet_4, "dataset_3_final")
dataset_test = load_jsonl_file("shared_data/dataset_2_test.jsonl")
dataset_train = load_jsonl_file("shared_data/dataset_2_train.jsonl")
dataset_validation = load_jsonl_file("shared_data/dataset_2_validation.jsonl")

dataset = dataset_test + dataset_train + dataset_validation

terms = []

for datapoint in tqdm(dataset, desc="Processing dataset"):
  sentence = datapoint["text"]

  doc = nlp(sentence)

  for token in doc:

    # affect_score = Afinn().score(token.text)

    # get 3 final characters of the token
    final_chars = token.text[-3:]

    if token.pos_ == "ADJ" and final_chars == "ing":  # and affect_score > 0
      print(token.text)
      terms.append(token.lemma_.lower())

terms = list(set(terms))

terms.sort()

print(json.dumps(terms))
