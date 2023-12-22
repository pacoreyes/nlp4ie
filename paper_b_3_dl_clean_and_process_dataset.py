# from pprint import pprint

import spacy
# from spacy.matcher import PhraseMatcher
from tqdm import tqdm

from db import firestore_db, spreadsheet_6
from lib.utils import write_to_google_sheet, load_jsonl_file

source_dataset = load_jsonl_file("shared_data/dataset_2_3_pair_sentences.jsonl")

text_coll_ref = firestore_db.collection("texts2")

nlp = spacy.load("en_core_web_sm")

# Sort the dataset by "label"
source_dataset = sorted(source_dataset, key=lambda k: k['label'])

# source_dataset = source_dataset[:1]

ignore_patterns = ["JOHNSON:",
                   "STEPHANOPOULOS:",
                   "MS. JEAN-PIERRE:",
                   "Specifically, the EMBRACE Act would:",
                   "permalink",
                   "Governor Ron DeSantis:",
                   "Ron DeSantis:",
                   "President Biden:",
                   "Boris Johnson:",
                   "Pedro Pierluisi:",
                   "Nihad Awad:",
                   "RADDATZ:",
                   "President Donald Trump:",
                   "JOHNSON:",
                   "CORRESPONDENT:"
                   "MS. JEAN-PIERRE:",
                   "Chicago Daily Herald:",
                   "Rohit Chopra:",
                   "Xavier Becerra:",
                   "Judith Persichilli:",
                   "Governor Andrew Cuomo:"
                   "Q:",
                   "Q.",
                   "Governor Cuomo:",
                   "Speaker 2:",
                   "ESPER:",
                   "THE PRESIDENT:",
                   ]

ignore_in_title_patterns = ["press release",
                            "statement",
                            "readout",
                            "announcement",
                            "declaration",
                            "letter",
                            "proclamation",
                            "Warren",
                            "Executive Order",
                            "Fact Sheet",
                            "Klobuchar",
                            "After Arrest Of Chinese",
                            "Blumenthal",
                            "This Week' Transcript",
                            "Brown, Colleagues Urge Biden",
                            "Schatz Introduces New Legislation",
                            "Pelosi Transcript of Press",
                            "Ahead of Medicare and Medicaid's",
                            "SCHUMER",
                            "Health IG Launches Inquiry",
                            "Baldwin"
                            "Schiff",
                            "Warner",
                            "Booker"]

ignore_in_text_patterns = ["WASHINGTON -",
                           "WASHINGTON –",
                           "WASHINGTON —",
                           "WASHINGTON-",
                           "WASHINGTON–",
                           "WASHINGTON—",
                           "Washington, D.C. -",
                           "Washington, D.C. –",
                           "Washington, D.C. —",
                           "WASHINGTON, D.C. -",
                           "WASHINGTON, D.C. –",
                           "WASHINGTON, D.C. —",
                           "Washington, DC -",
                           "Washington, DC –",
                           "Washington, DC —",
                           "Washington D.C. -",
                           "Washington D.C. –",
                           "Washington D.C. —",
                           "Washington D.C.-",
                           "Washington D.C.–",
                           "Washington D.C.—",
                           "Washington, DC-",
                           "Washington, DC–",
                           "Washington, DC—"
                           ]

undesirable_chars = [
  "{", "}", "[", "]", "(", ")"
]

""" #############################################
Step 1: Ensure datapoints uniqueness by removing duplicates
############################################# """

unique_texts = set()
c_dataset = []

print("Removing duplicates...")
for item in tqdm(source_dataset, desc=f"Processing {len(source_dataset)} datapoints", total=len(source_dataset)):
  if item['text'] not in unique_texts:
    unique_texts.add(item['text'])
    c_dataset.append(item)

print(
  f"Dataset 2 after duplicate removal: {len(source_dataset) - len(c_dataset)}  = {len(c_dataset)}\n")

""" #############################################
Step 2: Skip datapoints based on patterns 
############################################# """

rows = []
for idx, datapoint in enumerate(tqdm(c_dataset, desc=f"Processing {len(c_dataset)} datapoints")):
  skip_datapoint = False

  """ A: Skip sentences with speaker labels ("Name: ") """
  # Check if sentences contain any of the ignore patterns - usually speaker labels
  for item in ignore_patterns:
    if item in datapoint["text"]:
      # print(f"Skipping: {datapoint['text']}")
      skip_datapoint = True
      break
  # Skip
  if skip_datapoint:
    continue

  """ B: Skip malformed sentences: undesirable chars and start patterns """
  # Split sentences
  sentences = datapoint["text"].split("[SEP] ")

  for sentence in sentences:

    """ B.1 Check if sentence contains any of the undesirable chars """
    if any(char in sentence for char in undesirable_chars):
      # print(f"Skipping: {sentence}")
      skip_datapoint = True
      break

    """ B.2 check if starts with uppercase or number """
    first_char = sentence[0]
    doc = nlp(first_char)
    if doc[0].is_lower or doc[0].is_digit or doc[0].is_punct:
      # print(f"Skipping: {sentence}")
      skip_datapoint = True
      break
  # Skip
  if skip_datapoint:
    continue

  """ C: Skip sentences if source text is not valid """
  doc = text_coll_ref.document(datapoint["text_id"]).get()
  rec = doc.to_dict()

  title = rec["title"]
  text = rec["text"]

  # pprint(any(item in text for item in ignore_in_text_patterns))

  """ C.1 Check if title of source discourse contains any of the ignore patterns """
  for item in ignore_in_title_patterns:
    if item.lower() in title.lower():
      # print(f"Skipping: {title}")
      skip_datapoint = True
      break
  # Skip
  if skip_datapoint:
    continue

  """ C.2 Check if text of source discourse contains any of the ignore patterns """
  if any(item in text for item in ignore_in_text_patterns):
    # print(f"Skipping: {title}")
    skip_datapoint = True
    continue
  # Skip
  if skip_datapoint:
    continue

  row = [
    datapoint["id"],
    datapoint["passage_id"],
    datapoint["text"],
    datapoint["label"],
    datapoint["annotator"],
    # datapoint["text_id"]
  ]
  rows.append(row)

write_to_google_sheet(spreadsheet_6, "dataset_2", rows)

print(f"Uploaded {len(rows)} datapoints to Google Sheets")

#  8139 datapoints
# - 294 duplicated
#   ---
#  7845 datapoints
# -2563 malformed
#   ---
#  5282 datapoints
