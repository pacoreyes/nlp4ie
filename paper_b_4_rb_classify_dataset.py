# from pprint import pprint
from collections import Counter

from tqdm import tqdm
import spacy

from db import spreadsheet_6
from lib.utils import read_from_google_sheet, write_to_google_sheet, save_row_to_jsonl_file, empty_json_file
from lib.ner_processing import custom_anonymize_text
from lib.continuity_checks import (check_lexical_continuity, check_syntactic_continuity, check_semantic_continuity,
                                   check_transition_markers_continuity, check_coreference)


dataset = read_from_google_sheet(spreadsheet_6, "dataset_2")
# dataset = dataset[:10]

nlp_trf = spacy.load("en_core_web_trf")

# Initialize path and name of output JSON-L file
output_file = "shared_data/dataset_2_5_pair_sentences_reclass.jsonl"

# Initialize a JSONL file for the dataset
empty_json_file(output_file)


for datapoint in tqdm(dataset, desc=f"Reclassifying {len(dataset)} datapoints", total=len(dataset)):

  # Split the text into two sentences
  sentences = datapoint["text"].split("[SEP] ")
  sent1 = sentences[0]
  sent2 = sentences[1]

  continuity = []

  lexical_continuity = check_lexical_continuity(sent1, sent2)
  if lexical_continuity['lexical_continuity']:
    continuity.append(lexical_continuity)

  syntactic_continuity = check_syntactic_continuity(sent1, sent2)
  if syntactic_continuity['syntactic_continuity']:
    continuity.append(syntactic_continuity)

  semantic_continuity = check_semantic_continuity(sent1, sent2)
  if semantic_continuity['semantic_continuity']:
    continuity.append(semantic_continuity)

  transition_markers_continuity = check_transition_markers_continuity(sent2)
  if transition_markers_continuity['transition_markers_continuity']:
    continuity.append(transition_markers_continuity)

  coreference = check_coreference(sent1, sent2)
  if coreference['coreference']:
    continuity.append(coreference)

  if continuity:
    datapoint['reclass'] = "continue"
    datapoint['continuity'] = continuity
  else:
    datapoint['reclass'] = "not_continue"
    datapoint['continuity'] = ""

print(f"Reclassified dataset: {len(dataset)} datapoints\n")

for datapoint in tqdm(dataset, desc=f"Anonymizing {len(dataset)} datapoints", total=len(dataset)):
  pair = "[CLS] " + datapoint["text"] + " [SEP]"

  # Anonymize text
  more_entities = ["COVID-19", "COVID", "Army", "WeCanDoThis.HHS.gov", "HIV", "AIDS"]
  pair_anonym = custom_anonymize_text(pair, nlp_trf,
                                      ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
                                       "LAW", "DATE", "TIME", "MONEY", "QUANTITY"])

  for entity in more_entities:
    datapoint["text"] = pair_anonym.replace(entity, "[ENTITY]")

  # clone datapoint
  new_datapoint = datapoint.copy()
  # remove attributes annotator, label and continuity
  new_datapoint.pop("annotator")
  new_datapoint.pop("label")
  new_datapoint.pop("continuity")
  # rename attribute reclass to label
  new_datapoint["label"] = new_datapoint.pop("reclass")

  save_row_to_jsonl_file(new_datapoint, output_file)
print(f"Anonymized dataset: {len(dataset)} datapoints\n")


new_dataset = []
for _datapoint in tqdm(dataset, desc=f"Uploading {len(dataset)} datapoints", total=len(dataset)):
  _row = [
    _datapoint["id"],
    _datapoint["passage_id"],
    _datapoint["text"],
    _datapoint["label"],
    _datapoint["annotator"],
    _datapoint["reclass"],
    str(_datapoint["continuity"])
  ]
  new_dataset.append(_row)

write_to_google_sheet(spreadsheet_6, "dataset_2_reclass", new_dataset)

print(f"Uploaded dataset: {len(new_dataset)} datapoints")

counter = Counter([datapoint["label"] for datapoint in dataset])
continue_percentage = counter["continue"] / (counter["continue"] + counter["not_continue"]) * 100
not_continue_percentage = counter["not_continue"] / (counter["continue"] + counter["not_continue"]) * 100

print()
print("Class distribution:")
print(f"\n• Continue: {counter['continue']} ({continue_percentage:.2f})")
print(f"• Not continue: {counter['not_continue']} ({not_continue_percentage:.2f})")
