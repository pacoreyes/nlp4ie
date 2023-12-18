# from pprint import pprint

from tqdm import tqdm

from db import spreadsheet_6
from lib.utils import read_from_google_sheet, write_to_google_sheet
from lib.continuity_checks import (check_lexical_continuity, check_syntactic_continuity, check_semantic_continuity,
                                   check_transition_markers_continuity, check_coreference)


dataset = read_from_google_sheet(spreadsheet_6, "dataset_2")

# dataset = dataset[:100]

print(f"\nReclassifying datapoint {len(dataset)}...")

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
