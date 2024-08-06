from pprint import pprint
from collections import Counter
import json
import ast

from tqdm import tqdm
import spacy

from db import spreadsheet_7
from lib.utils import read_from_google_sheet, write_to_google_sheet

# Define count variables
lexical_continuity_counter = 0
syntactic_continuity_counter = 0
semantic_continuity_counter = 0
transition_markers_continuity_counter = 0
coreference_counter = 0


# Initialize path and name of output JSON-L file
gsheet = "test_dataset"

dataset = read_from_google_sheet(spreadsheet_7, gsheet)

# filter the not_continue class
dataset = [data for data in dataset if data["rb_label"] == "continue"]

for datapoint in tqdm(dataset, desc=f"Analyzing {len(dataset)} datapoints", total=len(dataset)):
  if datapoint["metadata"]:
    metadata = ast.literal_eval(datapoint["metadata"])
    metadata = metadata[0]

    if metadata.get("lexical_continuity"):
      lexical_continuity_counter += 1
    elif metadata.get("syntactic_continuity"):
      syntactic_continuity_counter += 1
    elif metadata.get("semantic_continuity"):
      semantic_continuity_counter += 1
    elif metadata.get("transition_markers_continuity"):
      if metadata.get("transition_markers_continuity")[0].get("continue"):
        transition_markers_continuity_counter += 1
    elif metadata.get("coreference"):
      coreference_counter += 1

print(f"\nLexical continuity: {lexical_continuity_counter}")
print(f"Syntactic continuity: {syntactic_continuity_counter}")
print(f"Semantic continuity: {semantic_continuity_counter}")
print(f"Transition markers continuity: {transition_markers_continuity_counter}")
print(f"Coreference: {coreference_counter}")
