from pprint import pprint
from collections import Counter
import json
import random

from tqdm import tqdm
import spacy

from db import spreadsheet_6
from lib.utils import (read_from_google_sheet, write_to_google_sheet, save_row_to_jsonl_file, empty_json_file,
                       save_jsonl_file)
from lib.ner_processing import custom_anonymize_text
from lib.continuity_checks import (check_lexical_continuity, check_syntactic_continuity, check_semantic_continuity,
                                   check_transition_markers_continuity, check_coreference)

# Global variables
SEED = 42

dataset = read_from_google_sheet(spreadsheet_6, "dataset_2")
# dataset = dataset[:500]

# Set seed
random.seed(SEED)
# Shuffle dataset
random.shuffle(dataset)

nlp = spacy.load("en_core_web_lg")

# Initialize path and name of output JSON-L file
output_file = "shared_data/dataset_2_6_1b_pair_sentences_gs.jsonl"

# Initialize a JSONL file for the dataset
empty_json_file(output_file)

new_dataset = []

for idx, datapoint in enumerate(tqdm(dataset, desc=f"Generating Gold Standard Dataset")):
  row = {
    "id": idx + 1,
    "text": datapoint["text"],
    "label": datapoint["reclass_4"],
    # "metadata": {
    #   # "gsheets_id": datapoint["id"],
    #  "rb_continuity_features": datapoint["continuity"],
    #  "rb_label": datapoint["reclass_rb2"],
    #  "annotator1": datapoint["annotator"],
    #  "annotator2": datapoint["annotator_3"]
    #}
  }
  if row["label"]:
    new_dataset.append(row)

dataset = []

# Anonymize dataset
"""for idx, datapoint in enumerate(tqdm(new_dataset, desc=f"Anonymizing dataset")):
  datapoint["text"] = custom_anonymize_text(datapoint["text"], nlp)"""

counter_classes = Counter([datapoint["label"] for datapoint in dataset])

save_jsonl_file(new_dataset, output_file)

print(counter_classes)
print(f"Saved {len(new_dataset)} datapoints to {output_file}")
