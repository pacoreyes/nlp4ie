from pprint import pprint
from collections import Counter

from db import spreadsheet_4
from lib.utils import load_jsonl_file, write_to_google_sheet, read_from_google_sheet, load_json_file
from lib.utils2 import remove_duplicated_datapoints, remove_examples_in_dataset
from lib.text_utils import expand_contractions


dataset = read_from_google_sheet(spreadsheet_4, "dataset_3_final")
# dataset_test = read_from_google_sheet(spreadsheet_4, "dataset_3_test")

# dataset = dataset3 + dataset_test

all_frames = set()

for datapoint in dataset:

  semantic_frames = eval(datapoint["metadata"])["semantic_frames"]
  print(semantic_frames)

  for frame in semantic_frames:
    all_frames.add(frame)

pprint(all_frames)
print(len(all_frames))
