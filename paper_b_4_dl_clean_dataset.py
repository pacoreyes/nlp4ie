# from pprint import pprint

from tqdm import tqdm

from db import spreadsheet_6
from lib.utils import read_from_google_sheet, write_to_google_sheet

dataset = read_from_google_sheet(spreadsheet_6, "dataset_2")
print(f"Source dataset: {len(dataset)} datapoints")

unique_texts = set()
cleaned_dataset = []

print("Preserving uniqueness of texts in the dataset...")
for item in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints", total=len(dataset)):
  if item['text'] not in unique_texts:
    unique_texts.add(item['text'])
    cleaned_dataset.append(item)

print(f"Cleaned dataset: {len(cleaned_dataset)} datapoints (removed {len(dataset) - len(cleaned_dataset)} duplicated)")
# pprint(dataset)

rows = []
for item in cleaned_dataset:
  row = [
    item["id"],
    item["passage_id"],
    item["text"],
    item["label"],
    # item["annotator"],
    # item["text_id"],
  ]
  rows.append(row)

# pprint(rows)

write_to_google_sheet(spreadsheet_6, "dataset_2_cleaned", rows)
