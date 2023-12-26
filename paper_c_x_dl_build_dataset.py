import spacy
from tqdm import tqdm

from db import spreadsheet_4
from lib.ner_processing import custom_anonymize_text
from lib.utils import read_from_google_sheet, write_to_google_sheet

# Load spacy model
nlp_trf = spacy.load("en_core_web_trf")

data = read_from_google_sheet(spreadsheet_4, "analize_frames")
output_spreadsheet = "dataset_3"

LABEL_MAP = {
  "0": "support",
  "1": "oppose"
}

dataset = []

for datapoint in tqdm(data, desc=f"Processing {len(data)} datapoints"):
  if datapoint["class"] == "0" or datapoint["class"] == "1":

    """# Anonymize text
    more_entities = ["COVID-19", "COVID", "Army", "WeCanDoThis.HHS.gov", "HIV", "AIDS"]
    datapoint["text"] = custom_anonymize_text(datapoint["text"], nlp_trf,
                                              ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
                                               "LAW", "DATE", "TIME", "MONEY", "QUANTITY"])

    for entity in more_entities:
      datapoint["text"] = datapoint["text"].replace(entity, "[ENTITY]")"""

    dataset.append(datapoint)

dataset_3 = []
for datapoint in tqdm(dataset, desc=f"Uploading {len(dataset)} datapoints", total=len(dataset)):
  _row = [
    datapoint["id"],
    datapoint["text"],
    LABEL_MAP[datapoint["class"]]
  ]
  dataset_3.append(_row)

write_to_google_sheet(spreadsheet_4, output_spreadsheet, dataset_3)
print("Saved Dataset 3 to spreadsheet")
