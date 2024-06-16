from db import spreadsheet_4, firestore_db
from lib.utils import read_from_google_sheet


# Fill ID number with zeros
def id_with_zeros(number):
  return str(number).zfill(10)


dataset = read_from_google_sheet(spreadsheet_4, "dataset_3_final")

dataset3_db_ref = firestore_db.collection("dataset3")

for idx, datapoint in enumerate(dataset):
  doc_id = datapoint["id"]
  datapoint["metadata"] = eval(datapoint["metadata"])
  datapoint["metadata"]["id"] = doc_id

  num_id = id_with_zeros(idx + 1)

  if datapoint["label"] == "support":
    datapoint["id"] = f"{num_id}S"
  else:
    datapoint["id"] = f"{num_id}O"

  doc_ref = dataset3_db_ref.document(datapoint["id"])
  doc_ref.set(datapoint)
  print(f"Added {datapoint['id']} to Firestore's dataset3 collection.")
