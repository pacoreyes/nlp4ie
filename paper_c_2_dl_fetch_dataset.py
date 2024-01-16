# from pprint import pprint

from db import firestore_db, spreadsheet_4
from lib.utils import write_to_google_sheet

dataset3_col_ref = firestore_db.collection("sentences")

# retrieve all documents from the collection
docs = dataset3_col_ref.stream()
recs = [doc.to_dict() for doc in docs]

data = []
for rec in recs:
  row = [
    rec["id"],
    rec["text"],
    rec["issue"],
    # rec["main_frame"],
    # ', '.join(rec["semantic_frames"]),
  ]
  data.append(row)

# Write data to Google Sheet
write_to_google_sheet(spreadsheet_4, "dataset_3_", data)
