from google.api_core.retry import Retry
from db import firestore_db
from lib.utils import save_row_to_jsonl_file

ref_coll_sentences = firestore_db.collection("sentences5")
query = ref_coll_sentences.order_by("id").limit(10000)

docs = query.stream(retry=Retry())
recs = [d.to_dict() for d in docs]


for rec in recs:
  row = {
    "id": rec["id"],
    "text": rec["text"],
    "issue": rec["issue"]
  }
  save_row_to_jsonl_file(row, "shared_data/dataset_3_9_unseen_unlabeled_sentences_3.jsonl")
