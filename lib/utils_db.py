from tqdm import tqdm
from google.api_core.retry import Retry

from db import firestore_db

""" ##############################
Firestore Database Functions
############################## """


def retrieve_many_from_firestore(collection, retry=False, limit=0):
  """
  Retrieves many documents from a Firestore collection.

  Args:
      collection (str): The name of the Firestore collection.
      retry (bool, optional): Whether to retry the operation. Defaults to False.
      limit (int, optional): The maximum number of documents to retrieve. Defaults to 0.
  """
  coll_ref = firestore_db.collection(collection)
  if limit > 0:
    query = coll_ref.limit(limit)
  else:
    query = coll_ref
  if retry:
    docs = query.stream(retry=Retry())
  else:
    docs = coll_ref.stream()
  return [doc.to_dict() for doc in docs]
