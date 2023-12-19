from transformers import BertTokenizer
from db import firestore_db

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

_id = "presidencyucsbedudocumentsremarksmajorappointmentsthedepartmenttransportation"

doc = firestore_db.collection('texts2').document(_id).get()
rec = doc.to_dict()

text = " ".join(rec["dataset1_text"])

print(len(tokenizer.tokenize(text)))
