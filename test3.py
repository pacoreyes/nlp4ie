from lib.utils_db import retrieve_many_from_firestore
from lib.utils import save_jsonl_file, load_jsonl_file, empty_json_file
from lib.utils2 import remove_duplicated_datapoints, remove_examples_in_dataset

# Load dataset3 from Google Sheets
print(" Retrieving sentences from Firestore...")
unlabeled_sentences2 = retrieve_many_from_firestore(
  "sentences2", True, 10000)

# load unlabeled sentences dataset
unlabeled_sentences = load_jsonl_file("shared_data/dataset_3_9_unseen_unlabeled_sentences.jsonl")

# Remove duplicated datapoints
remove_duplicated_datapoints(unlabeled_sentences2)

# Remove examples in dataset3
unlabeled_sentences = remove_examples_in_dataset(unlabeled_sentences2, unlabeled_sentences)

sentences = []
for i in unlabeled_sentences:
  row = {
    "id": i['id'],
    "text": i['text'],
    "issue": i['issue']
  }
  sentences.append(row)

# Save dataset
print(f"Saving dataset with {len(unlabeled_sentences)} datapoints...")
save_jsonl_file(sentences, "shared_data/dataset_3_9_unseen_unlabeled_sentences_2.jsonl")
