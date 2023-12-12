from db import firestore_db, spreadsheet_5
from tqdm import tqdm
from utils import save_row_to_jsonl_file, load_jsonl_file, empty_json_file, firestore_timestamp_to_string

from transformers import BertTokenizer


# Initialize Firestore DB
source_texts_ref = firestore_db.collection('texts2')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Get Dataset1 from Google Sheets
source_gsheet_dataset1 = spreadsheet_5.worksheet("dataset_1")
dataset1_from_gsheets = source_gsheet_dataset1.get_all_values()
dataset1_from_gsheets = dataset1_from_gsheets[1:]  # Remove the header

empty_json_file('shared_data/dataset_1.jsonl')

all_ids = [row[0] for row in dataset1_from_gsheets]

# Initialize counters
exceeded_token_limit = 0
number_of_datapoints = 0

monologic_class_counter = 0
dialogic_class_counter = 0

index = 0
for _id in tqdm(all_ids, desc=f"Processing {len(all_ids)} datapoints"):
  doc = source_texts_ref.document(_id).get()
  rec = doc.to_dict()

  text = " ".join(rec["dataset1_text"])

  if len(tokenizer.tokenize(text)) < 510:
    exceeded_token_limit += 1
    continue
  else:
    politician = dataset1_from_gsheets[index][2].split(", ")
    gender = dataset1_from_gsheets[index][3].split(", ")

    if rec.get("publication_date"):
      publication_date = firestore_timestamp_to_string(rec["publication_date"])
    else:
      publication_date = ""

    datapoint = {
      "text": rec["dataset1_text"],
      "discourse_type": rec["dataset1_class"],
      "metadata": {
        "text_id": rec["id"],
        "title": rec["title"],
        "source": rec["url"],
        "publication_date": publication_date,
        "crawling_date": firestore_timestamp_to_string(rec["crawling_date"]),
        "politician": politician,
        "gender": gender
      }
    }
    number_of_datapoints += 1
    if datapoint["discourse_type"] == 0:
      monologic_class_counter += 1
    else:
      dialogic_class_counter += 1
    save_row_to_jsonl_file(datapoint, "shared_data/dataset_1.jsonl")

  index += 1

print("Sorting the dataset by discourse_type")
dataset1 = load_jsonl_file("shared_data/dataset_1.jsonl")
# Empty output JSONL file
empty_json_file('shared_data/dataset_1.jsonl')
# Sort the dataset by "discourse_type"
dataset1 = sorted(dataset1, key=lambda k: k['discourse_type'])

index = 0
for d in tqdm(dataset1, desc=f"Saving {len(dataset1)} datapoints ordered by discourse_type"):
  datapoint = {
    "id": index,
    # "text_id": d["metadata"]["text_id"],  # TODO: remove this
    "text": d["text"],
    "discourse_type": d["discourse_type"],
    "metadata": {
      "text_id": d["metadata"]["text_id"],
      "title": d["metadata"]["title"],
      "source": d["metadata"]["source"],
      "publication_date": d["metadata"]["publication_date"],
      "crawling_date": d["metadata"]["crawling_date"],
      "politician": d["metadata"]["politician"],
      "gender": d["metadata"]["gender"]
    }
  }
  save_row_to_jsonl_file(datapoint, "shared_data/dataset_1.jsonl")
  index += 1

print("\nSkipped datapoints due to token limit:", exceeded_token_limit)
print("Total number of datapoints:", number_of_datapoints)
print(f"• Monologic class counter: {monologic_class_counter}")
print(f"• Dialogic class counter: {dialogic_class_counter}")
