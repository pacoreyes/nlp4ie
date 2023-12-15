from db import firestore_db, spreadsheet_6
from lib.utils import write_to_google_sheet, load_jsonl_file
from tqdm import tqdm
from pprint import pprint

source_dataset = load_jsonl_file("shared_data/dataset_2_2_pair_sentences.jsonl")

write_to_google_sheet(spreadsheet_6, "dataset_2", source_dataset)

"""dataset = read_from_google_sheet(spreadsheet_6, "dataset_2")
dataset = dataset[1:]  # Remove the header

pprint(dataset)"""
