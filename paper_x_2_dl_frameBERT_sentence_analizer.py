from db import spreadsheet_4, firestore_db
from lib.utils import read_from_google_sheet, save_json_file
from lib.issues_matcher import match_issues

from pprint import pprint
import requests

# Get dataset from Google Sheet
dataset = read_from_google_sheet(spreadsheet_4, "dataset_3_raw")

# Create reference to the database
dataset3_coll = firestore_db.collection("_dataset3")

# define IP address and port
IP = "141.43.202.175"
# IP = "localhost"
PORT = "5000"


def get_frames(sent):
  print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  pprint(match_issues(sent))
  response = requests.get(f"http://{IP}:{PORT}/api/extract_frames/" + sent)
  response = response.json()
  results = response["frames"]["textae"]
  print(f"> Sentence: {sent}")
  # (results)
  save_json_file(results, "shared_data/frame_net_analysis.json", indent=2)
  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


# Set sentence id
sentence_id = "0000000010"
# Get datapoint
datapoint = dataset3_coll.document(sentence_id).get().to_dict()
# datapoint = find_dict_by_key_value(dataset, "id", sentence_id)
# Get sentence
sentence = datapoint["text"]

# Get frames and save them to a JSON file
get_frames(sentence)

"""
for sent in sentences:
  get_frames(sent)"""

"""
PROMPT:

This is the FrameNet analysis data. Step 1: Begin by identifying the specific issue in the sentence that is either supported or opposed (start with that!!!). Step 2: Next, determine which specific FrameNet frame is the most crucial for classifying the sentence as expressing a supportive or opposing stance. This will assist in creating a rule for generalization: 

"""
