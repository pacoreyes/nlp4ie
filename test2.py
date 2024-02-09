from pprint import pprint

# from google.api_core.retry import Retry

# from db import firestore_db, spreadsheet_4
from lib.utils import load_json_file, save_json_file, load_jsonl_file, read_from_google_sheet

sentences = load_json_file("shared_data/paper_c_3_11_sentences_with_frame_net.json")

frames = set()
for rec in sentences:
  frames_data = rec["frameNet_data"]
  for frame in frames_data:
    frame_name = frame["frame"]
    # frame_objects = next((f["object"] for f in rule_frames if f["name"] == frame_name), None)
    # pprint(frame_name)
    frames.add(frame_name)

pprint(frames)
