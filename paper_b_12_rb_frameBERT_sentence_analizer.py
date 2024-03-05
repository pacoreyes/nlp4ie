from lib.utils import load_json_file, save_json_file

from pprint import pprint
import requests

# Get dataset from Google Sheet
dataset = load_json_file("shared_data/paper_c_3_11_sentences_with_frame_net.json")
sentence = dataset[0]

# define IP address and port
# IP = "141.43.202.175"
IP = "localhost"
PORT = "5001"


def get_frames(sent):
  print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
  response = requests.get(f"http://{IP}:{PORT}/api/extract_frames_graph/" + sent)
  response = response.json()
  print(f"> Sentence: {sent}")
  pprint(response)
  # save_json_file(results, "shared_data/frame_net_analysis.json", indent=2)
  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


# Get frames and save them to a JSON file
get_frames(sentence["text"])

"""
PROMPT:

This is the FrameNet analysis data. 
Step 1: Begin by identifying the specific issue in the sentence that is either supported or opposed 
(start with that!!!).
Step 2: Next, determine which specific FrameNet frame is the most crucial for classifying the sentence as expressing 
a supportive or opposing stance. This will assist in creating a rule for generalization: 
"""
