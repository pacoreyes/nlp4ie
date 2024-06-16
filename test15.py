from pprint import pprint
from lib.semantic_frames import api_get_frames

from db import spreadsheet_4
from lib.utils import read_from_google_sheet

dataset = read_from_google_sheet(spreadsheet_4, "dataset_3_final")

all_frames = set()

for datapoint in dataset:
  sentence = datapoint["text"]
  present_frames_response = api_get_frames(sentence, "localhost", "5001", "all")

  frames = present_frames_response["frames"]["textae"]

  frame_names = []
  for frame in frames:
      for idx, f in enumerate(frame["denotations"]):
        if idx == 0:
          all_frames.add(f["obj"])
          frame_names.append(f["obj"])

  pprint(frame_names)


all_frames = list(all_frames)
all_frames.sort()

#pprint(all_frames)
#print(len(all_frames))
