from pprint import pprint

from tqdm import tqdm

from db import spreadsheet_4
from lib.semantic_frames import api_get_frames
from lib.utils import write_to_google_sheet, read_from_google_sheet


# Initialize constants
dataset_input = read_from_google_sheet(spreadsheet_4, "dataset_3_final")
dataset_output = "dataset_3_final_c"

# dataset = dataset[:2000]

all_frame_names = set()

# Process dataset

data = []
for idx, rec in enumerate(tqdm(dataset_input, desc=f"Processing {len(dataset_input)} datapoints"), start=1):
  sentence = rec["text"]
  metadata = eval(rec["metadata"])

  frameBERT_response = api_get_frames(sentence, "localhost", "5001", "all")

  semantic_frames = frameBERT_response["frames"]["textae"]

  frame_names = []
  for frame in semantic_frames:
    for _idx, f in enumerate(frame["denotations"]):
      if _idx == 0:
        all_frame_names.add(f["obj"])
        frame_names.append(f["obj"])

  frame_names.sort()
  frame_names = list(set(frame_names))  # Remove duplicates
  metadata["semantic_frames"] = frame_names
  rec["metadata"] = str(metadata)

  data.append([
    rec["id"],
    rec["text"],
    rec["label"],
    rec["metadata"],
  ])

  # pprint(metadata)

all_frames = list(all_frame_names)
all_frames.sort()

pprint(all_frames)
print(f"\nExtracted semantic frames: {len(all_frames)}")

# Write data to Google Sheet
write_to_google_sheet(spreadsheet_4, dataset_output, data)
