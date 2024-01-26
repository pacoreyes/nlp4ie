# from pprint import pprint

import torch
import tqdm
from setfit import SetFitModel

from db import spreadsheet_4
from lib.utils import load_jsonl_file, write_to_google_sheet


# Load dataset
dataset = load_jsonl_file("shared_data/dataset_3_7_unlabeled_sentences_1.jsonl")

# dataset = dataset[:2000]  # use it to test the code


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


# Load model
model_setfit_path = "models/3"
model = SetFitModel.from_pretrained(model_setfit_path, local_files_only=True)

# Get best device
device = get_device()

# Move the model to the chosen device
model.to(device)

# Create list of sentences
sentences = [datapoint["text"] for datapoint in dataset]
# pprint(sentences)

# Make predictions using the model
print("Making predictions...")
predictions = model.predict_proba(sentences)

# Move predictions to CPU
if predictions.is_cuda:
    predictions = predictions.cpu()

# Convert predictions to list
predictions_list = predictions.numpy().tolist()


# Filter predictions by class and generate inference
predictions = []
for idx, p in tqdm.tqdm(enumerate(predictions_list)):
  pred_class = None
  pred_score = 0
  if p[0] > p[1] and p[0] > 0:  # 0.9947 is the threshold
    pred_class = "support"
    pred_score = p[0]
  elif p[0] < p[1] and p[1] > 0:  # 0.9947 is the threshold
    pred_class = "oppose"
    pred_score = p[1]
  else:
    pred_class = "undefined"
  if pred_class in ["oppose", "support"]:
    """print(sentences[idx])
    print(pred_class)
    print(pred_score)
    print("-------")"""
    row = [
      dataset[idx]['id'],
      sentences[idx],
      pred_class,
      pred_score
    ]
    predictions.append(row)

# pprint(predictions)

# Write predictions to Google Sheet
print("Writing predictions to Google Sheet...")
write_to_google_sheet(spreadsheet_4, "error_analysis_0", predictions)
