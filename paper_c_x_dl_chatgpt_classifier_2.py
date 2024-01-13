from pprint import pprint

from sklearn.metrics import (confusion_matrix, roc_auc_score, matthews_corrcoef,
                             accuracy_score, precision_recall_fscore_support)

from lib.utils import load_json_file, load_jsonl_file, read_from_google_sheet
from lib.utils2 import anonymize_text, remove_duplicated_datapoints
from lib.visualizations import plot_confusion_matrix
from db import spreadsheet_4

import openai
import spacy
from tqdm import tqdm

# WITH_ANONYMIZATION = False

# The ID of your ChatGPT fine-tuned model

model_id = "ft:babbage-002:paco::8flKlgD8"  # anonymized
# model_id = "ft:davinci-002:paco::8fbaMxVv"  # non-anonymized

# -------------------------------------------------------------------------------------

# BEST model
# model_id = "ft:babbage-002:paco::8flKlgD8"  # anonymized

# -------------------------------------------------------------------------------------

# Your API key from OpenAI
openai.api_key = load_json_file("credentials/openai_credentials.json")["openai_api_key"]

# Load test dataset
dataset = read_from_google_sheet(spreadsheet_4, "analize_frames_")

# Remove duplicated datapoints
dataset = remove_duplicated_datapoints(dataset)

print("################################")

for datapoint in dataset:
  res = openai.Completion.create(model=model_id, prompt=datapoint["text"], max_tokens=1, temperature=0, logprobs=2)

  print(datapoint["text"])
  # print(f"Real class: {datapoint['label']}")
  print(f"Predicted class: {res['choices'][0]['text']}")
  pprint(res['choices'][0]['logprobs']['top_logprobs'][0])
  """oppose_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["op"]
  support_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["support"]
  print(f"Oppose: {oppose_prediction}")
  print(f"Support: {support_prediction}")"""




"""oppose_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["op"]
support_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["support"]
print(f"Oppose: {oppose_prediction}")
print(f"Support: {support_prediction}")

# Get the highest logprob
close_to_0 = closest_to_zero(oppose_prediction, support_prediction)
print(close_to_0)"""

"""print("--------------------------------")
if close_to_0 > -0.5:
  prediction = {
    "text": "oppose",
    "score": oppose_prediction
  }
  print(prediction)
elif close_to_0 > -0.5:
  prediction = {
    "text": "support",
    "score": support_prediction
  }
  print(prediction)
else:
  prediction = {
    "text": "undecided",
    "score": close_to_0
  }
  print(prediction)
print("--------------------------------")"""
