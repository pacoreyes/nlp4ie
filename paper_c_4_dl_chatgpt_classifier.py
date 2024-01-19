from pprint import pprint

from sklearn.metrics import (confusion_matrix, roc_auc_score, matthews_corrcoef,
                             accuracy_score, precision_recall_fscore_support)

from lib.utils import load_json_file, load_jsonl_file
from lib.utils2 import anonymize_text
from lib.visualizations import plot_confusion_matrix

import openai
import spacy
from tqdm import tqdm

# Paper: Stance Detection Benchmark: How Robust is Your Stance Detection?
# https://link.springer.com/article/10.1007/s13218-021-00714-w
# https://www.pure.ed.ac.uk/ws/portalfiles/portal/205831185/Stance_detection_ALDAYEL_DOA20032021_AFV.pdf

# The ID of your ChatGPT fine-tuned model
model_id = "ft:davinci-002:paco::8ibJ3H2Y"  # anonymized BEST model
# model_id = "ft:davinci-002:paco::8fbaMxVv"  # non-anonymized

# -------------------------------------------------------------------------------------

# BEST model
# model_id = "ft:babbage-002:paco::8flKlgD8"  # anonymized babbage-002

# -------------------------------------------------------------------------------------

# Your API key from OpenAI
openai.api_key = load_json_file("credentials/openai_credentials.json")["openai_api_key"]

# Load test dataset
dataset = load_jsonl_file("shared_data/dataset_3_6_test_anonym.jsonl")

# Remap labels
dataset_remapped = []

print("Filter datapoints...")

for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  row = {"text": datapoint["prompt"], "label": datapoint["completion"]}
  dataset_remapped.append(row)


dataset = dataset_remapped

# Load spaCy model
# nlp_trf = spacy.load("en_core_web_trf")


def closest_to_zero(num1, num2):
  return min(num1, num2, key=abs)


print("################################")

for datapoint in dataset:

  res = openai.Completion.create(model=model_id, prompt=datapoint["text"], max_tokens=1, temperature=0, logprobs=2)

  print(datapoint["text"])
  print(f"Real class: {datapoint['label']}")
  print(f"Predicted class: {res['choices'][0]['text']}")
  # pprint(res)
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
