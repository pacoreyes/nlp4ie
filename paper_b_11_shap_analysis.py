import spacy
import torch
import shap
import pandas as pd
import numpy as np
from setfit import SetFitModel
from transformers import AutoTokenizer
from tqdm import tqdm

from db import spreadsheet_4
from lib.utils import load_jsonl_file, write_to_google_sheet


# Set spaCy to use GPU if available
if spacy.prefer_gpu():
    print("spaCy is using GPU!")
else:
    print("GPU not available, spaCy is using CPU instead.")


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


nlp = spacy.load("en_core_web_trf")

# Get best device
device = get_device()

CLASS_NAMES = ['support', 'oppose']  # 0/1

GSHEET = "SHAP"

model_id = "sentence-transformers/paraphrase-mpnet-base-v2"  # Example model_id

# Load dataset
test_dataset = load_jsonl_file("shared_data/dataset_2_test.jsonl")
validation_dataset = load_jsonl_file("shared_data/dataset_2_validation.jsonl")

DATASET = test_dataset + validation_dataset

support_class = [data for data in DATASET if data["label"] == "support"]
oppose_class = [data for data in DATASET if data["label"] == "oppose"]

# Load model
model_setfit_path = "models/22"
model = SetFitModel.from_pretrained(model_setfit_path, local_files_only=True, device=device)

# Load the corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)


def f(x):  # x is a list of strings
  # Make predictions using the model
  predictions = model.predict_proba(x)
  return predictions.cpu()


explainer = shap.Explainer(
  model=f,  # prediction function
  masker=tokenizer,
  output_names=["support", "oppose"],
  algorithm="auto",
  linearize_link=None,
  seed=42
)

dataset = support_class + oppose_class

# dataset = dataset[:5]

texts = [data["text"] for data in dataset]

shap_values_data = []

for idx, sentence in tqdm(enumerate(texts), "Analyzing SHAP values", total=len(texts)):
  shap_values = explainer([sentence], fixed_context=None)
  # print(shap_values)
  summed_shap_values = shap_values.mean(0).values
  summed_shap_features = shap_values.mean(0).feature_names

  _base_values = shap_values.base_values  # prediction
  predicted_class_index = np.argmax(_base_values, axis=1)[0]
  predicted_class_name = CLASS_NAMES[predicted_class_index]
  # print(f"Predicted class: {predicted_class_name}")

  # flattened_shap_values
  summed_shap_values = [row[0] for row in summed_shap_values]

  # Convert to DataFrame for easy CSV saving
  df_shap_values = pd.DataFrame(summed_shap_values, columns=["Value"])

  # Add a column for the feature names if available
  df_shap_values["Feature"] = summed_shap_features
  df_shap_values = df_shap_values.sort_values(by="Value", ascending=False)

  feature_value_pairs = []
  for idx2, value in df_shap_values.iterrows():
    row = {
      "feature": value["Feature"],
      "value": value["Value"]
    }
    feature_value_pairs.append(row)

  support_words = [word["feature"] for word in feature_value_pairs if word["value"] > 0 and word["feature"] != ""]
  oppose_words = [word["feature"] for word in feature_value_pairs if word["value"] < 0 and word["feature"] != ""]

  # Remove punctuations with spacy
  support_words = [word for word in support_words if not nlp(word)[0].is_punct]
  oppose_words = [word for word in oppose_words if not nlp(word)[0].is_punct]

  # Reverse the order of the oppose words
  oppose_words = oppose_words[::-1]

  row = [
    dataset[idx]["id"],
    dataset[idx]["label"],
    predicted_class_name,
    sentence,
    str(support_words),
    str(oppose_words),
    str(feature_value_pairs),
    str(dataset[idx]["metadata"]),
  ]
  shap_values_data.append(row)
  # print(row)

# Write to Google Sheet
write_to_google_sheet(spreadsheet_4, GSHEET, shap_values_data)
