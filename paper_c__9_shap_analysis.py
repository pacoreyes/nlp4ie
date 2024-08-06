# from pprint import pprint

import spacy
import shap
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from db import spreadsheet_7
from lib.utils import load_jsonl_file, write_to_google_sheet

SEED = 42
CLASS_NAMES = ['continue', 'not_continue']

# Load dataset
test_dataset_continue = load_jsonl_file("shared_data/topic_boundary_continue_class.jsonl")
test_dataset_not_continue = load_jsonl_file("shared_data/topic_boundary_not_continue_class.jsonl")
DATASET = test_dataset_not_continue + test_dataset_continue

# DATASET = DATASET[146:147]  # before:pointer

GSHEET_SHAP = "__SHAP"
GSHEET_EA = "EA"


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


# Set device
device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize constants
BERT_MODEL = "bert-base-uncased"
MODEL_PATH = "models/3/TopicContinuityBERT.pth"

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Never split tokens
tokenizer.add_tokens(["1,000", "2,000", "endures", "decency", "stockpile", "ventilators", "Blackwater", "standpoint",
                      "dismantle", "empower", "frack", "polluters", "Saddamists", "rejectionists", "Qaida", "maiming",
                      "torturing", "healthier", "massively", "asymptomatic", "Pocan", "unfairly", "1,400", "'s",
                      "62,000", "hospitalizations", "490,050", "commend", "F-16", "opioid", "pushers", "peddling",
                      "Ebola", "czar", "reiterate", "USAID", "maximally", "unwittingly", "'d", "Assad", "pandemic",
                      "deadliest", "defunding", "ATF", "pressuring", "DACA", "U.S.", "standpoint", "basing",
                      "hospitalization", "COVID", "incentivize", "reimagine", "dictate", "beneficiary", "closures",
                      "lawmakers", "equipping", "vaccination", "retrain", "Hun-", "nutritious", "inhumane",
                      "qualifies", "lifeblood", "forecasts", "vaccinated", "1619", "hundreds", "70,000", "legislating",
                      "Javits", "childcare", "reemphasized", "destabilizing", "COVID-19", "vaccinations", "ISR",
                      "Abound", "1,500", "FDIC", "2.9", "IndyMac", "5,000", "borrowers", "foreclosure", "mortgages",
                      "2.2", "pand"])

# Load the model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(CLASS_NAMES))

# Move the model to the device
model = model.to(device)
# Load the model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# Set the model to evaluation mode
model.eval()


def predict(_texts):
  encoding = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=_texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"  # Return PyTorch tensors
  )
  input_ids = encoding["input_ids"].to(device)
  attention_mask = encoding["attention_mask"].to(device)

  _logits = model(input_ids, attention_mask=attention_mask)[0]
  _probabilities = _logits.detach().cpu().numpy()
  return _probabilities


# Initialize the SHAP explainer
explainer = shap.Explainer(
  model=predict,
  masker=tokenizer,
  output_names=CLASS_NAMES,
  seed=SEED
)

rows = []
classification_errors = []

for example in tqdm(DATASET, desc="Processing dataset..."):

  text = example["text"]
  sentence1, sentence2 = text.split('[SEP]')
  sentence1 = sentence1.strip()
  sentence2 = sentence2.strip()
  text = sentence1 + " " + sentence2

  # Get the token length of the first sentence
  sentence1_length = len(tokenizer.tokenize(sentence1))

  # print()
  # print("--------------------------------------------------------------------------")
  """print(f"S1: {sentence1}")
  print(f"S2: {sentence2}")
  print(f"Label human: {example['label_human']}")
  print(f"Label RB: {example['label_rb']}")
  print(f"Label BERT: {example['label_bert']}")"""
  # print("--------------------------------------------------------------------------")
  # print()

  # Compute SHAP values for the selected samples
  shap_values = explainer([text], fixed_context=1)

  _shap_values = [(shap_values.values[0][i][0], shap_values.values[0][i][1], shap_values.feature_names[0][i])
                  for i in range(len(shap_values.values[0]))]

  # Remove rows with blank features ("")
  _shap_values = [row for row in _shap_values if row[2] != ""]

  # Convert to DataFrame
  df_shap_values = pd.DataFrame(_shap_values, columns=["continue", "not_continue", "feature"])

  if not example.get("metadata"):
    continue

  metadata = eval(example["metadata"])

  # Check if the example is misclassified
  if example['label_human'] != example['label_bert']:
    print(f"Classification Error: {example['id']}")
    classification_errors.append([
      example['id'],
      example["label_human"],
      example["label_bert"],
    ])
  else:
    for continuity_feature in metadata:

      # Feature 1: Lexical continuity (RB)
      if continuity_feature.get("lexical_continuity"):
        for feature in continuity_feature["lexical_continuity"]:

          # Word from sentence 1
          rb_word1 = nlp(feature[0][0])[0].lemma_.lower()
          rb_index1 = feature[0][1]
          # Word from sentence 2
          rb_word2 = nlp(feature[1][0])[0].lemma_.lower()
          rb_index2 = feature[1][1]

          # Access the SHAP data from the DataFrame
          shap_row1 = df_shap_values.iloc[rb_index1]
          shap_row2 = df_shap_values.iloc[sentence1_length + rb_index2]  # length of s1 + index in s2

          # Get words from SHAP data
          shap_word1 = shap_row1["feature"]
          shap_word1 = nlp(shap_word1)[0].lemma_.lower()
          shap_word2 = shap_row2["feature"]
          shap_word2 = nlp(shap_word2)[0].lemma_.lower()

          # Check if the words match
          if shap_word1 != rb_word1 or shap_word2 != rb_word2:
            print(f"Mismatch ({example['id']}): |{shap_word1}:{rb_word1}| / |{shap_word2}:{rb_word2}|")

          row = [
            example["id"],
            "s1",
            rb_index1,
            rb_word1,
            example["label_human"],
            example["label_rb"],
            example["label_bert"],
            "lexical",
            shap_row1[example['label_bert']]
          ]
          rows.append(row)

          row = [
            example["id"],
            "s2",
            rb_index2,
            rb_word2,
            example["label_human"],
            example["label_rb"],
            example["label_bert"],
            "lexical",
            shap_row2[example['label_bert']]
          ]
          rows.append(row)

      # Feature 2: syntactic continuity (RB)
      if continuity_feature.get("syntactic_continuity"):
        for feature in continuity_feature["syntactic_continuity"]:

          # Word 1 from sentence 1
          rb_word1 = nlp(feature[0][1][0])[0].lemma_.lower()
          rb_index1 = feature[0][1][1]
          # Word 2 from sentence 1
          rb_word2 = nlp(feature[0][0][0])[0].lemma_.lower()
          rb_index2 = feature[0][0][1]
          # Word 1 from sentence 2
          rb_word3 = nlp(feature[1][1][0])[0].lemma_.lower()
          rb_index3 = feature[1][1][1]
          # Word 2 from sentence 2
          rb_word4 = nlp(feature[1][0][0])[0].lemma_.lower()
          rb_index4 = feature[1][0][1]

          # Access the SHAP data from the DataFrame
          shap_row1 = df_shap_values.iloc[rb_index1]
          shap_row2 = df_shap_values.iloc[rb_index2]
          shap_row3 = df_shap_values.iloc[sentence1_length + rb_index3]
          shap_row4 = df_shap_values.iloc[sentence1_length + rb_index4]

          # Get words from SHAP data
          shap_word1 = shap_row1["feature"]
          shap_word1 = nlp(shap_word1)[0].lemma_.lower()
          shap_word2 = shap_row2["feature"]
          shap_word2 = nlp(shap_word2)[0].lemma_.lower()
          shap_word3 = shap_row3["feature"]
          shap_word3 = nlp(shap_word3)[0].lemma_.lower()
          shap_word4 = shap_row4["feature"]
          shap_word4 = nlp(shap_word4)[0].lemma_.lower()

          # Check if the words match
          if shap_word1 != rb_word1 or shap_word2 != rb_word2 or shap_word3 != rb_word3 or shap_word4 != rb_word4:
            print(f"Mismatch ({example['id']}): |{shap_word1}:{rb_word1}| / |{shap_word2}:{rb_word2}| "
                  f"/ |{shap_word3}:{rb_word3}| / |{shap_word4}:{rb_word4}|")

          row = [
            example["id"],
            "s1",
            rb_index1,
            rb_word1,
            example["label_human"],
            example["label_rb"],
            example["label_bert"],
            "syntactic",
            shap_row1[example['label_bert']]
          ]
          rows.append(row)

          row = [
            example["id"],
            "s1",
            rb_index2,
            rb_word2,
            example["label_human"],
            example["label_rb"],
            example["label_bert"],
            "syntactic",
            shap_row2[example['label_bert']]
          ]
          rows.append(row)

          row = [
            example["id"],
            "s2",
            rb_index3,
            rb_word3,
            example["label_human"],
            example["label_rb"],
            example["label_bert"],
            "syntactic",
            shap_row3[example['label_bert']]
          ]
          rows.append(row)

          row = [
            example["id"],
            "s2",
            rb_index4,
            rb_word4,
            example["label_human"],
            example["label_rb"],
            example["label_bert"],
            "syntactic",
            shap_row4[example['label_bert']]
          ]
          rows.append(row)

      # Feature 3: coreference (RB)
      if continuity_feature.get("coreference"):
        for feature in continuity_feature["coreference"]:

          # Word from sentence 1
          rb_word1 = nlp(feature[0][0])[0].lemma_.lower()
          rb_index1 = feature[0][1]
          # Word from sentence 2
          rb_word2 = nlp(feature[1][0])[0].lemma_.lower()
          rb_index2 = feature[1][1]

          # Access the SHAP data from the DataFrame
          shap_row1 = df_shap_values.iloc[rb_index1]
          shap_row2 = df_shap_values.iloc[rb_index2]

          # Get words from SHAP data
          shap_word1 = shap_row1["feature"]
          shap_word1 = nlp(shap_word1)[0].lemma_.lower()
          shap_word2 = shap_row2["feature"]
          shap_word2 = nlp(shap_word2)[0].lemma_.lower()

          # Check if the words match
          if shap_word1 != rb_word1 or shap_word2 != rb_word2:
            print(f"Mismatch ({example['id']}): |{shap_word1}:{rb_word1}| / |{shap_word2}:{rb_word2}|")

          row = [
            example["id"],
            "s1",
            rb_index1,
            rb_word1,
            example["label_human"],
            example["label_rb"],
            example["label_bert"],
            "coreference",
            shap_row1[example['label_bert']]
          ]
          rows.append(row)

          row = [
            example["id"],
            "s2",
            rb_index2,
            rb_word2,
            example["label_human"],
            example["label_rb"],
            example["label_bert"],
            "coreference",
            shap_row2[example['label_bert']]
          ]
          rows.append(row)

      # Feature 4: transition markers (RB)
      if continuity_feature.get("transition_markers"):
        for feature in continuity_feature["transition_markers"]:

          # Transition word from sentence 2
          if feature.get("continue"):
            rb_word1 = nlp(feature["continue"])[0].lemma_.lower()

            rb_index1 = sentence1_length + 0

            # Access the SHAP data from the DataFrame
            shap_row1 = df_shap_values.iloc[rb_index1]

            # Get words from SHAP data
            shap_word1 = shap_row1["feature"]
            shap_word1 = nlp(shap_word1)[0].lemma_.lower()

            row = [
              example["id"],
              "s2",
              rb_index1,
              rb_word1,
              example["label_human"],
              example["label_rb"],
              example["label_bert"],
              "transition_markers",
              shap_row1[example['label_bert']]
            ]
            rows.append(row)

          if feature.get("shift"):
            rb_word1 = nlp(feature["shift"])[0].lemma_.lower()
            rb_index1 = sentence1_length + 0

            # Access the SHAP data from the DataFrame
            shap_row1 = df_shap_values.iloc[rb_index1]

            # Get words from SHAP data
            shap_word1 = shap_row1["feature"]
            shap_word1 = nlp(shap_word1)[0].lemma_.lower()

            # Check if the words match
            if shap_word1 != rb_word1:
              print(f"Mismatch ({example['id']}): |{shap_word1}:{rb_word1}|")

            row = [
              example["id"],
              "s2",
              rb_index1,
              rb_word1,
              example["label_human"],
              example["label_rb"],
              example["label_bert"],
              "transition_markers",
              shap_row1[example['label_bert']]
            ]
            # print(row)
            rows.append(row)

write_to_google_sheet(spreadsheet_7, GSHEET_SHAP, rows)
write_to_google_sheet(spreadsheet_7, GSHEET_EA, classification_errors)
