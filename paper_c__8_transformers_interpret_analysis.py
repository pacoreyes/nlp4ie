import random

import spacy
from tqdm import tqdm
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers_interpret import PairwiseSequenceClassificationExplainer

from db import spreadsheet_7
from lib.utils import load_jsonl_file, write_to_google_sheet

SEED = 42


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


def set_seed(seed_value):
  """Set seed for reproducibility."""
  random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


# Set seed for reproducibility
set_seed(SEED)

# Set device
device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")

CLASS_NAMES = ['continue', 'not_continue']

GSHEET_SHAP = "_TI"

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
not_continue_test_dataset = load_jsonl_file("shared_data/topic_boundary_not_continue_class.jsonl")
continue_test_dataset = load_jsonl_file("shared_data/topic_boundary_continue_class.jsonl")

DATASET = not_continue_test_dataset + continue_test_dataset

# DATASET = DATASET[231:232]

# Initialize constants
BERT_MODEL = 'bert-base-uncased'
MODEL_PATH = 'models/3/TopicContinuityBERT.pth'

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("models/3/tokenizer")

# Load the model
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)

# Move the model to the device
# model = model.to(device)

# Load the model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# Set the model to evaluation mode
model.eval()

model.resize_token_embeddings(len(tokenizer))

# Explainer initialization
pairwise_explainer = PairwiseSequenceClassificationExplainer(
  model=model,
  tokenizer=tokenizer,
  custom_labels=CLASS_NAMES,
  attribution_type="lig",
)

rows = []

for example in tqdm(DATASET, desc="Processing dataset..."):
  # print(example["id"])
  text = example["text"]
  true_label = example["label_human"]
  sentence1, sentence2 = text.split('[SEP]')

  sentence1 = sentence1.strip()
  sentence2 = sentence2.strip()

  # Get the token length of the first sentence
  sentence1_length = len(tokenizer.tokenize(sentence1))

  explanation_data = pairwise_explainer(text1=sentence1, text2=sentence2)  # flip_sign=True,
  # print(explanation_data)

  # Remove rows with blank features ("")
  explanation_data = [row for row in explanation_data if row[0] not in ["[PAD]", "[CLS]", "[SEP]"]]
  # Aggregate subtokens
  # explanation_data = aggregate_subtokens(explanation_data)

  # Convert to DataFrame
  df_values = pd.DataFrame(explanation_data, columns=["feature", "value"])

  if not example.get("metadata"):
    continue

  metadata = eval(example["metadata"])

  """# Check if the example is misclassified
  if example['label_human'] != example['label_bert']:
    print(f"Classification Error: {example['id']}")
    continue
  else:"""
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
        shap_row1 = df_values.iloc[rb_index1]
        shap_row2 = df_values.iloc[sentence1_length + rb_index2]  # length of s1 + index in s2

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
          shap_row1["value"]
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
          shap_row2["value"]
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
        shap_row1 = df_values.iloc[rb_index1]
        shap_row2 = df_values.iloc[rb_index2]
        shap_row3 = df_values.iloc[sentence1_length + rb_index3]
        shap_row4 = df_values.iloc[sentence1_length + rb_index4]

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
          shap_row1["value"]
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
          shap_row2["value"]
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
          shap_row3["value"]
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
          shap_row4["value"]
        ]
        rows.append(row)

    # Feature 3: coreference (RB)
    if continuity_feature.get("coreference"):
      for feature in continuity_feature["coreference"][0]:

        # Word
        rb_word = nlp(feature[0])[0].lemma_.lower()
        # Index
        rb_index = feature[1]
        # Access the SHAP data from the DataFrame
        shap_row = df_values.iloc[rb_index]
        # Get words from SHAP data
        shap_word = shap_row["feature"]
        shap_word = nlp(shap_word)[0].lemma_.lower()

        if shap_word != rb_word:
          print(f"Mismatch ({example['id']}): |{shap_word}:{rb_word}|")

        row = [
          example["id"],
          feature[2],
          rb_index,
          rb_word,
          example["label_human"],
          example["label_rb"],
          example["label_bert"],
          "coreference",
          shap_row["value"]
        ]
        rows.append(row)

    # Feature 4: semantic continuity (RB)
    if continuity_feature.get("semantic_continuity"):
      for feature in continuity_feature["semantic_continuity"]:

        # Word from sentence 1
        rb_word1 = nlp(feature[0][0])[0].lemma_.lower()
        rb_index1 = feature[0][1]
        # Word from sentence 2
        rb_word2 = nlp(feature[1][0])[0].lemma_.lower()
        rb_index2 = feature[1][1]

        # Access the SHAP data from the DataFrame
        shap_row1 = df_values.iloc[rb_index1]
        shap_row2 = df_values.iloc[sentence1_length + rb_index2]

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
          "semantic",
          shap_row1["value"]
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
          "semantic",
          shap_row2["value"]
        ]
        rows.append(row)

    # Feature 5: transition markers (RB)
    if continuity_feature.get("transition_markers"):
      for feature in continuity_feature["transition_markers"]:

        # Transition word from sentence 2
        if feature.get("continue"):
          rb_word1 = nlp(feature["continue"])[0].lemma_.lower()

          rb_index1 = sentence1_length + 0

          # Access the SHAP data from the DataFrame
          shap_row1 = df_values.iloc[rb_index1]

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
            shap_row1["value"]
          ]
          rows.append(row)

        if feature.get("shift"):
          rb_word1 = nlp(feature["shift"])[0].lemma_.lower()
          rb_index1 = sentence1_length + 0

          # Access the SHAP data from the DataFrame
          shap_row1 = df_values.iloc[rb_index1]

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
            shap_row1["value"]
          ]
          # print(row)
          rows.append(row)

write_to_google_sheet(spreadsheet_7, GSHEET_SHAP, rows)
