# from pprint import pprint

import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
from afinn import Afinn
import contractions

from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file
from lib.semantic_frames import api_get_frames, encode_sentence_frames
from lib.stance_markers import certainty_adj, certainty_adv, certainty_verbs, positive_affect_adj

# Set spaCy to use GPU if available
if spacy.prefer_gpu():
    print("spaCy is using GPU!")
else:
    print("GPU not available, spaCy is using CPU instead.")

# load transformer model
nlp = spacy.load("en_core_web_lg")

# Initialize Afinn sentiment analyzer
afinn = Afinn()

# Load datasets
dataset_train = load_jsonl_file("shared_data/dataset_2_train.jsonl")
dataset_validation = load_jsonl_file("shared_data/dataset_2_validation.jsonl")
# dataset_test = load_jsonl_file("shared_data/dataset_2_test.jsonl")

# Join datasets
dataset = dataset_train + dataset_validation

"""dataset1 = dataset[:10]
dataset = dataset1 + dataset[-10:]"""

output_file = "shared_data/dataset_2_2_1_features.jsonl"

# Empty JSONL file
empty_json_file(output_file)


def measure_word_length(_doc):
  return [len(token.text) for token in _doc if token.is_alpha]


def measure_sentence_complexity(_doc):
  # Number of tokens with specific dependencies
  # return [len([token for token in _doc if token.dep_ in ['advcl', 'relcl', 'acl', 'ccomp', 'xcomp']])]
  return [1 if token.dep_ in ['advcl', 'relcl', 'acl', 'ccomp', 'xcomp'] else 0 for token in _doc if not token.is_punct]


def measure_word_freq(_doc):
  # return len([token for token in _doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']])
  return [1 if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] else 0 for token in _doc if not token.is_punct]


def measure_negation_freq(_doc):
  # return len([token for token in _doc if token.dep_ == "neg"])
  return [1 if token.dep_ == "neg" else 0 for token in _doc if not token.is_punct]


def measure_modal_verb_freq(_doc):
  return [1 if token.tag_ == 'MD' else 0 for token in _doc]


def abc(_doc, terms, pos, pole):
  # Initialize lists
  single_token = []
  multi_token = []
  scores = []

  # Process each string to categorize it
  for string in terms:
    doc = nlp(string)
    # Count the number of tokens in the processed string
    num_tokens = len([token for token in doc if not token.is_punct])
    # Categorize based on the number of tokens
    if num_tokens > 1:
      multi_token.append(string)
    else:
      single_token.append(string)

  # Process each token in the _doc
  for token in _doc:
    # Get sentiment score
    score = afinn.score(token.text)
    # Check if the token is in the single token list
    if token.text in single_token and token.pos_ == pos and not token.is_stop:
      # scores.append(1)
      if pole == "pos" and score > 0:
        scores.append(1)
      elif pole == "neg" and score < 0:
        scores.append(1)
    else:
      scores.append(0)

  if multi_token:
    for string in multi_token:
      # Check if the string is in the _doc
      if string in _doc.text:
        score = afinn.score(string)
        if pole == "pos" and score > 0:
          scores.append(1)
        elif pole == "neg" and score < 0:
          scores.append(1)

  return scores


def measure_adjective_polarity(_doc, pole):
  # Get the polarity words
  polarity_score = []
  for token in _doc:
    # Get sentiment score
    score = afinn.score(token.text)
    if pole == "pos":
      if token.pos_ == "ADJ" and not token.is_stop and token.ent_type_ == "" and score > 0:
        polarity_score.append(score)
        # print(token.text, score)
      else:
        polarity_score.append(0)
    elif pole == "neg":
      if token.pos_ == "ADJ" and not token.is_stop and token.ent_type_ == "" and score < 0:
        polarity_score.append(abs(score))
      else:
        polarity_score.append(0)
  return polarity_score


def measure_verb_polarity(_doc, pole):
  # Get the polarity words
  polarity_score = []
  for token in _doc:
    # Get sentiment score
    score = afinn.score(token.text)
    if pole == "pos":
      if token.pos_ == "VERB" and not token.is_stop and token.ent_type_ == "" and score > 0:
        polarity_score.append(score)
      else:
        polarity_score.append(0)
    elif pole == "neg":
      if token.pos_ == "VERB" and not token.is_stop and token.ent_type_ == "" and score < 0:
        polarity_score.append(abs(score))
      else:
        polarity_score.append(0)
  return polarity_score


def measure_adverb_polarity(_doc, pole):
  # Get the polarity words
  polarity_score = []
  for token in _doc:
    # Get sentiment score
    score = afinn.score(token.text)
    if pole == "pos":
      if token.pos_ == "ADV" and not token.is_stop and token.ent_type_ == "" and score > 0:
        polarity_score.append(score)
      else:
        polarity_score.append(0)
    elif pole == "neg":
      if token.pos_ == "ADV" and not token.is_stop and token.ent_type_ == "" and score < 0:
        polarity_score.append(abs(score))
      else:
        polarity_score.append(0)
  return polarity_score


for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  # Remove contractions
  text = contractions.fix(datapoint["text"])

  # Query the semantic frames API
  # present_frames_response = api_get_frames(datapoint["text"], "localhost", "5001", "all")
  # Encode the frames with one-hot encoding
  # encoded_frames = encode_sentence_frames(present_frames_response)

  doc = nlp(datapoint["text"])

  row = {
    "id": datapoint["id"],
    # "text": datapoint["text"],
    "label": datapoint["label"],
    "word_length": measure_word_length(doc),
    "sentence_complexity": measure_sentence_complexity(doc),
    "lexical_word_freq": measure_word_freq(doc),
    "negation_freq": measure_negation_freq(doc),
    "modal_verb_freq": measure_modal_verb_freq(doc),
    "pos_adj": measure_adjective_polarity(doc, "pos"),
    "neg_adj": measure_adjective_polarity(doc, "neg"),
    "pos_adv": measure_adverb_polarity(doc, "pos"),
    "neg_adv": measure_adverb_polarity(doc, "neg"),
    "pos_verb": measure_verb_polarity(doc, "pos"),
    "neg_verb": measure_verb_polarity(doc, "neg"),

    # "semantic_frames": encoded_frames
  }
  save_row_to_jsonl_file(row, output_file)
