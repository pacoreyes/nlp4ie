# from pprint import pprint

import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
from afinn import Afinn
import contractions

from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file
from lib.semantic_frames import api_get_frames, encode_sentence_frames
from lib.stance_markers import certainty_adj, certainty_adv, certainty_verbs

# Set spaCy to use GPU if available
if spacy.prefer_gpu():
    print("spaCy is using GPU!")
else:
    print("GPU not available, spaCy is using CPU instead.")

# load transformer model
nlp = spacy.load("en_core_web_lg")
# Initialize matcher
matcher_certainty_adj = Matcher(nlp.vocab)
matcher_certainty_adv = Matcher(nlp.vocab)

# Pattern for certainty adjectives
pattern_certainty_adj1 = [
  # {"LOWER": {"IN": ["it", "that"]}},
  {"POS": {"IN": ["PRON", "NOUN"]}},
  {"POS": "ADV", "OP": "?"},
  {"LEMMA": {"IN": ["be", "seem"]}},
  {"LOWER": {"IN": ["apparent", "clear", "definite", "plain", "sure"]}, "POS": "ADJ"}
]
#
pattern_certainty_adj2 = [
  {"LOWER": {"IN": ["i", "we"]}},
  {"POS": "ADV", "OP": "?"},
  {"LEMMA": {"IN": ["be", "seem", "feel"]}},
  {"LOWER": {"IN": ["convinced", "definite", "sure"]}, "POS": "ADJ"}
]

matcher_certainty_adj.add("RULE_CERTAINTY_ADJ", [pattern_certainty_adj1, pattern_certainty_adj2])

# Pattern for certainty adverb
pattern_certainty_adv1 = [
    {"LEMMA": "be"},  # Matches any form of "to be"
    {"POS": "ADV"},  # Matches an adverb
    {"LOWER": {"IN": ["certain", "clear", "realistic", "sure"]}, "POS": "ADJ"}  # Matches the specified adjectives
]

matcher_certainty_adv.add("RULE_CERTAINTY_ADV", [pattern_certainty_adv1])


""" ----------------------------------------------  """

# Separate single token adjectives from multi-token adjectives
token_level_adj = []
multi_token_adj = []
for adj in certainty_adj:
  doc = nlp(adj)
  if len(doc) == 1:
    token_level_adj.append(adj)
  elif len(doc) > 1:
    multi_token_adj.append(adj)

# Separate single token adverbs from multi-token adverbs
token_level_adv = []
multi_token_adv = []
for adv in certainty_adv:
  doc = nlp(adv)
  if len(doc) == 1:
    token_level_adv.append(adv)
  elif len(doc) > 1:
    multi_token_adv.append(adv)

# Separate single token verbs from multi-token verbs
token_level_verbs = []
multi_token_verbs = []
for verb in certainty_verbs:
  doc = nlp(verb)
  if len(doc) == 1:
    token_level_verbs.append(verb)
  elif len(doc) > 1:
    multi_token_verbs.append(verb)


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


def measure_lexical_density(_doc):
  # return len([token for token in _doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']])
  return [1 if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] else 0 for token in _doc if not token.is_punct]


"""def measure_modal_verbs_use(_doc):
  mod_verbs = len([token for token in _doc if token.tag_ == 'MD'])
  return mod_verbs / len(_doc)"""


def measure_negation_use(_doc):
  # return len([token for token in _doc if token.dep_ == "neg"])
  return [1 if token.dep_ == "neg" else 0 for token in _doc if not token.is_punct]


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


"""def measure_certainty_adj(_doc):

  container = []

  for token in _doc:
    match = False
    for token_adj in token_level_adj:
      if token.text == token_adj and token.pos_ == "ADJ":
        match = True
        break
    if match:
      container.append(1)
    else:
      container.append(0)

  # multi-token level
  # multi_token_container = []
  for adj in multi_token_adj:
    if adj in _doc.text:
      container.append(1)
    else:
      container.append(0)

  # patterns
  matches = matcher_certainty_adj(_doc)
  if matches:
    container.append(1)
  else:
    container.append(0)

  return container"""


def measure_certainty_adj(_doc):
  # Token level
  container = [1 if token.text in token_level_adj and token.pos_ == "ADJ" else 0 for token in _doc]
  # multi-token level
  container.extend(1 if adj in _doc.text.lower() else 0 for adj in multi_token_adj)
  # patterns
  container.append(1 if matcher_certainty_adj(_doc) else 0)
  return container


def measure_certainty_adv(_doc):
  container = []

  # Check if first token of _doc (the sentence) is an adverb in the certainty_adv list
  if _doc[0].pos_ == "ADV":
    if _doc[0].text.lower() in ["clearly", "plainly", "surely"]:
      container.append(1)
    else:
      container.append(0)

  # Check if the sentence contains a clause that starts with an adverb in the certainty_adv list
  for token in _doc:

    if token == _doc[0]:  # Skip the first token
      continue
    if token.dep_ in ["advcl", "relcl"]:
      clause = token.subtree  # Get the clause
      clause_tokens = list(clause)  # Convert to list to access by index
      # Check if the first token of the clause is an adverb
      if clause_tokens[0].pos_ == "ADV":
        if clause_tokens[0].text.lower() in ["clearly", "plainly", "surely"]:
          container.append(1)
    else:
      if token.pos_ == "ADV":
        if token.text.lower() in token_level_adv:
          container.append(1)


      container.append(0)

  if 1 in container:
    print(container)
  return container


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
    "lexical_density": measure_lexical_density(doc),
    "negation_use": measure_negation_use(doc),
    "pos_adj": measure_adjective_polarity(doc, "pos"),
    "neg_adj": measure_adjective_polarity(doc, "neg"),
    "pos_adv": measure_adverb_polarity(doc, "pos"),
    "neg_adv": measure_adverb_polarity(doc, "neg"),
    "pos_verb": measure_verb_polarity(doc, "pos"),
    "neg_verb": measure_verb_polarity(doc, "neg"),
    "certainty_adj": measure_certainty_adj(doc),
    "certainty_adv": measure_certainty_adv(doc)

    # "semantic_frames": encoded_frames
  }
  save_row_to_jsonl_file(row, output_file)
