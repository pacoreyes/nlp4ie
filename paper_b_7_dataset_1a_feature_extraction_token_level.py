from collections import Counter

import spacy
from spacy.matcher import Matcher

from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file
from lib.semantic_frames import api_get_frames, encode_sentence_frames

# Import stance markers for adjectives
from lib.stance_markers_adj import (
  create_positive_adj,
  create_negative_adj,
  create_certainty_adj,
  create_doubt_adj,
  create_emphatic_adj,
  create_hedge_adj,
  create_pro_adj,
  create_con_adj,
)

# Import stance markers for adverbs
from lib.stance_markers_adv import (
  create_positive_adv,
  create_negative_adv,
  create_certainty_adv,
  create_doubt_adv,
  create_emphatic_adv,
  create_hedge_adv,
  create_pro_adv,
  create_con_adv,
)

# Import stance markers for verbs
from lib.stance_markers_verb import (
  create_positive_verb,
  create_negative_verb,
  create_certainty_verb,
  create_doubt_verb,
  create_emphatic_verb,
  create_hedge_verb,
  create_pro_verb,
  create_con_verb,
)

# Import stance markers for modality
from lib.stance_markers_modals import (
  create_predictive_modal,
  create_possibility_modal,
  create_necessity_modal,
)

# Set spaCy to use GPU if available
if spacy.prefer_gpu():
    print("spaCy is using GPU!")
else:
    print("GPU not available, spaCy is using CPU instead.")

# load transformer model
nlp = spacy.load("en_core_web_trf")

# Load datasets
dataset_train = load_jsonl_file("shared_data/dataset_2_train.jsonl")
dataset_validation = load_jsonl_file("shared_data/dataset_2_validation.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_2_test.jsonl")

# Join datasets
dataset_train = dataset_train + dataset_validation

dataset = dataset_train + dataset_test

# pprint(dataset)
"""dataset1 = dataset[:100]
dataset = dataset1 + dataset[-100:]"""

output_file_training = "shared_data/dataset_2_2_1a_train_features.jsonl"
output_file_test = "shared_data/dataset_2_2_1a_test_features.jsonl"

# Empty JSONL files
empty_json_file(output_file_training)
empty_json_file(output_file_test)

# Initialize counters
feature_counter = Counter()

"""
---------------------- Adjective Features ----------------------
"""

# Create matchers for positive adjectives
matcher_positive_adj = create_positive_adj(Matcher(nlp.vocab))
"""matcher_positive_adj_negation = create_positive_adj_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_positive_adj_phrase = create_positive_adj_phrase(nlp, phrase_matcher)"""

# Create matchers for negative adjectives
matcher_negative_adj = create_negative_adj(Matcher(nlp.vocab))
"""matcher_negative_adj_negation = create_negative_adj_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_negative_adj_phrase = create_negative_adj_phrase(nlp, phrase_matcher)"""

# Create matchers for certainty adjectives
matcher_certainty_adj = create_certainty_adj(Matcher(nlp.vocab))
"""matcher_certainty_adj_negation = create_certainty_adj_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_certainty_adj_phrase = create_certainty_adj_phrase(nlp, phrase_matcher)"""

# Create matchers for doubt adjectives
matcher_doubt_adj = create_doubt_adj(Matcher(nlp.vocab))
# matcher_doubt_adj_negation = create_doubt_adj_negation(Matcher(nlp.vocab))

# Create matchers for emphatic adjectives
matcher_emphatic_adj = create_emphatic_adj(Matcher(nlp.vocab))
"""matcher_emphatic_adj_negation = create_emphatic_adj_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_emphatic_adj_phrase = create_emphatic_adj_phrase(nlp, phrase_matcher)
matcher_emphatic_adj_patterns = create_emphatic_adj_patterns(Matcher(nlp.vocab))  # Patterns"""

# Create matchers for hedge adjectives
matcher_hedge_adj = create_hedge_adj(Matcher(nlp.vocab))
# matcher_hedge_adj_negation = create_hedge_adj_negation(Matcher(nlp.vocab))

# Create matchers for pro adjectives
matcher_pro_adj = create_pro_adj(Matcher(nlp.vocab))
"""matcher_pro_adj_negation = create_pro_adj_negation(Matcher(nlp.vocab))
matcher_pro_adj_patterns = create_pro_adj_patterns(Matcher(nlp.vocab))  # Patterns"""

# Create matchers for con adjectives
matcher_con_adj = create_con_adj(Matcher(nlp.vocab))
"""matcher_con_adj_negation = create_con_adj_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_con_adj_phrase = create_con_adj_phrase(nlp, phrase_matcher)
matcher_con_adj_patterns = create_con_adj_patterns(Matcher(nlp.vocab))  # Patterns"""

"""
---------------------- Adverbs Features ----------------------
"""

# Create matchers for positive adverbs
matcher_positive_adv = create_positive_adv(Matcher(nlp.vocab))
# matcher_positive_adv_negation = create_positive_adv_negation(Matcher(nlp.vocab))

# Create matchers for negative adverbs
matcher_negative_adv = create_negative_adv(Matcher(nlp.vocab))
# matcher_negative_adv_negation = create_negative_adv_negation(Matcher(nlp.vocab))

# Create matchers for certainty adverbs
matcher_certainty_adv = create_certainty_adv(Matcher(nlp.vocab))
"""matcher_certainty_adv_negation = create_certainty_adv_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_certainty_adv_phrase = create_certainty_adv_phrase(nlp, phrase_matcher)"""

# Create matchers for doubt adverbs
matcher_doubt_adv = create_doubt_adv(Matcher(nlp.vocab))
# matcher_doubt_adv_negation = create_doubt_adv_negation(Matcher(nlp.vocab))

# Create matchers for emphatic adverbs
matcher_emphatic_adv = create_emphatic_adv(Matcher(nlp.vocab))
"""matcher_emphatic_adv_negation = create_emphatic_adv_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_emphatic_adv_phrase = create_emphatic_adv_phrase(nlp, phrase_matcher)
matcher_emphatic_adv_patterns = create_emphatic_adv_patterns(Matcher(nlp.vocab))  # Patterns"""

# Create matchers for hedge adverbs
matcher_hedge_adv = create_hedge_adv(Matcher(nlp.vocab))
"""matcher_hedge_adv_negation = create_hedge_adv_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_hedge_adv_phrase = create_hedge_adv_phrase(nlp, phrase_matcher)"""

# Create matchers for pro adverbs
matcher_pro_adv = create_pro_adv(Matcher(nlp.vocab))
"""matcher_pro_adv_negation = create_pro_adv_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_pro_adv_phrase = create_pro_adv_phrase(nlp, phrase_matcher)"""

# Create matchers for con adverbs
matcher_con_adv = create_con_adv(Matcher(nlp.vocab))
"""matcher_con_adv_negation = create_con_adv_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_con_adv_phrase = create_con_adv_phrase(nlp, phrase_matcher)
matcher_con_adv_patterns = create_con_adv_patterns(Matcher(nlp.vocab))  # Patterns"""

"""
---------------------- Verbs Features ----------------------
"""

# Create matchers for positive verbs
matcher_positive_verb = create_positive_verb(Matcher(nlp.vocab))
# matcher_positive_verb_negation = create_positive_verb_negation(Matcher(nlp.vocab))

# Create matchers for negative verbs
matcher_negative_verb = create_negative_verb(Matcher(nlp.vocab))
# matcher_negative_verb_negation = create_negative_verb_negation(Matcher(nlp.vocab))

# Create matchers for certainty verbs
matcher_certainty_verb = create_certainty_verb(Matcher(nlp.vocab))
"""matcher_certainty_verb_negation = create_certainty_verb_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_certainty_verb_phrase = create_certainty_verb_phrase(nlp, phrase_matcher)
matcher_certainty_verb_patterns = create_certainty_verb_patterns(Matcher(nlp.vocab))  # Patterns"""

# Create matchers for doubt verbs
matcher_doubt_verb = create_doubt_verb(Matcher(nlp.vocab))
# matcher_doubt_verb_negation = create_doubt_verb_negation(Matcher(nlp.vocab))

# Create matchers for emphatic verbs
matcher_emphatic_verb = create_emphatic_verb(Matcher(nlp.vocab))
"""matcher_emphatic_verb_negation = create_emphatic_verb_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_emphatic_verb_phrase = create_emphatic_verb_phrase(nlp, phrase_matcher)
matcher_emphatic_verb_patterns = create_emphatic_verb_patterns(Matcher(nlp.vocab))  # Patterns"""

# Create matchers for hedge verbs
matcher_hedge_verb = create_hedge_verb(Matcher(nlp.vocab))
"""matcher_hedge_verb_negation = create_hedge_verb_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_hedge_verb_phrase = create_hedge_verb_phrase(nlp, phrase_matcher)"""

# Create matchers for pro verbs
matcher_pro_verb = create_pro_verb(Matcher(nlp.vocab))
"""matcher_pro_verb_negation = create_pro_verb_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_pro_verb_phrase = create_pro_verb_phrase(nlp, phrase_matcher)
matcher_pro_verb_patterns = create_pro_verb_patterns(Matcher(nlp.vocab))  # Patterns"""

# Create matchers for con verbs
matcher_con_verb = create_con_verb(Matcher(nlp.vocab))
"""matcher_con_verb_negation = create_con_verb_negation(Matcher(nlp.vocab))
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_con_verb_phrase = create_con_verb_phrase(nlp, phrase_matcher)
matcher_con_verb_patterns = create_con_verb_patterns(Matcher(nlp.vocab))  # Patterns"""

"""
---------------------- Modality ----------------------
"""

# Create matchers for predictive modality
matcher_predictive_modal = create_predictive_modal(Matcher(nlp.vocab))
"""phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
matcher_predictive_modal_phrases = create_predictive_modal_phrase(nlp, phrase_matcher)"""

# Create matchers for possibility modality
matcher_possibility_modal = create_possibility_modal(Matcher(nlp.vocab))

# Create matchers for necessity modality
matcher_necessity_modal = create_necessity_modal(Matcher(nlp.vocab))
"""phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
matcher_necessity_modal_phrases = create_necessity_modal_phrase(nlp, phrase_matcher)"""


def extract_stance_modality(_doc, modality, label):

  """
  This function extracts stance modality features from a sentence.
  :param _doc: The sentence in a spaCy doc format
  :param modality: The modality type (predictive, possibility, necessity)
  :param label: The label of the sentence
  :return: List of 0/1 values
  """

  """1. Create matcher with lexicon and find/separate matches"""

  _matcher_single = None
  _matcher_phrase = None

  if modality == "predictive":
    _matcher_single = matcher_predictive_modal
    #_matcher_phrase = matcher_predictive_modal_phrases
  elif modality == "possibility":
    _matcher_single = matcher_possibility_modal
  elif modality == "necessity":
    _matcher_single = matcher_necessity_modal
    #_matcher_phrase = matcher_necessity_modal_phrases

  # Lists to hold matches based on their number of tokens
  _single_token_matches = []
  _multi_token_matches = []

  all_matches = []

  # Find lexicon matches
  _matches_single = _matcher_single(_doc)
  all_matches.append(_matches_single)
  """if _matcher_phrase:
    _matches_phrase = _matcher_phrase(_doc)
    all_matches.append(_matches_phrase)"""

  # Extract indexes of tokens in the sentence, separating single and multi-token matches
  for _match in all_matches:
    for _, start, end in _match:
      num_tokens = end - start
      if num_tokens > 1:
        _multi_token_matches.append([start, end])
      else:  # Single-token
        _single_token_matches.append(start)
      # update counter
      if label == "support":
        feature_counter[f"support_modality_{modality}"] += 1
      elif label == "oppose":
        feature_counter[f"oppose_modality_{modality}"] += 1

  """2. Create a new doc and "token map" after removing punctuation"""

  token_map = []
  tokens = []

  # Create lists of indexes and tokens in the sentence skipping punctuation
  for token in _doc:
    undesirable_puncts = ["!", ".", "?", ",", ";", ":", "-", "–", "—"]
    if token.text in undesirable_puncts:
      continue
    token_map.append(token.i)  # Collect indexes of tokens in map
    tokens.append(token.text)  # Collect tokens

  # Create a new doc with the cleaned tokens
  tokenized_doc = nlp(" ".join(tokens))

  """3. Score tokens based on lexicon"""

  # Initialize list to hold extracted values (0/1)
  extracted_features = []

  # Iterate sentence tokens and score them based on matches
  for idx, token_level1 in enumerate(token_map):
    _token = tokenized_doc[idx]
    # Check if index token is in a single-token match
    if token_level1 in _single_token_matches:
      print(f"- {_token} ({_token.i}) - Modality {modality} (lexicon word)")
      extracted_features.append(1)
    elif _multi_token_matches:  # If sentence has a multi-token match
      for item2 in _multi_token_matches:
        # Check if index token is in a multi-token match
        if token_level1 in range(item2[0], item2[1]):
          print(f"- {_token} ({_token.i}) - Modality {modality} (lexicon phrase)")
          extracted_features.append(1)
        else:
          extracted_features.append(0)
    else:
      extracted_features.append(0)

  return extracted_features


def extract_stance_features(_doc, pos, feature, label):
  """
  This function extracts stance features from a sentence based on
  1. lexicon of terms that indicate stance
  2. linguistic patterns that indicate stance
  3. sentiment score that indicates stance

  :param _doc: The sentence
  :param pos: part of speech
  :param feature: stance feature (positive, negative, certainty, doubt, emphatic, hedge, pro, con)
  :param label: The label of the sentence
  :return: List of 0/1 values
  """

  """1. Create matcher with lexicon and find/separate matches"""

  _matcher_single = None
  # _matcher_phrase = None
  # _matcher_neg = None
  # _matcher_pattern = None

  """_matcher_pos_neg = None
  _matcher_neg_neg = None
  _matcher_cert_neg = None
  _matcher_doubt_neg = None
  _matcher_emph_neg = None"""

  # Adjective
  if pos == "ADJ" and feature == "positive":
    _matcher_single = matcher_positive_adj
    #_matcher_phrase = phrase_matcher_positive_adj_phrase
    #_matcher_neg = matcher_negative_adj_negation
  elif pos == "ADJ" and feature == "negative":
    _matcher_single = matcher_negative_adj
    #_matcher_phrase = phrase_matcher_negative_adj_phrase
    #_matcher_neg = matcher_positive_adj_negation
  elif pos == "ADJ" and feature == "certainty":
    _matcher_single = matcher_certainty_adj
    #_matcher_phrase = phrase_matcher_certainty_adj_phrase
    #_matcher_neg = matcher_doubt_adj_negation
  elif pos == "ADJ" and feature == "doubt":
    _matcher_single = matcher_doubt_adj
    #_matcher_neg = matcher_certainty_adj_negation
  elif pos == "ADJ" and feature == "emphatic":
    _matcher_single = matcher_emphatic_adj
    #_matcher_phrase = phrase_matcher_emphatic_adj_phrase
    #_matcher_neg = matcher_hedge_adj_negation
  elif pos == "ADJ" and feature == "hedge":
    _matcher_single = matcher_hedge_adj
    #_matcher_neg = matcher_emphatic_adj_negation
    # _matcher_pos_neg = matcher_positive_adj_negation
    # _matcher_neg_neg = matcher_negative_adj_negation
    # _matcher_cert_neg = matcher_certainty_adj_negation
    # _matcher_doubt_neg = matcher_doubt_adj_negation
    # _matcher_emph_neg = matcher_emphatic_adj_negation
  elif pos == "ADJ" and feature == "pro":
    _matcher_single = matcher_pro_adj
    #_matcher_neg = matcher_con_adj_negation
  elif pos == "ADJ" and feature == "con":
    _matcher_single = matcher_con_adj
    #_matcher_phrase = phrase_matcher_con_adj_phrase
    #_matcher_neg = matcher_pro_adj_negation  # negation using pro

  # Adverb
  elif pos == "ADV" and feature == "positive":
    _matcher_single = matcher_positive_adv
    #_matcher_neg = matcher_negative_adv_negation
  elif pos == "ADV" and feature == "negative":
    _matcher_single = matcher_negative_adv
    #_matcher_neg = matcher_positive_adv_negation
  elif pos == "ADV" and feature == "certainty":
    _matcher_single = matcher_certainty_adv
    #_matcher_phrase = phrase_matcher_certainty_adv_phrase
    #_matcher_neg = matcher_doubt_adv_negation
  elif pos == "ADV" and feature == "doubt":
    _matcher_single = matcher_doubt_adv
    #_matcher_neg = matcher_certainty_adv_negation
  elif pos == "ADV" and feature == "emphatic":
    _matcher_single = matcher_emphatic_adv
    #_matcher_phrase = phrase_matcher_emphatic_adv_phrase
    #_matcher_neg = matcher_hedge_adv_negation
  elif pos == "ADV" and feature == "hedge":
    _matcher_single = matcher_hedge_adv
    #_matcher_phrase = phrase_matcher_hedge_adv_phrase
    #_matcher_neg = matcher_emphatic_adv_negation
    # _matcher_neg = matcher_hedge_adv_negation
    # _matcher_pos_neg = matcher_positive_adv_negation
    # _matcher_neg_neg = matcher_negative_adv_negation
    #_matcher_cert_neg = matcher_certainty_adv_negation
    # _matcher_doubt_neg = matcher_doubt_adv_negation
  elif pos == "ADV" and feature == "pro":
    _matcher_single = matcher_pro_adv
    #_matcher_phrase = phrase_matcher_pro_adv_phrase
    #_matcher_neg = matcher_con_adv_negation
  elif pos == "ADV" and feature == "con":
    _matcher_single = matcher_con_adv
    #_matcher_phrase = phrase_matcher_con_adv_phrase
    #_matcher_neg = matcher_pro_adv_negation  # negation using pro

  # Verb
  elif pos == "VERB" and feature == "positive":
    _matcher_single = matcher_positive_verb
    #_matcher_neg = matcher_negative_verb_negation
  elif pos == "VERB" and feature == "negative":
    _matcher_single = matcher_negative_verb
    #_matcher_neg = matcher_positive_verb_negation
  elif pos == "VERB" and feature == "certainty":
    _matcher_single = matcher_certainty_verb
    #_matcher_phrase = phrase_matcher_certainty_verb_phrase
    #_matcher_neg = matcher_doubt_verb_negation
  elif pos == "VERB" and feature == "doubt":
    _matcher_single = matcher_doubt_verb
    #_matcher_neg = matcher_certainty_verb_negation
  elif pos == "VERB" and feature == "emphatic":
    _matcher_single = matcher_emphatic_verb
    #_matcher_phrase = phrase_matcher_emphatic_verb_phrase
    #_matcher_neg = matcher_hedge_verb_negation
  elif pos == "VERB" and feature == "hedge":
    _matcher_single = matcher_hedge_verb
    #_matcher_phrase = phrase_matcher_hedge_verb_phrase
    #_matcher_neg = matcher_emphatic_verb_negation
    # _matcher_neg = matcher_hedge_verb_negation
    # _matcher_pos_neg = matcher_positive_verb_negation
    # _matcher_neg_neg = matcher_negative_verb_negation
    # _matcher_cert_neg = matcher_certainty_verb_negation
    # _matcher_doubt_neg = matcher_doubt_verb_negation
    # _matcher_emph_neg = matcher_emphatic_verb_negation
  elif pos == "VERB" and feature == "pro":
    _matcher_single = matcher_pro_verb
    #_matcher_phrase = phrase_matcher_pro_verb_phrase
    #_matcher_neg = matcher_con_verb_negation
  elif pos == "VERB" and feature == "con":
    _matcher_single = matcher_con_verb
    #_matcher_phrase = phrase_matcher_con_verb_phrase
    #_matcher_neg = matcher_pro_verb_negation  # negation using pro

  # Lists to hold matches based on their number of tokens
  _single_token_matches = []
  _multi_token_matches = []

  all_matches = []

  # Find lexicon matches
  _matches_single = _matcher_single(_doc)
  all_matches.append(_matches_single)
  """if _matcher_phrase:
    _matches_phrase = _matcher_phrase(_doc)
    all_matches.append(_matches_phrase)
  if _matcher_neg:
    _matches_phrase_neg = _matcher_neg(_doc)
    all_matches.append(_matches_phrase_neg)"""

  """if _matcher_pos_neg:
    _matches_pos_neg = _matcher_pos_neg(_doc)
    all_matches.append(_matches_pos_neg)
  if _matcher_neg_neg:
    _matches_neg_neg = _matcher_neg_neg(_doc)
    all_matches.append(_matches_neg_neg)
  if _matcher_cert_neg:
    _matches_cert_neg = _matcher_cert_neg(_doc)
    all_matches.append(_matches_cert_neg)
  if _matcher_doubt_neg:
    _matches_doubt_neg = _matcher_doubt_neg(_doc)
    all_matches.append(_matches_doubt_neg)
  if _matcher_emph_neg:
    _matches_emph_neg = _matcher_emph_neg(_doc)
    all_matches.append(_matches_emph_neg)"""

  # Extract indexes of tokens in the sentence, separating single and multi-token matches
  for _match in all_matches:
    for _, start, end in _match:
      num_tokens = end - start
      if num_tokens > 1:
        _multi_token_matches.append([start, end])
      else:  # Single-token
        _single_token_matches.append(start)
      # update counter
      if label == "support":
        feature_counter[f"support_{pos.lower()}_{feature}"] += 1
      elif label == "oppose":
        feature_counter[f"oppose_{pos.lower()}_{feature}"] += 1

  """2. Create a new doc and "token map" after removing punctuation"""

  token_map = []
  tokens = []

  # Create lists of indexes and tokens in the sentence skipping punctuation
  for token in _doc:
    undesirable_puncts = ["!", ".", "?", ",", ";", ":", "-", "–", "—"]
    if token.text in undesirable_puncts:
      continue
    token_map.append(token.i)  # Collect indexes of tokens in map
    tokens.append(token.text)  # Collect tokens

  # Create a new doc with the cleaned tokens
  tokenized_doc = nlp(" ".join(tokens))

  """3. Score tokens based on lexicon"""

  # Initialize list to hold extracted values (0/1)
  extracted_features = []

  # Iterate sentence tokens and score them based on matches
  for idx, token_level1 in enumerate(token_map):
    _token = tokenized_doc[idx]
    # Check if index token is in a single-token match
    if token_level1 in _single_token_matches:
      print(f"- {_token} ({_token.i}) - {pos} {feature} (lexicon word)")
      extracted_features.append(1)
    elif _multi_token_matches:  # If sentence has a multi-token match
      for item2 in _multi_token_matches:
        # Check if index token is in a multi-token match
        if token_level1 in range(item2[0], item2[1]):
          print(f"- {_token} ({_token.i}) - {pos} {feature} (lexicon phrase)")
          extracted_features.append(1)
        else:
          extracted_features.append(0)
    else:
      extracted_features.append(0)

  """4. Score tokens based on patterns"""

  """if use_patterns:

    # Create a matcher using the patterns based on the POS and direction
    matcher = None
    if pos == "ADJ" and feature == "emphatic":
      matcher = matcher_emphatic_adj_patterns
    elif pos == "ADJ" and feature == "pro":
      matcher = matcher_pro_adj_patterns
    elif pos == "ADJ" and feature == "con":
      matcher = matcher_con_adj_patterns
    elif pos == "ADV" and feature == "emphatic":
      matcher = matcher_emphatic_adv_patterns
    elif pos == "ADV" and feature == "con":
      matcher = matcher_con_adv_patterns
    elif pos == "VERB" and feature == "certainty":
      matcher = matcher_certainty_verb_patterns
    elif pos == "VERB" and feature == "emphatic":
      matcher = matcher_emphatic_verb_patterns
    elif pos == "VERB" and feature == "pro":
      matcher = matcher_pro_verb_patterns
    elif pos == "VERB" and feature == "con":
      matcher = matcher_con_verb_patterns

    # Find pattern matches
    matches = matcher(tokenized_doc)
    # print(matches)

    # Filter out overlapping matches
    spans = [Span(tokenized_doc, start, end, label=match_id) for match_id, start, end in matches]

    # Filter overlapping matches
    filtered_spans = filter_spans(spans)

    # Collect the indexes of the tokens within each match
    token_indexes_in_spans = []
    for span in filtered_spans:
      indexes = [token.i for token in span]
      token_indexes_in_spans.append(indexes)
      # update counter
      if label == "support":
        feature_counter[f"support_{pos.lower()}_{feature}_pattern"] += 1
      elif label == "oppose":
        feature_counter[f"oppose_{pos.lower()}_{feature}_pattern"] += 1

    # Replace scores "0" with "1" based on the pattern matches
    for idx, item in enumerate(extracted_features):
      for item2 in token_indexes_in_spans:
        if idx in item2:
          _token = tokenized_doc[idx]
          print(f"- {_token} ({_token.i}) - {pos} {feature} (pattern)")
          extracted_features[idx] = 1"""

  return extracted_features


"""-------------------Stance Features Extraction Processing-------------------"""

counter = 0

datapoint_id = 1274  # 1276

# iterate over the dataset and filter out the datapoint with the specified ID
datapoint = next((item for item in dataset if item["id"] == datapoint_id), None)

# Calculate total number of datapoints
# total_datapoints = len(dataset_train) + len(dataset_test)

for _idx, d in enumerate([dataset_train, dataset_test]):

  output_file = None
  if _idx == 0:
    output_file = output_file_training
  elif _idx == 1:
    output_file = output_file_test

  for datapoint in d:
    counter += 1
    total_features = 0

    print(f"Processing datapoint {datapoint['id']} ({counter}/{len(dataset)})")

    print(datapoint["text"])

    _doc = nlp(datapoint["text"])
    _label = datapoint["label"]

    # Adjective features
    _positive_adj = extract_stance_features(_doc, "ADJ", "positive", _label)
    _negative_adj = extract_stance_features(_doc, "ADJ", "negative", _label)
    _certainty_adj = extract_stance_features(_doc, "ADJ", "certainty", _label)
    _doubt_adj = extract_stance_features(_doc, "ADJ", "doubt", _label)
    _emphatic_adj = extract_stance_features(_doc, "ADJ", "emphatic", _label)
    _hedge_adj = extract_stance_features(_doc, "ADJ", "hedge", _label)
    _pro_adj = extract_stance_features(_doc, "ADJ", "pro", _label)
    _con_adj = extract_stance_features(_doc, "ADJ", "con", _label)

    # Adverb features
    _positive_adv = extract_stance_features(_doc, "ADV", "positive", _label)
    _negative_adv = extract_stance_features(_doc, "ADV", "negative", _label)
    _certainty_adv = extract_stance_features(_doc, "ADV", "certainty", _label)
    _doubt_adv = extract_stance_features(_doc, "ADV", "doubt", _label)
    _emphatic_adv = extract_stance_features(_doc, "ADV", "emphatic", _label)
    _hedge_adv = extract_stance_features(_doc, "ADV", "hedge", _label)
    _pro_adv = extract_stance_features(_doc, "ADV", "pro", _label)
    _con_adv = extract_stance_features(_doc, "ADV", "con", _label)

    # Verb features
    _positive_verb = extract_stance_features(_doc, "VERB", "positive", _label)
    _negative_verb = extract_stance_features(_doc, "VERB", "negative", _label)
    _certainty_verb = extract_stance_features(_doc, "VERB", "certainty", _label)
    _doubt_verb = extract_stance_features(_doc, "VERB", "doubt", _label)
    _emphatic_verb = extract_stance_features(_doc, "VERB", "emphatic", _label)
    _hedge_verb = extract_stance_features(_doc, "VERB", "hedge", _label)
    _pro_verb = extract_stance_features(_doc, "VERB", "pro", _label)
    _con_verb = extract_stance_features(_doc, "VERB", "con", _label)

    # Modality features
    _predictive_modal = extract_stance_modality(_doc, "predictive", _label)
    _possibility_modal = extract_stance_modality(_doc, "possibility", _label)
    _necessity_modal = extract_stance_modality(_doc, "necessity", _label)

    # Query the semantic frames API
    present_frames_response = api_get_frames(datapoint["text"], "localhost", "5001", "all")
    # Encode the frames with one-hot encoding
    # print(present_frames_response)
    _encoded_frames = encode_sentence_frames(present_frames_response)

    metadata = eval(datapoint["metadata"])
    # _encoded_frames = encode_sentence_frames(metadata["semantic_frames"])
    print(f"- frames: {metadata['semantic_frames']}")

    # datapoint["metadata"]["text"] = text

    row = {
      "id": datapoint["id"],
      "text": datapoint["text"],
      "label": datapoint["label"],

      # Adjective features
      "positive_adj": _positive_adj,
      "negative_adj": _negative_adj,
      "certainty_adj": _certainty_adj,
      "doubt_adj": _doubt_adj,
      "emphatic_adj": _emphatic_adj,
      "hedge_adj": _hedge_adj,
      "pro_adj": _pro_adj,
      "con_adj": _con_adj,

      # Adverb features
      "positive_adv": _positive_adv,
      "negative_adv": _negative_adv,
      "certainty_adv": _certainty_adv,
      "doubt_adv": _doubt_adv,
      "emphatic_adv": _emphatic_adv,
      "hedge_adv": _hedge_adv,
      "pro_adv": _pro_adv,
      "con_adv": _con_adv,

      # Verb features
      "positive_verb": _positive_verb,
      "negative_verb": _negative_verb,
      "certainty_verb": _certainty_verb,
      "doubt_verb": _doubt_verb,
      "emphatic_verb": _emphatic_verb,
      "hedge_verb": _hedge_verb,
      "pro_verb": _pro_verb,
      "con_verb": _con_verb,

      # Modal verbs
      "predictive_modal": _predictive_modal,
      "possibility_modal": _possibility_modal,
      "necessity_modal": _necessity_modal,

      # Semantic frames
      "semantic_frames": _encoded_frames
    }

    # Convert metadata to dictionary
    if "metadata" in datapoint:
      row["metadata"] = datapoint["metadata"]

    print()
    save_row_to_jsonl_file(row, output_file)

print(f"Total features extracted: {feature_counter}")
