import spacy
from afinn import Afinn
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
from spacy.util import filter_spans

from lib.stance_markers import (positive_affect_adj, positive_affect_adv, positive_affect_verb,
                                negative_affect_adv, negative_affect_adj, negative_affect_verb,
                                certainty_adj, doubt_adj, certainty_verbs, doubt_verbs)

# Set spaCy to use GPU if available
if spacy.prefer_gpu():
    print("spaCy is using GPU!")
else:
    print("GPU not available, spaCy is using CPU instead.")

# Load transformer model
nlp = spacy.load("en_core_web_lg")

# Initialize Afinn sentiment analyzer
afinn = Afinn()


def measure_affect(_doc, terms, pos, pole):
  """
  :param _doc: sentence
  :param terms: list of terms
  :param pos: POS of token to evaluate effect
  :param pole: pole to evaluate sentiment
  :return: list of 0/1 values containing affect
  """

  """1. Match and separate single/multi-token matches"""

  # Initialize Matcher
  phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

  # Put terms into the matcher
  patterns = [nlp(phrase) for phrase in terms]  # Convert the terms to patterns
  phrase_matcher.add("PHRASES", patterns)  # Add the patterns to the matcher

  # Lists to hold matches based on their length
  single_token_matches = []
  multi_token_matches = []

  matches = phrase_matcher(_doc)  # Find matches

  # Separate single token matches from multi-token matches
  for _, start, end in matches:
    num_tokens = end - start
    if num_tokens > 1:  # Multi-token
      multi_token_matches.append([start, end])  # Multi-token
    else:
      single_token_matches.append(start)  # Single-token

  """2. Measure affect token by token"""

  # Collect indexes of tokens in the sentence, skipping punctuation
  token_indexes = []
  for token in _doc:
    if token.is_punct:
      continue
    token_indexes.append(token.i)

  # Measure affect evaluating if terms are single or multi-token
  measure_values = []
  # Iterate over tokens in sentence
  for token_level1 in token_indexes:
    if token_level1 in single_token_matches:  # Add affect value if token is a match
      measure_values.append(1)
    elif multi_token_matches:  # If sentence has a multi-token match
      for item2 in multi_token_matches:
        if token_level1 in range(item2[0], item2[1]):  # Check if token is part of multi-token match
          measure_values.append(1)
        else:
          measure_values.append(0)
    else:
      measure_values.append(0)

    """3. Use AFINN to evaluate sentiment in not matched spans"""
    # Check sentiment of 0 score tokens
    for idx, item3 in enumerate(measure_values):
      if item3 == 0:

        if _doc[token_indexes[idx]].pos_ == pos:
          # Get sentiment score
          score = afinn.score(_doc[token_level1].text)

          if pole == "pos" and score > 0:
            measure_values[idx] = 1
          elif pole == "neg" and score < 0:
            measure_values[idx] = 1
  return measure_values


def measure_noun_affect(_doc, pole):
  measure_values = []

  for token in _doc:
    if token.is_punct:
      continue

    # Get sentiment score
    score = afinn.score(token.text)
    if pole == "pos":
      if token.pos_ == "NOUN" and not token.is_stop and score > 0:
        measure_values.append(1)
        # print(token.text, score)
      else:
        measure_values.append(0)
    elif pole == "neg":
      if token.pos_ == "NOUN" and not token.is_stop and score < 0:
        measure_values.append(1)
      else:
        measure_values.append(0)

  return measure_values


def measure_certain_verb(_doc, terms, pos):
  # Initialize measures
  measures = []

  """1. process single token"""
  # Process each token in the _doc
  for token in _doc:
    if token.is_punct:
      continue

    # Get sentiment score
    score = afinn.score(token.text)

    # Check if the token is in the single token list
    if token.lemma_ in terms and token.pos_ == pos:
      # print(token.lemma_)
      measures.append(1)
    elif token.lemma_ not in terms and token.pos_ == pos and not token.is_stop and score > 0:
      measures.append(1)
    else:
      measures.append(0)

  """2. handle patterns"""
  matcher = Matcher(nlp.vocab)

  # Pattern for "that (be) find/show"
  pattern1 = [
    {"LOWER": "that"},
    {"LEMMA": "be", "OP": "?"},
    {"LEMMA": {"IN": ["find", "show"]}}
  ]

  pattern2 = [
    {"POS": "ADV", "OP": "?"},
    {"LOWER": "follows"},
    {"LOWER": "that"}
  ]

  # Pattern for the|this|that|it (NOUN) show (that)
  pattern3 = [
    {"LEMMA": {"IN": ["the", "this", "that", "it"]}},
    {"POS": "NOUN", "OP": "?"},
    {"LEMMA": "show"},
    {"LOWER": "that", "OP": "?"}
  ]

  # Pattern for the|these|those NOUN show (that)
  pattern4 = [
    {"LEMMA": {"IN": ["the", "these", "those"]}},
    {"POS": "NOUN"},
    {"LEMMA": "show"},
    {"LOWER": "that", "OP": "?"}
  ]

  # Add the pattern to the matcher
  matcher.add("THAT_FIND_SHOW", [pattern1])
  matcher.add("FOLLOWS_THAT", [pattern2])
  matcher.add("THE_SHOW_THAT", [pattern3])
  matcher.add("THESE_SHOW_THAT", [pattern4])

  # Find matches
  matches = matcher(_doc)

  # Filter out overlapping matches
  # Convert match ids and positions to spans
  # spans = [spacy.tokens.Span(_doc, start, end, label=match_id) for match_id, start, end in matches]
  spans = [Span(_doc, start, end, label=match_id) for match_id, start, end in matches]

  # Filter overlapping spans
  filtered_spans = filter_spans(spans)

  """3. Update scores based on the filtered spans"""

  # Count number of tokens in the filtered spans
  num_tokens = len([token for span in filtered_spans for token in span])

  # Remove "0" from scores based on the number of tokens and add "1" based on the number of tokens
  if num_tokens > 0:
    for i in range(len(measures)):
      if num_tokens == 0:
        break
      if measures[i] == 0:
        measures[i] = 1
        num_tokens -= 1

  # print(scores)
  return measures


def measure_doubt_verb(_doc, terms, pos):
  # Initialize scores
  scores = []

  """1. process single token"""
  # Process each token in the _doc
  for token in _doc:
    if token.is_punct:
      continue

    # Get sentiment score
    score = afinn.score(token.text)

    # Check if the token is in the single token list
    if token.lemma_ in terms and token.pos_ == pos:
      # print(token.lemma_)
      scores.append(1)
    elif token.lemma_ not in terms and token.pos_ == pos and not token.is_stop and score < 0:
      scores.append(1)
    else:
      scores.append(0)

  """2. handle patterns"""
  matcher = Matcher(nlp.vocab)

  pattern1 = [
    {"LEMMA": {"IN": ["the", "this", "that", "it"]}},
    {"POS": "NOUN", "OP": "?"},
    {"LOWER": "that", "OP": "?"},
    {"LEMMA": {"IN": ["intimate", "suggest"]}}
  ]

  pattern2 = [
    {"LEMMA": {"IN": ["the", "these", "those"]}},
    {"POS": "NOUN"},
    {"LOWER": "that", "OP": "?"},
    {"LEMMA": {"IN": ["intimate", "suggest"]}}
  ]

  # Add the pattern to the matcher
  matcher.add("THE_THAT_INTIMATE", [pattern1])
  matcher.add("THESE_THAT_INTIMATE", [pattern2])

  # Find matches
  matches = matcher(_doc)

  # Filter out overlapping matches
  spans = [Span(_doc, start, end, label=match_id) for match_id, start, end in matches]

  # Filter overlapping spans
  filtered_spans = filter_spans(spans)

  """3. Update scores based on the filtered spans"""

  # Count number of tokens in the filtered spans
  num_tokens = len([token for span in filtered_spans for token in span])

  # Remove "0" from scores based on the number of tokens and add "1" based on the number of tokens
  if num_tokens > 0:
    for i in range(len(scores)):
      if num_tokens == 0:
        break
      if scores[i] == 0:
        scores[i] = 1
        num_tokens -= 1

  # print(scores)
  return scores
