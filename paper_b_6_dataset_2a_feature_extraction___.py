import contractions
import spacy
from afinn import Afinn
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
from spacy.util import filter_spans

from lib.stance_markers import (positive_affect_adj, positive_affect_adv, positive_affect_verb,
                                negative_affect_adv, negative_affect_adj, negative_affect_verb,
                                certainty_adj, certainty_adv, certainty_adv_to_be, certainty_verbs,
                                doubt_adj, doubt_adv, doubt_verbs,
                                hedges, emphatics)

# Set spaCy to use GPU if available
if spacy.prefer_gpu():
  print("spaCy is using GPU!")
else:
  print("GPU not available, spaCy is using CPU instead.")

# Load transformer model
nlp = spacy.load("en_core_web_trf")

# Initialize Afinn sentiment analyzer
afinn = Afinn()

"""Define patterns for certainty"""


def create_certain_verb_matcher():
  matcher = Matcher(nlp.vocab)

  # Pattern: lemma of [find, show] that
  pattern1 = [
    {"LEMMA": {"IN": ["find", "show"]}},
    {"LOWER": "that"}
  ]

  # Pattern: the/this/that/it (ADJ) (NOUN) lemma of shows (that)
  pattern2 = [
    {"LOWER": {"IN": ["the", "this", "that", "it"]}},
    {"POS": "ADJ", "OP": "?"},
    {"POS": "NOUN", "OP": "?"},
    {"LEMMA": "show"},
    {"LOWER": "that", "OP": "?"}
  ]

  # Pattern: the/these/those (ADJ) NOUN lemma of show (that)
  pattern3 = [
    {"LOWER": {"IN": ["the", "these", "those"]}},
    {"POS": "ADJ", "OP": "?"},
    {"POS": "NOUN"},
    {"LEMMA": "show"},
    {"LOWER": "that", "OP": "?"}
  ]

  # Pattern: lemma of follows that
  pattern4 = [
    {"POS": "ADV", "OP": "?"},
    {"LEMMA": "follow"},
    {"LOWER": "that"}
  ]

  # Add patterns to the matcher
  matcher.add("FIND_SHOW_THAT", [pattern1])
  matcher.add("THE_SHOW_THAT", [pattern2])
  matcher.add("THESE_SHOW_THAT", [pattern3])
  matcher.add("FOLLOWS_THAT", [pattern4])

  return matcher


def create_doubt_verb_matcher():
  matcher = Matcher(nlp.vocab)

  # Pattern: the/this/that/it (ADJ) (NOUN) (that) lemma of [intimates, suggests]
  pattern1 = [
    {"LOWER": {"IN": ["the", "this", "that", "it"]}},
    {"POS": "ADJ", "OP": "?"},
    {"POS": "NOUN", "OP": "?"},
    {"LEMMA": {"IN": ["intimate", "suggest"]}},
    {"LOWER": "that", "OP": "?"}
  ]

  # Pattern: the/these/those (ADJ) NOUN (that) lemma of [intimate, suggest]
  pattern2 = [
    {"LOWER": {"IN": ["the", "these", "those"]}},
    {"POS": "ADJ", "OP": "?"},
    {"POS": "NOUN"},
    {"LEMMA": {"IN": ["intimate", "suggest"]}},
    {"LOWER": "that", "OP": "?"},
  ]

  # Add patterns to the matcher
  matcher.add("THE_THAT_INTIMATE", [pattern1])
  matcher.add("THESE_THAT_INTIMATE", [pattern2])

  return matcher


def create_emphatic_patterns():
  matcher = Matcher(nlp.vocab)

  # Pattern for "do" followed by any verb (VERB)
  pattern_do_verb = [
    {'LEMMA': 'do'},
    {'POS': 'VERB'}]

  # Pattern for "real" followed by any adjective (ADJ)
  pattern_real_adj = [
    {'LOWER': 'real'},
    {'POS': 'ADJ'}]

  # Pattern for "so" followed by any adjective (ADJ)
  pattern_so_adj = [
    {'LOWER': 'so'},
    {'POS': 'ADJ'}]

  # Pattern for "so" followed by any adverb (ADV)
  pattern_so_adv = [
    {'LOWER': 'so'},
    {'POS': 'ADV'}]

  # Add patterns to the matcher
  matcher.add('DO_VERB', [pattern_do_verb])
  matcher.add('REAL_ADJ', [pattern_real_adj])
  matcher.add('SO_ADJ', [pattern_so_adj])
  matcher.add('SO_ADV', [pattern_so_adv])

  return matcher


def create_certain_adv_pattern():
  matcher = Matcher(nlp.vocab)

  # Pattern: to be lemma of [certain, clear, realistic, sure]
  pattern1 = [
    {"LEMMA": "be"},
    {"LEMMA": {"IN": certainty_adv_to_be}}
  ]

  # Add the pattern to the matcher
  matcher.add("BE_CERTAIN", [pattern1])

  return matcher


def extract_stance_features(_doc, lexicon, pos, direction, use_patterns=False):
  """
  This function extracts stance features from a sentence based on
  1. lexicon of terms that indicate stance
  2. linguistic patterns that indicate stance
  3. sentiment score that indicates stance

  :param _doc: The sentence
  :param lexicon: List of terms that indicate stance
  :param pos: part of speech
  :param direction: stance inclination 1. Affect (certain/doubt) or 2. Epistemic Modality (certainty/doubt)
  :param use_patterns: patterns to match (optional)
  :return: List of 0/1 values
  """

  """1. Create matcher with lexicon and find/separate matches"""

  # Initialize PhraseMatcher
  phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")

  # Put lexicon into the matcher
  patterns = [nlp(phrase) for phrase in lexicon]
  phrase_matcher.add("PHRASES", patterns)

  # Lists to hold matches based on their number of tokens
  single_token_matches = []
  multi_token_matches = []

  # Find lexicon matches
  matches = phrase_matcher(_doc)

  # Extract indexes of tokens in the sentence, separating single and multi-token matches
  for _, start, end in matches:
    num_tokens = end - start
    if num_tokens > 1:  # Multi-token
      multi_token_matches.append([start, end])
    else:  # Single-token
      single_token_matches.append(start)

  """2. Create a new doc and "token index" after removing punctuation"""

  token_idxs = []
  tokens = []

  # Create lists of indexes and tokens in the sentence skipping punctuation
  for token in _doc:
    undesirable_puncts = ["!", ".", "?", ",", ";", ":", "-", "–", "—"]
    if token.text in undesirable_puncts:
      continue
    token_idxs.append(token.i)  # Collect indexes of tokens
    tokens.append(token.text)  # Collect tokens

  # Create a new doc with the cleaned tokens
  tokenized_doc = nlp(' '.join(tokens))

  print(f"{len(token_idxs)}: {token_idxs}")
  print(tokens)
  print(f"{len(tokenized_doc)}: {tokenized_doc}")

  """3. Score tokens based on lexicon"""

  # Initialize list to hold extracted values (0/1)
  extracted_features = []

  # Iterate sentence tokens and score them based on matches
  for idx, token_level1 in enumerate(token_idxs):
    _token = _doc[token_idxs[idx]]
    # Check if index token is in a single-token match
    if token_level1 in single_token_matches and _token.pos_ == pos:
      print(f"- {_token} ({_token.i}) - lexicon single-token")
      extracted_features.append(1)
    elif multi_token_matches:  # If sentence has a multi-token match
      for item2 in multi_token_matches:
        # Check if index token is in a multi-token match
        if token_level1 in range(item2[0], item2[1]):
          print(f"- {_token} ({_token.i}) - lexicon multi-token")
          extracted_features.append(1)
        else:
          extracted_features.append(0)
    else:
      extracted_features.append(0)

  """4. Score tokens based on patterns"""

  if use_patterns:

    # Create a matcher using the patterns based on the POS and direction
    matcher = None
    if pos == "VERB" and direction == "certain":
      matcher = create_certain_verb_matcher()
    elif pos == "VERB" and direction == "doubt":
      matcher = create_doubt_verb_matcher()
    elif pos == "ADV" and direction == "certain":
      matcher = create_certain_adv_pattern()

    # Find pattern matches
    matches = matcher(tokenized_doc)

    # Filter out overlapping matches
    spans = [Span(tokenized_doc, start, end, label=match_id) for match_id, start, end in matches]

    # Filter overlapping matches
    filtered_spans = filter_spans(spans)

    # print(f"-->{filtered_spans}")

    # Collect the indexes of the tokens within each match
    token_indexes_in_spans = []
    for span in filtered_spans:
      indexes = [token.i for token in span]
      token_indexes_in_spans.append(indexes)

    # Replace scores "0" with "1" based on the pattern matches
    for idx, item in enumerate(extracted_features):
      for item2 in token_indexes_in_spans:
        if idx in item2:
          _token = tokenized_doc[idx]
          print(f"- {_token} ({_token.i}) - pattern")
          extracted_features[idx] = 1

  """5. Score tokens using sentiment score using POS and direction"""

  # Replace scores "0" with "1" based on the POS and direction
  for idx, item in enumerate(extracted_features):
    # Observe tokens not scored yet
    if item == 0:
      _token = tokenized_doc[idx]
      if _token.pos_ == pos:
        sentiment_score = afinn.score(_token.text)
        if direction == "doubt" and sentiment_score < 0:
          print(f"- {_token} ({idx}) - pos/polarity")
          extracted_features[idx] = 1
        elif direction == "certain" and sentiment_score > 0:
          print(f"- {_token} ({idx}) - pos/polarity")
          extracted_features[idx] = 1

  return extracted_features


def extract_noun_affect(_doc, direction):
  extracted_features = []

  token_idxs = []
  tokens = []

  # Create lists of indexes and tokens in the sentence skipping punctuation
  for token in _doc:
    undesirable_puncts = ["!", ".", "?", ",", ";", ":", "-", "–", "—"]
    if token.text in undesirable_puncts:
      continue
    token_idxs.append(token.i)  # Collect indexes of tokens
    tokens.append(token.text)  # Collect tokens

  # Create a new doc with the cleaned tokens
  tokenized_doc = nlp(' '.join(tokens))

  print(f"{len(token_idxs)}: {token_idxs}")
  print(tokens)
  print(f"{len(tokenized_doc)}: {tokenized_doc}")

  # Iterate over tokens in the sentence, skipping punctuation
  for idx, token in enumerate(tokenized_doc):
    if token.is_punct:
      continue

    # Get sentiment score of token
    sentiment_score = afinn.score(token.text)
    if direction == "pos":
      if token.pos_ == "NOUN" and not token.is_stop and sentiment_score > 0:
        print(f"- {token.text} ({token.i}) - noun sentiment score")
        extracted_features.append(1)
      else:
        extracted_features.append(0)
    elif direction == "neg":
      if token.pos_ == "NOUN" and not token.is_stop and sentiment_score < 0:
        print(f"- {token.text} ({token.i}) - noun sentiment score")
        extracted_features.append(1)
      else:
        extracted_features.append(0)

  return extracted_features


def extract_modal_qualifiers(_doc, direction):
  # Create a matcher
  phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

  if direction == "hedges":
    patterns = [nlp.make_doc(term) for term in hedges]
    # Add the pattern to the matcher
    phrase_matcher.add("HEDGES", patterns)
  elif direction == "emphatics":
    patterns = [nlp.make_doc(term) for term in emphatics]
    # Add the pattern to the matcher
    phrase_matcher.add("EMPHATICS", patterns)

  # Remove punctuation
  tokenized_text = [token.text for token in _doc if not token.is_punct]
  tokenized_doc = nlp(' '.join(tokenized_text))

  print(tokenized_doc)

  measure_values = [0] * len(tokenized_doc)

  # Find matches
  phrase_matches = phrase_matcher(tokenized_doc)

  # Remove overlapping spans
  spans = []
  for match_id, start, end in phrase_matches:
    spans.append(tokenized_doc[start:end])

  filtered_spans = filter_spans(spans)

  # print(filtered_spans)

  # Collect the indexes of the tokens within each span
  token_indexes_in_spans = []
  for span in filtered_spans:
    indexes = [token.i for token in span]
    token_indexes_in_spans.append(indexes)

  # print(token_indexes_in_spans)

  token_indexes = []
  # Collect the indexes of the tokens in the sentence
  for token in tokenized_doc:
    token_indexes.append(token.i)

  # Replace "0" with "1" based on the number of tokens in the filtered spans
  for idx, item in enumerate(measure_values):
    for item2 in token_indexes_in_spans:
      if idx in item2:
        token_ = tokenized_doc[token_indexes[idx]]
        print(f"- {token_} ({token_.i})")
        measure_values[idx] = 1

  if direction == "emphatics":

    matcher = create_emphatic_patterns()

    # Find matches
    matches = matcher(tokenized_doc)

    # Remove overlapping spans
    spans = []
    for match_id, start, end in matches:
      spans.append(tokenized_doc[start:end])

    filtered_spans = filter_spans(spans)

    print(filtered_spans)

    # Collect the indexes of the tokens within each span
    token_indexes_in_spans = []
    for span in filtered_spans:
      indexes = [token.i for token in span]
      token_indexes_in_spans.append(indexes)

    # Replace "0" with "1" based on the number of tokens in the filtered spans
    for idx, item in enumerate(measure_values):
      for item2 in token_indexes_in_spans:
        if idx in item2:
          token_ = tokenized_doc[token_indexes[idx]]
          print(f"- {token_} ({token_.i})")
          measure_values[idx] = 1

  return measure_values


# Positive affect
# text = "The fair is growing and clearly self-evident and fantastic for its delightful rides and attractions."
# Positive affect with certainty verb
# text = "The fair was without question found that and clearly self-evident for its delightful rides and attractions."

# Negative affect
# text = "Abortion vehemently alarmed and the sanctity of life, irresponsibly allowing individuals to evade the natural consequences of their actions with alarming disregard."

# certainty adj, adv
# text = "It is certain and well-known, the inevitable success of the project was apparent, bolstered by the team's flawless without question execution and unwavering commitment."
# text = "It is certain, and well-known"

# certainty verb
# text = "Demonstrating a keen ability to discern market trends, the analyst conclusively is established the investment's potential that was shown through comprehensive research."
# text = "The research shows that these experiments show the hypothesis is valid."
# text = "Given the undeniable evidence presented, it naturally follows that our hypothesis is not only valid but fundamentally irrefutable."

# certainty adv
# text = "It is clear that undoubtedly, she handled the situation decisively and, indeed, demonstrated her capability without question."

# doubt adj
# text = "The outcome of the venture remains uncertain, clouded by dubious planning and a questionable market strategy."

# doubt verb
# text = "Given the current evidence suggests good news, one might assume, suspect, and even wonder if whether the proposed solution truly destroys the core issue, casting a shadow of doubt on its efficacy."
text = "I'm sure the improved policy suggests that the proposed solution truly destroys the core issue, casting a shadow of doubt on its efficacy."

# doubt adv
# text = "Ostensibly, the project was completed on time, but technically and sadly, some essential components were still under development, and reportedly, the team faced significant delays."

# hedges
# text = "She was kind of tired after the hike, maybe because it was more or less challenging than she expected."

# emphatics
# text = "I do believe she really appreciated the gift, just knowing it came from him meant a lot, and so importantly, she felt the most joyful she had ever been, even as the sky turned real dark."

# Remove contractions
text = contractions.fix(text)
doc = nlp(text)

"""------------------------------"""

# Positive affect
"""print(extract_stance_features(doc, positive_affect_adj, "ADJ", "certain"))
print()
print(extract_stance_features(doc, positive_affect_adv, "ADV", "certain"))
print()
print(extract_stance_features(doc, positive_affect_verb, "VERB", "certain"))
print()
print(extract_noun_affect(doc, "pos"))
print()"""

# Negative affect
"""print(extract_stance_features(doc, negative_affect_adj, "ADJ", "doubt"))
print()
print(extract_stance_features(doc, negative_affect_adv, "ADV", "doubt"))
print()
print(extract_stance_features(doc, negative_affect_verb, "VERB", "doubt"))
print()
print(measure_noun_affect(doc, "neg"))"""

# Certainty
# print(extract_stance_features(doc, certainty_adj, "ADJ", "certain"))
# print(extract_stance_features(doc, certainty_verbs, "VERB", "certain", use_patterns=True))
# print(extract_stance_features(doc, certainty_adv, "ADV", "certain", use_patterns=True))

# Doubt
# print(extract_stance_features(doc, doubt_adj, "ADJ", "doubt"))
print(extract_stance_features(doc, doubt_verbs, "VERB", "doubt", use_patterns=True))
# print(extract_stance_features(doc, doubt_adv, "ADV", "doubt"))

# Hedges
# print(measure_epistemic_modality(doc, "hedges"))
# print(measure_epistemic_modality(doc, "emphatics"))
