import spacy
from afinn import Afinn
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
from spacy.util import filter_spans

from lib.stance_markers import (positive_affect_adj, positive_affect_adv, positive_affect_verb,
                                negative_affect_adv, negative_affect_adj, negative_affect_verb,
                                certainty_adj, doubt_adj, certainty_verbs, doubt_verbs, certainty_adv,
                                doubt_adv)

# Set spaCy to use GPU if available
if spacy.prefer_gpu():
    print("spaCy is using GPU!")
else:
    print("GPU not available, spaCy is using CPU instead.")

# Load transformer model
nlp = spacy.load("en_core_web_lg")

# Initialize Afinn sentiment analyzer
afinn = Afinn()


def measure_affect(_doc, terms, pos, polarity):
  """
  :param _doc: sentence
  :param terms: list of terms
  :param pos: POS of token to evaluate effect
  :param polarity: polarity direction to evaluate sentiment
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

  # Measure affect evaluating if terms are in predefined list
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
          # Evaluate polarization to add affect value
          if polarity == "certainty" and score > 0:
            measure_values[idx] = 1
          elif polarity == "doubt" and score < 0:
            measure_values[idx] = 1

  return measure_values


def measure_noun_affect(_doc, polarity):
  measure_values = []

  # Iterate over tokens in the sentence, skipping punctuation
  for token in _doc:
    if token.is_punct:
      continue

    # Get sentiment score of token
    score = afinn.score(token.text)
    if polarity == "certainty":
      if token.pos_ == "NOUN" and not token.is_stop and score > 0:
        measure_values.append(1)
      else:
        measure_values.append(0)
    elif polarity == "doubt":
      if token.pos_ == "NOUN" and not token.is_stop and score < 0:
        measure_values.append(1)
      else:
        measure_values.append(0)

  return measure_values


def measure_certain_verb(_doc, terms, pos):
  # Initialize measures
  measure_values = []

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
      measure_values.append(1)
    elif token.lemma_ not in terms and token.pos_ == pos and not token.is_stop and score > 0:
      measure_values.append(1)
    else:
      measure_values.append(0)

  """2. handle patterns"""
  matcher = Matcher(nlp.vocab)

  pattern1 = [
    {"LEMMA": {"IN": ["find", "show"]}},
    {"LOWER": "that"}
  ]

  # Pattern for "that (be) find/show"
  """pattern1 = [
    {"LOWER": "that"},
    {"LEMMA": "be", "OP": "?"},
    {"LEMMA": {"IN": ["find", "show"]}}
  ]"""

  pattern2 = [
    {"POS": "ADV", "OP": "?"},
    {"LOWER": "follows"},
    {"LOWER": "that"}
  ]

  # Pattern for the|this|that|it (NOUN) show (that)
  pattern3 = [
    {"LEMMA": {"IN": ["the", "this", "that", "it"]}},
    {"POS": "ADJ", "OP": "?"},
    {"POS": "NOUN", "OP": "?"},
    {"LEMMA": "show"},
    {"LOWER": "that", "OP": "?"}
  ]

  # Pattern for the|these|those NOUN show (that)
  pattern4 = [
    {"LEMMA": {"IN": ["the", "these", "those"]}},
    {"POS": "ADJ", "OP": "?"},
    {"POS": "NOUN"},
    {"LEMMA": "show"},
    {"LOWER": "that", "OP": "?"}
  ]

  # Add the pattern to the matcher
  matcher.add("FIND_SHOW_THAT", [pattern1])
  matcher.add("THAT_FIND_SHOW", [pattern1])
  matcher.add("FOLLOWS_THAT", [pattern2])
  matcher.add("THE_SHOW_THAT", [pattern3])
  matcher.add("THESE_SHOW_THAT", [pattern4])

  # Find matches
  matches = matcher(_doc)

  # Filter out overlapping matches, convert match ids and positions to spans
  spans = [Span(_doc, start, end, label=match_id) for match_id, start, end in matches]

  # Filter overlapping spans
  filtered_spans = filter_spans(spans)

  """3. Update scores based on the filtered spans"""

  # Count number of tokens in the filtered spans
  num_tokens = len([token for span in filtered_spans for token in span])

  # Remove "0" from scores based on the number of tokens and add "1" based on the number of tokens
  if num_tokens > 0:
    for i in range(len(measure_values)):
      if num_tokens == 0:
        break
      if measure_values[i] == 0:
        measure_values[i] = 1
        num_tokens -= 1

  return measure_values


def measure_doubt_verb(_doc, lexicon, pos, polarity):
  """
  :param _doc: The sentence
  :param lexicon: List of terms that indicate doubt
  :param pos: part of speech
  :param polarity: direction
  :return: List of 0/1 values containing doubt
  """

  """1. Match and separate single/multi-token terms using PhraseMatcher"""

  # Initialize Matcher
  phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

  # Put terms into the matcher
  patterns = [nlp(phrase) for phrase in lexicon]  # Convert the terms to patterns
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

  """2. Score individual tokens based on lexicon"""

  # Collect indexes of tokens in the sentence, skipping punctuation
  token_indexes = []
  for token in _doc:
    if token.is_punct:
      continue
    token_indexes.append(token.i)

  measure_values = []
  # Iterate over tokens in sentence
  for token_level1 in token_indexes:
    if token_level1 in single_token_matches:  # Add doubt value if token is a match
      measure_values.append(1)
    elif multi_token_matches:  # If sentence has a multi-token match
      for item2 in multi_token_matches:
        if token_level1 in range(item2[0], item2[1]):  # Check if token is part of multi-token match
          measure_values.append(1)
        else:
          measure_values.append(0)
    else:
      measure_values.append(0)

  """3. handle patterns"""

  matcher = Matcher(nlp.vocab)

  pattern1 = [
    {"LEMMA": {"IN": ["the", "this", "that", "it"]}},
    {"POS": "ADJ", "OP": "?"},
    {"POS": "NOUN", "OP": "?"},
    {"LOWER": "that", "OP": "?"},
    {"LEMMA": {"IN": ["intimate", "suggest"]}}
  ]

  pattern2 = [
    {"LEMMA": {"IN": ["the", "these", "those"]}},
    {"POS": "ADJ", "OP": "?"},
    {"POS": "NOUN"},
    {"LOWER": "that", "OP": "?"},
    {"LEMMA": {"IN": ["intimate", "suggest"]}}
  ]

  # Add the pattern to the matcher
  matcher.add("THE_THAT_INTIMATE", [pattern1])
  matcher.add("THESE_THAT_INTIMATE", [pattern2])

  """4. Update measure values based on pattern matches"""

  # Find matches
  matches = matcher(_doc)

  # Filter out overlapping matches
  spans = [Span(_doc, start, end, label=match_id) for match_id, start, end in matches]

  # Filter overlapping spans
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
        measure_values[idx] = 1

  """5. Update measure values based on POS and polarity direction"""

  for idx, item in enumerate(measure_values):
    if item == 0:
      token_ = _doc[token_indexes[idx]]
      if token_.pos_ == pos:
        score = afinn.score(token_.text)
        if polarity == "doubt" and score < 0:
          measure_values[idx] = 1
        elif polarity == "doubt" and score < 0:
          measure_values[idx] = 1

      # print(token_)

  return measure_values


# Positive affect
# text = "The fair is growing and clearly self-evident and fantastic for its delightful rides and attractions."
# text = "The fair was found that and clearly self-evident for its delightful rides and attractions."

# Negative affect
# text2 = "Abortion vehemently undermines the sanctity of life, irresponsibly allowing individuals to evade the natural consequences of their actions with alarming disregard."

# certainty adj
# text = "The inevitable success of the project was apparent, bolstered by the team's flawless execution and unwavering commitment."

# doubt adj
# text = "The outcome of the venture remains uncertain, clouded by dubious planning and a questionable market strategy."

# certainty verb
# text = "Demonstrating a keen ability to discern market trends, the analyst conclusively is established the investment's potential that was shown through comprehensive research."
# text = "The research shows that these experiments show the hypothesis is valid."
# text = "Given the undeniable evidence presented, it naturally follows that our hypothesis is not only valid but fundamentally irrefutable."

# doubt verb
text = "Given the current evidence suggests good news, one might assume, suspect, and even wonder if whether the proposed solution truly destroys the core issue, casting a shadow of doubt on its efficacy."

doc = nlp(text)

"""print(measure_affect(doc, positive_affect_adj, "ADJ", "pos"))
print(measure_affect(doc, positive_affect_adv, "ADV", "pos"))
print(measure_affect(doc, positive_affect_verb, "VERB", "pos"))
print(measure_noun_affect(doc, "pos"))"""

"""doc2 = nlp(text2)

print(measure_affect(doc2, negative_affect_adj, "ADJ", "neg"))
print(measure_affect(doc2, negative_affect_adv, "ADV", "neg"))
print(measure_affect(doc2, negative_affect_verb, "VERB", "neg"))
print(measure_noun_affect(doc2, "neg"))"""

# print(measure_affect(doc, certainty_adj, "ADJ", "pos"))
# print(measure_affect(doc, doubt_adj, "ADJ", "neg"))

# print(measure_certain_verb(doc, certainty_verbs, "VERB"))
print(measure_doubt_verb(doc, doubt_verbs, "VERB", "doubt"))

# print(measure_affect(doc, certainty_adv, "ADV", "pos"))
