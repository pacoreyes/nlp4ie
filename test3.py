import spacy
from afinn import Afinn
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
from spacy.util import filter_spans

from lib.stance_markers import (positive_affect_adj, positive_affect_adv, positive_affect_verb, negative_affect_adv,
                                negative_affect_adj, negative_affect_verb, certainty_adj, doubt_adj, certainty_verbs,
                                doubt_verbs)

# Load transformer model
nlp = spacy.load("en_core_web_lg")

# Initialize Afinn sentiment analyzer
afinn = Afinn()


def measure_affect(_doc, terms, pos, pole):
  measure = []
  sentence_indexes = []

  # Initialize Matcher
  phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

  patterns = [nlp(phrase) for phrase in terms]

  phrase_matcher.add("PHRASES", patterns)

  # Lists to hold matches based on their length
  single_token_matches = []
  multi_token_matches = []

  # Find matches
  matches = phrase_matcher(_doc)

  for _, start, end in matches:
    num_tokens = end - start  # Length of the match in tokens

    match_text = _doc[start:end].text  # Get the text
    if num_tokens > 1:
      multi_token_matches.append([start, end])  # Multi-token
    else:
      # Match spans a single token
      single_token_matches.append(start)  # Single-token

  """print(single_token_matches)
  print(multi_token_matches)"""

  # Process each token in the _doc
  for token in _doc:
    if token.is_punct:
      continue

    sentence_indexes.append(token.i)

  for item in sentence_indexes:

    if item in single_token_matches:
      measure.append(1)
    elif multi_token_matches:
      for item2 in multi_token_matches:
        if item in range(item2[0], item2[1]):
          measure.append(1)
        else:
          measure.append(0)
    else:
      measure.append(0)

    # Check sentiment of 0 score tokens
    for idx, item3 in enumerate(measure):
      if item3 == 0:

        if _doc[sentence_indexes[idx]].pos_ == pos:
          # Get sentiment score
          score = afinn.score(_doc[item].text)

          if pole == "pos" and score > 0:
            measure[idx] = 1
          elif pole == "neg" and score < 0:
            measure[idx] = 1

  return measure


def measure_affect2(_doc, terms, pos, pole):
  # Initialize Matcher
  phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

  # Initialize lists
  single_token = []
  multi_token = []
  scores = []

  """1. filter single and multi token terms"""
  # Process each string to categorize it
  for string in terms:
    temp_doc = nlp(string)
    # Count the number of tokens in the processed string
    num_tokens = len([token for token in temp_doc if not token.is_punct])
    # Categorize based on the number of tokens
    if num_tokens > 1:
      multi_token.append(string)
    else:
      single_token.append(string)

  """2. remove multi-token terms in the doc"""
  # Convert the phrases into Doc objects and add them as patterns to the matcher
  patterns = [nlp(phrase) for phrase in multi_token]
  phrase_matcher.add("PHRASES", patterns)

  # Find matches
  matches = phrase_matcher(_doc)  # Here are stored the multi-token matches
  # print(matches)

  # To exclude matches, we'll create a new list of tokens not part of any matches
  tokens_to_keep = []
  matches = sorted(matches, key=lambda x: x[1])  # Sort matches by start index
  match_index = 0
  for i, token in enumerate(_doc):
    if token.is_punct:
      continue

    if match_index < len(matches) and matches[match_index][1] <= i < matches[match_index][2]:
      # Skip tokens that are part of a match
      if i == matches[match_index][2] - 1:
        # Move to the next match
        match_index += 1
      continue
    tokens_to_keep.append(token)

  # Reconstruct the document from remaining tokens
  cleaned_doc = " ".join(token.text for token in tokens_to_keep if not token.is_punct)
  # print(cleaned_doc)
  cleaned_doc = nlp(cleaned_doc)

  """3. process single token terms in the cleaned doc"""
  # Process each token in the _doc
  for token in cleaned_doc:
    if token.is_punct:
      continue

    # Get sentiment score
    score = afinn.score(token.text)
    # Check if the token is in the single token list
    if token.lemma_ in single_token and token.pos_ == pos and not token.is_stop:
      scores.append(1)
    elif token.lemma_ not in single_token and token.pos_ == pos and not token.is_stop:
      if pole == "pos" and score > 0:
        scores.append(1)
      elif pole == "neg" and score < 0:
        scores.append(1)
    else:
      scores.append(0)

  """4. add multi-token terms to the scores"""
  if multi_token:
    scores.extend([1] * len(matches))  # Add 1 for each match

  return scores


def measure_noun_affect(_doc, pole):
  # Get the polarity words
  scores = []
  for token in _doc:
    if token.is_punct:
      continue

    # Get sentiment score
    score = afinn.score(token.text)
    if pole == "pos":
      if token.pos_ == "NOUN" and not token.is_stop and score > 0:
        scores.append(1)
        # print(token.text, score)
      else:
        scores.append(0)
    elif pole == "neg":
      if token.pos_ == "NOUN" and not token.is_stop and score < 0:
        scores.append(1)
      else:
        scores.append(0)

  return scores


def measure_certain_verb(_doc, terms, pos):

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
    elif token.lemma_ not in terms and token.pos_ == pos and not token.is_stop and score > 0:
      scores.append(1)
    else:
      scores.append(0)

  """2. handle patterns"""
  matcher = Matcher(nlp.vocab)

  # Pattern for "that (be) find/show"
  pattern1 = [
    {"LOWER": "that"},
    {"LEMMA": "be", "OP": "?"},
    {"LEMMA": {"IN": ["find", "show"]}}
  ]

  # Pattern for list of verbs
  """pattern2 = [
    {"LEMMA": "be"},
    {"LOWER": {"IN": ['ascertained', 'calculated', 'concluded', 'deduced', 'demonstrated', 'determined',
                      'discerned', 'established', 'known', 'noted', 'perceived', 'proved', 'realized']}}
  ]"""

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
  # matcher.add("CERTAINTY_VERBS", [pattern2])
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
    for i in range(len(scores)):
      if num_tokens == 0:
        break
      if scores[i] == 0:
        scores[i] = 1
        num_tokens -= 1

  # print(scores)
  return scores


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
text = "Given the current evidence, one might assume, suspect, and even wonder whether the proposed solution truly addresses the core issue, casting a shadow of doubt on its efficacy."

doc = nlp(text)

"""print(measure_affect(doc, positive_affect_adj, "ADJ", "pos"))
print(measure_affect(doc, positive_affect_adv, "ADV", "pos"))
print(measure_affect(doc, positive_affect_verb, "VERB", "pos"))
print(measure_noun_affect(doc, "pos"))

doc2 = nlp(text2)

print(measure_affect(doc2, negative_affect_adj, "ADJ", "neg"))
print(measure_affect(doc2, negative_affect_adv, "ADV", "neg"))
print(measure_affect(doc2, negative_affect_verb, "VERB", "neg"))
print(measure_noun_affect(doc2, "neg"))"""

# print(measure_affect(doc, certainty_adj, "ADJ", "pos"))
# print(measure_affect(doc, doubt_adj, "ADJ", "neg"))

# print(measure_certain_verb(doc, certainty_verbs, "VERB"))
print(measure_doubt_verb(doc, doubt_verbs, "VERB"))


"""
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
text = "Given the current evidence, one might assume, suspect, and even wonder whether the proposed solution truly addresses the core issue, casting a shadow of doubt on its efficacy."

doc = nlp(text)

print(measure_affect(doc, positive_affect_adj, "ADJ", "pos"))
print(measure_affect(doc, positive_affect_adv, "ADV", "pos"))
print(measure_affect(doc, positive_affect_verb, "VERB", "pos"))
print(measure_noun_affect(doc, "pos"))

doc2 = nlp(text2)

print(measure_affect(doc2, negative_affect_adj, "ADJ", "neg"))
print(measure_affect(doc2, negative_affect_adv, "ADV", "neg"))
print(measure_affect(doc2, negative_affect_verb, "VERB", "neg"))
print(measure_noun_affect(doc2, "neg"))

# print(measure_affect(doc, certainty_adj, "ADJ", "pos"))
# print(measure_affect(doc, doubt_adj, "ADJ", "neg"))

# print(measure_certain_verb(doc, certainty_verbs, "VERB"))
print(measure_doubt_verb(doc, doubt_verbs, "VERB"))
"""
