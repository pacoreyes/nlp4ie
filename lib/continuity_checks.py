"""
This script analyzes the linguistic features that link or not pairs of sentences to classify
them as "continue" or "not continue" pairs.

The rules are based on the paper "A Corpus of Sentence-level Discourse Parsing in English News"
https://aclanthology.org/N03-1030.pdf
"""

import spacy
# from scipy.spatial.distance import cosine
from spacy.matcher import PhraseMatcher

from lib.lexicons import transition_markers

# from pprint import pprint

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_lg")

# Load small model for morphological analysis
nlp_sm_coref = spacy.load("en_core_web_sm")

# Load coreference resolution model
nlp_coref = spacy.load("en_coreference_web_trf")

nlp_coref.replace_listeners("transformer", "coref", ["model.tok2vec"])
nlp_coref.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])

nlp_sm_coref.add_pipe("coref", source=nlp_coref)
nlp_sm_coref.add_pipe("span_resolver", source=nlp_coref)

# Initialize PhraseMatchers to match transition markers
leading_continuity_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
leading_shift_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
flexible_continuity_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
flexible_shift_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Retrieve transition markers from lexicons.py
leading_markers_continuity = transition_markers["leading_markers"]["topic_continuity"]
leading_markers_shift = transition_markers["leading_markers"]["topic_shift"]
flexible_markers_continuity = transition_markers["flexible_markers"]["topic_continuity"]
flexible_markers_shift = transition_markers["flexible_markers"]["topic_shift"]

# Add transition markers to PhraseMatchers
patterns = [nlp.make_doc(text) for text in leading_markers_continuity]
leading_continuity_matcher.add("leading_continuity", patterns)
patterns = [nlp.make_doc(text) for text in leading_markers_shift]
leading_shift_matcher.add("leading_shift", patterns)
patterns = [nlp.make_doc(text) for text in flexible_markers_continuity]
flexible_continuity_matcher.add("flexible_continuity", patterns)
patterns = [nlp.make_doc(text) for text in flexible_markers_shift]
flexible_shift_matcher.add("flexible_shift", patterns)

ignore_units = {"be", "ROOT", "other"}


# Check if the sentences have lexical continuity
def check_lexical_continuity(_sent1, _sent2):
  """
  This function checks if two sentences have lexical continuity, which means that they have at least one common
  noun, proper noun, verb, adjective or adverb.
  :param _sent1:
  :param _sent2:
  :return: dict
  """

  def extract_key_syntactic_units(doc):
    # return [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ", "ADV"]]
    syntactic_units = [token.lemma_ for token in doc if token.pos_ in
                       ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"]]

    # Remove tokens that are in the ignore list
    return list(set([unit for unit in syntactic_units if unit not in ignore_units]))

  doc1 = nlp(_sent1)
  doc2 = nlp(_sent2)

  # Lexical Analysis: Extract important tokens
  tokens1 = extract_key_syntactic_units(doc1)
  tokens2 = extract_key_syntactic_units(doc2)

  common_elements = set(tokens1).intersection(tokens2)
  common_elements = list(common_elements)

  # Construct the result with metadata
  return {
    "lexical_continuity": common_elements
  }


def check_syntactic_continuity(_sent1, _sent2):
  """
  This function checks if two sentences have syntactic continuity, which means that they have at least one common
  dependency pattern.
  :param _sent1:
  :param _sent2:
  :return: dict
  """

  def extract_dependency_patterns(doc):
    """
    This function extracts dependency patterns from a sentence.
    :param doc:
    :return:
    """
    return {(token.head.lemma_, token.lemma_, token.dep_) for token in doc}

  doc1 = nlp(_sent1)
  doc2 = nlp(_sent2)

  ignore_dependencies = {"punct"}

  # Enhanced Dependency Analysis
  dependencies1 = extract_dependency_patterns(doc1)
  dependencies2 = extract_dependency_patterns(doc2)

  # Remove dependencies that are in the ignore list
  dependencies1 = {dep for dep in dependencies1 if dep[2] not in ignore_dependencies}

  # Find common dependency patterns
  common_dependencies = dependencies1.intersection(dependencies2)

  # Construct the result with metadata
  return {
    "syntactic_continuity": common_dependencies
  }


def check_semantic_continuity(_sent1, _sent2, similarity_threshold=0.94):
  """
  This function checks if two sentences have semantic continuity, which means that they have at least one common
  key semantic unit (noun chunk or individual key token) with a cosine
  similarity above a threshold.
  :param _sent1:
  :param _sent2:
  :param similarity_threshold: float for the cosine similarity threshold
  :return: dict
  """

  def extract_key_semantic_units(doc):
    # Extracting noun chunks and individual key tokens as semantic units
    return [chunk.text for chunk in doc.noun_chunks if 'PRON' not in [token.pos_ for token in chunk]] + \
      [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and token.pos_ != "PRON"]

  """def vectorize_text(text):
    # Utilizing Spacy's ability to vectorize larger text chunks
    return nlp(text).vector"""

  def filter_subsets(tuples):
    # Remove tuples that are subsets of other tuples
    filtered_tuples = []
    for tuple1 in tuples:
      is_subset = False
      for tuple2 in tuples:
        if tuple1 != tuple2 and all(any(word1 in word2 for word1 in tuple1) for word2 in tuple2):
          # If all elements of tuple1 are subsets of elements in tuple2, set the flag to True
          is_subset = True
          break
      if not is_subset:
        filtered_tuples.append(tuple1)
    return filtered_tuples

  doc1 = nlp(_sent1)
  doc2 = nlp(_sent2)

  key_units1 = sorted(extract_key_semantic_units(doc1), key=len, reverse=True)
  key_units2 = sorted(extract_key_semantic_units(doc2), key=len, reverse=True)

  # Remove subphrases and overlapping words preferring longer phrases
  filtered_units1 = []
  for phrase in key_units1:
    if not any(phrase != other and other in phrase for other in filtered_units1):
      filtered_units1.append(phrase)
  filtered_units2 = []
  for phrase in key_units2:
    if not any(phrase != other and other in phrase for other in filtered_units2):
      filtered_units2.append(phrase)

  common_semantic_units = []
  for unit1 in key_units1:
    unit1 = nlp(str(unit1).lower())
    for unit2 in key_units2:
      unit2 = nlp(str(unit2).lower())
      if unit1.has_vector and unit2.has_vector:  # Ensuring both units have vectors
        similarity = unit1.similarity(unit2)
        if similarity >= similarity_threshold:
          # Check for overlapping terms, ensuring no unit is part of the other
          if not (unit1.text in unit2.text or unit2.text in unit1.text) and unit1.text != unit2.text:
            # Convert units to tokens and remove stop words and punctuation
            tokens1 = [token.text for token in nlp(unit1.text) if not token.is_stop and token.is_alpha]
            tokens2 = [token.text for token in nlp(unit2.text) if not token.is_stop and token.is_alpha]
            # Check if the units have any common tokens to avoid very lexically similar units
            if not bool(set(tokens1).intersection(tokens2)):
              common_semantic_units.append((unit1.text, unit2.text))

  # Remove tuples that are subsets of other tuples
  common_semantic_units = filter_subsets(common_semantic_units)

  return {
    "semantic_continuity": common_semantic_units
  }


def check_transition_markers_continuity(_sent):
  """
  This function checks if a sentence starts with a transition marker or contains a transition marker.
  :param _sent:
  :return: dict
  """
  doc = nlp(_sent)

  results = {
    "transition_markers_continuity": []
  }

  # Check if the sentence starts with a continuity marker
  matches = leading_continuity_matcher(doc)
  markers = {
    "type": "continue",
    "marker": [],
    "location": "leading"
  }
  for match_id, start, end in matches:
    marker = doc[start:end].text
    if start == 0:
      markers["marker"].append(marker.lower())
  if markers["marker"]:
    results["transition_markers_continuity"].append(markers)

  # Check if the sentence starts with a shift marker
  matches = leading_shift_matcher(doc)
  markers = {
    "type": "shift",
    "marker": [],
    "location": "leading"
  }
  for match_id, start, end in matches:
    marker = doc[start:end].text
    if start == 0:
      markers["marker"].append(marker.lower())
  if markers["marker"]:
    results["transition_markers_continuity"].append(markers)

  # Check if the sentence contains a continuity marker, not at the beginning
  matches = flexible_continuity_matcher(doc)
  markers = {
    "type": "continue",
    "marker": [],
    "location": "flexible"
  }
  for match_id, start, end in matches:
    marker = doc[start:end].text
    if start > 0:
      markers["marker"].append(marker.lower())
  if markers["marker"]:
    results["transition_markers_continuity"].append(markers)

  # Check if the sentence contains a shift marker, not at the beginning
  markers = {
    "type": "shift",
    "marker": [],
    "location": "flexible"
  }
  matches = flexible_shift_matcher(doc)
  for match_id, start, end in matches:
    marker = doc[start:end].text
    if start > 0:
      markers["marker"].append(marker.lower())
  if markers["marker"]:
    results["transition_markers_continuity"].append(markers)

  return results


def check_coreference(sent1, sent2):
  """
  #   This function uses the experimental spacy coreference resolution model to check if two sentences have
  #   coreference between each other. The function checks the presence of anaphoric references between the two sentences
  #   (when a word in the second sentence refers to a word in the first sentence) and cataphoric references (when a
  #   word in the first sentence refers to a word in the second sentence).
  #   The coreference model receives a joined sentence of the two sentences and checks if there is a coreference between
  #   the two sentences.
  #
  #   :param sent1:
  #   :param sent2:
  #   :return:
  #   """

  # join 2 sentences and create a new doc
  doc = nlp_sm_coref(sent1 + " " + sent2)

  # Get total number of tokens in sentence1
  doc1 = nlp(sent1)
  sentence1_end = len(doc1) - 1

  clusters = {}
  cluster_id = 1

  # pprint(doc.spans)

  # check if there are any coreference clusters that link both sentences
  for cluster in doc.spans:
    cluster_group = []
    if "head" in cluster:
      # If all spans are in sentence1 or all spans are in sentence2, which means no coreference between sentences
      if all(span.start < sentence1_end for span in doc.spans[cluster]) or all(
          span.start > sentence1_end for span in doc.spans[cluster]):
        continue
      else:
        for coreference in doc.spans[cluster]:
          # decompose the span in tokens, iterate them and check that they all are not punctuation
          # if they are not punctuation, add them to the cluster
          if not all(token.is_punct for token in coreference):
            coref = {
              "coref": coreference.text,
              "start": coreference.start,
              "end": coreference.end,
            }
            cluster_group.append(coref)

        # Skip coreferences with clusters of 2 elements that refer to the same
        if len(cluster_group) == 2 and cluster_group[0]["coref"].lower() == cluster_group[1]["coref"].lower():
          continue
        clusters[f"coreference_group_{cluster_id}"] = cluster_group
        cluster_id += 1

  return {
    "coreference": clusters
  }
