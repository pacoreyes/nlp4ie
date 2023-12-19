"""
This script analyzes the linguistic features that link or not pairs of sentences to classify
them as "continue" or "not continue" pairs.

The rules are based on the paper "A Corpus of Sentence-level Discourse Parsing in English News"
https://aclanthology.org/N03-1030.pdf
"""
from collections import Counter

import spacy
from scipy.spatial.distance import cosine

from lib.lexicons import transition_markers

# from pprint import pprint

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_lg")

# Load coreference resolution model
nlp_sm_coref = spacy.load("en_core_web_sm")
nlp_coref = spacy.load("en_coreference_web_trf")

nlp_coref.replace_listeners("transformer", "coref", ["model.tok2vec"])
nlp_coref.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])

nlp_sm_coref.add_pipe("coref", source=nlp_coref)
nlp_sm_coref.add_pipe("span_resolver", source=nlp_coref)

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
                       ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "SCONJ", "NUM"]]

    # Remove tokens that are in the ignore list
    return list(set([unit for unit in syntactic_units if unit not in ignore_units]))

  doc1 = nlp(_sent1)
  doc2 = nlp(_sent2)

  # Lexical Analysis: Extract important tokens
  tokens1 = extract_key_syntactic_units(doc1)
  tokens2 = extract_key_syntactic_units(doc2)

  # Count occurrences of each token
  counter1 = Counter(tokens1)
  counter2 = Counter(tokens2)

  # Find common elements and their counts
  common_elements = counter1 & counter2

  # Construct the result with metadata
  return {
    "lexical_continuity": len(common_elements) > 0,
    "metadata": {
      "common_elements": dict(common_elements)
    }
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

  ignore_dependencies = {"be", "punct", "."}

  # Enhanced Dependency Analysis
  dependencies1 = extract_dependency_patterns(doc1)
  dependencies2 = extract_dependency_patterns(doc2)

  # Remove dependencies that are in the ignore list
  dependencies1 = {dep for dep in dependencies1 if dep[2] not in ignore_dependencies}

  # Find common dependency patterns
  common_dependencies = dependencies1.intersection(dependencies2)
  has_common_dependencies = len(common_dependencies) > 0

  # Construct the result with metadata
  return {
    "syntactic_continuity": has_common_dependencies,
    "metadata": {
      "common_dependencies": common_dependencies
    }
  }


def check_semantic_continuity(_sent1, _sent2, similarity_threshold=0.75):
  """
  This function checks if two sentences have semantic continuity, which means that they have at least one common
  named entity or at least one pair of key semantic units (nouns, verbs, adjectives, adverbs) with a semantic
  similarity above a threshold.
  :param _sent1:
  :param _sent2:
  :param similarity_threshold:
  :return: dict
  """
  def extract_key_semantic_units(doc):
    """
    This function extracts key semantic units from a sentence.
    :param doc:
    :return:
    """
    # Extract lemmas of key semantic units
    return [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]

  def check_common_entities(_doc1, _doc2):
    # Check for common named entities
    entities1 = {ent.lemma_ for ent in _doc1.ents}
    entities2 = {ent.lemma_ for ent in _doc2.ents}
    common = entities1.intersection(entities2)
    return common, len(common) > 0

  doc1 = nlp(_sent1)
  doc2 = nlp(_sent2)

  # Extract key semantic units: nouns, verbs, adjectives, adverbs
  key_units1 = extract_key_semantic_units(doc1)
  key_units2 = extract_key_semantic_units(doc2)

  # Calculate semantic similarity for key unit pairs
  similarities_above_threshold = {}
  for unit1 in key_units1:
    for unit2 in key_units2:
      vec1 = nlp.vocab[unit1].vector
      vec2 = nlp.vocab[unit2].vector
      if vec1.any() and vec2.any():  # Check if vectors exist
        similarity = 1 - cosine(vec1, vec2)
        if similarity >= similarity_threshold:
          similarities_above_threshold[(unit1, unit2)] = similarity

  # Named Entity Recognition: Check for common named entities
  common_entities, entity_continuity = check_common_entities(doc1, doc2)

  # Evaluate continuity based on semantic analysis
  _continuity = bool(similarities_above_threshold) or entity_continuity

  # Construct the result with metadata
  return {
    "semantic_continuity": _continuity,
    "metadata": {
      "similarities": similarities_above_threshold,
      "common_entities": list(common_entities)
    }
  }


def check_transition_markers_continuity(_sent):
  """
  This function checks if a sentence starts with a transition marker or contains a transition marker.
  :param _sent:
  :return: dict
  """
  for location, categories in transition_markers.items():
    for category, markers in categories.items():
      for marker in markers:

        if _sent.lower().startswith(marker.lower()) and location == "leading_markers":
          return {
            "transition_markers_continuity": True,
            "metadata": {
              "transition_marker": marker,
              "location": location
            }
          }
        elif not _sent.lower().startswith(
            marker.lower()) and marker.lower() in _sent.lower() and location == "flexible_markers":
          return {
            "transition_markers_continuity": True,
            "metadata": {
              "transition_marker": marker,
              "location": location
            }
          }
  return {
    "transition_markers_continuity": False,
  }


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

  # check if there are any coreference clusters that link both sentences
  doc1 = nlp(sent1)
  # Get total number of tokens in sentence1
  sentence1_end = len(doc1) - 1

  clusters = {}
  cluster_id = 1

  for cluster in doc.spans:
    cluster_group = []
    if "head" in cluster:
      # If all spans are in sentence1 or all spans are sentence2,
      # which means no coreference between sentences
      if all(span.start < sentence1_end for span in doc.spans[cluster]) or all(
          span.start > sentence1_end for span in doc.spans[cluster]
      ):
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

        clusters[f"coreference_group_{cluster_id}"] = cluster_group
        cluster_id += 1

  return {
    "coreference": bool(clusters),
    "metadata": {
      "coreference_clusters": clusters
    }
  }
