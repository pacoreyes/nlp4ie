"""
This script analyzes the linguistic features that link or not pairs of sentences to classify them as "continue" or
"not continue" pairs.

The rules are based on the paper "A Corpus of Sentence-level Discourse Parsing in English News"
https://aclanthology.org/N03-1030.pdf
"""

import spacy

from lib.transition_markers import transition_markers

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

# Retrieve transition markers from lexicons.py
leading_markers_continuity = transition_markers["leading_markers"]["topic_continuity"]
leading_markers_shift = transition_markers["leading_markers"]["topic_shift"]

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

  referential_pronouns = [
    "this", "these", "those",  # Demonstrative Determiners/Pronouns (less "that" which is a determiner)
    # "my", "your", "his", "her", "its", "our", "their",  # Possessive Determiners
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours", "he", "him", "his", "she", "her",
    "hers", "it", "its", "they", "them", "their", "theirs",  # Personal Pronouns
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves",  # Reflexive Pronouns
    "who", "whom", "whose", "which",  # Relative Pronouns (less "that" which is a determiner)
    "what", "which", "who", "whom", "whose",  # Interrogative Pronouns
    "anyone", "anything", "everybody", "everything", "someone", "something", "none", "nothing",  # Indefinite Pronouns
    # "each other", "one another"  # Reciprocal Pronouns
  ]

  def extract_key_lexical_units(doc):
    lexical_units = [(token.lemma_.lower(), token.i) for token in doc
                     if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"]
                     or token.text.lower() in referential_pronouns
                     or (token.text.lower() == "that" and token.pos_ == "DET")]
    # Remove tokens that are in the ignore list
    return list(set([unit for unit in lexical_units if unit[0] not in ignore_units]))

  doc1 = nlp(_sent1)
  doc2 = nlp(_sent2)

  # Lexical Analysis: Extract important terms
  terms1 = extract_key_lexical_units(doc1)
  terms2 = extract_key_lexical_units(doc2)

  # Create dictionaries to map strings to their full tuple for each document
  map1 = {term[0]: term for term in terms1}
  map2 = {term[0]: term for term in terms2}

  # Find common strings and get the tuples from both sets
  common_terms = [((map1[string][0], map1[string][1]), (map2[string][0], map2[string][1]))
                  for string in set(map1.keys()).intersection(map2.keys())]

  # Construct the result with metadata
  return {
    "lexical_continuity": common_terms
  }


def check_syntactic_continuity(_sent1, _sent2):
  """
  This function checks if two sentences have syntactic continuity, which means that they have at least one common
  dependency pattern.
  :param _sent1: First sentence (string)
  :param _sent2: Second sentence (string)
  :return: dict
  """

  def extract_dependency_patterns(doc):
    """
    This function extracts dependency patterns from a sentence.
    :param doc: Spacy Doc
    :return: set
    """
    # Returning patterns with (head_lemma, lemma, dep) structure
    return {((token.head.lemma_, token.head.i), (token.lemma_, token.i), token.dep_) for token in doc}

  doc1 = nlp(_sent1)
  doc2 = nlp(_sent2)

  ignore_dependencies = {"punct"}

  # Extract dependencies ignoring punctuation
  dependencies1 = extract_dependency_patterns(doc1)
  dependencies2 = extract_dependency_patterns(doc2)

  # Transform dependencies to ignore numbers
  transformed_deps1 = {(dep[0][0], dep[1][0], dep[2]) for dep in dependencies1 if dep[2] not in ignore_dependencies}
  transformed_deps2 = {(dep[0][0], dep[1][0], dep[2]) for dep in dependencies2 if dep[2] not in ignore_dependencies}

  # Find common dependency patterns based on transformed data
  common_transformed_deps = transformed_deps1.intersection(transformed_deps2)

  # Mapping back to original full dependencies with indexes
  original_deps_map1 = {(dep[0][0], dep[1][0], dep[2]): dep for dep in dependencies1}
  original_deps_map2 = {(dep[0][0], dep[1][0], dep[2]): dep for dep in dependencies2}

  common_dependencies = [(original_deps_map1[dep], original_deps_map2[dep]) for dep in common_transformed_deps]

  # Construct the result with metadata
  return {
    "syntactic_continuity": common_dependencies
  }


def check_semantic_continuity(_sent1, _sent2, similarity_threshold=0.75):
  """
  This function checks if two sentences have semantic continuity at the token level, evaluating
  cosine similarity above a specified threshold for tokens classified as "NOUN", "VERB", "ADJ", or "ADV".
  Identical tokens are excluded from the similarity check.
  :param _sent1: First sentence (string)
  :param _sent2: Second sentence (string)
  :param similarity_threshold: float for the cosine similarity threshold
  :return: dict containing lists of tuples with similar terms and their token indexes
  """

  # Initialize NLP model, assuming nlp is already imported and configured
  doc1 = nlp(_sent1)
  doc2 = nlp(_sent2)

  # Extract tokens of interest based on their POS tags and enumerate them to get the token index
  tokens1 = [(token.lemma_.lower(), token.i, token.pos_) for token in doc1
             if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]]
  tokens2 = [(token.lemma_.lower(), token.i, token.pos_) for token in doc2
             if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]]

  common_semantic_units = []

  # Compare each token in doc1 to each token in doc2, excluding identical tokens
  for token1, idx1, pos1 in tokens1:
    token1_doc = nlp(token1)
    for token2, idx2, pos2 in tokens2:
      if token1 != token2:  # Skip identical tokens /  and pos1 == pos2
        token2_doc = nlp(token2)
        if token1_doc.has_vector and token2_doc.has_vector:  # Ensure both tokens have vectors
          similarity = token1_doc.similarity(token2_doc)
          if similarity >= similarity_threshold:
            # Append tuple pairs (string, index) to the results
            common_semantic_units.append(((token1, idx1), (token2, idx2)))

  return {
    "semantic_continuity": common_semantic_units
  }


def check_transition_markers_continuity(_sent):
  """
  This function checks if a sentence starts with a transition marker.
  :param _sent:
  :return: dict
  """
  doc = nlp(_sent)

  results = {
    "transition_markers": []
  }

  # Get first token from doc
  first_token = doc[0].text.lower()
  # Check if first token is in leading_markers_continuity
  if first_token in leading_markers_continuity:
    results["transition_markers"].append({"continue": first_token})
  # Check if first token is in leading
  elif first_token in leading_markers_shift:
    results["transition_markers"].append({"shift": first_token})

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

  clusters = []
  cluster_id = 1

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

            # loc_sentence = None
            if coreference.start < sentence1_end:
              loc_sentence = "s1"
            else:
              loc_sentence = "s2"

            coref = [
              coreference.text,
              coreference.start,
              loc_sentence
            ]
            cluster_group.append(coref)

        # Skip coreferences with clusters of 2 elements that refer to the same
        if len(cluster_group) == 2 and cluster_group[0][0].lower() == cluster_group[1][0].lower():
          continue
        clusters.append(cluster_group)
        cluster_id += 1

  return {
    "coreference": clusters
  }
