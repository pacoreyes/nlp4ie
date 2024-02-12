
def check_minimal_meaning(doc):
  """
  Check if a sentence has at least one subject, predicate, and object
  :param doc: spaCy doc
  :return: bool
  """
  # These are the dependency tags for subjects, predicates, and objects
  subject_tags = {'nsubj', 'nsubjpass'}
  predicate_tags = {'ROOT'}
  object_tags = {'dobj', 'iobj', 'pobj'}

  # Initialize empty sets for subjects, predicates, and objects
  subjects = set()
  predicates = set()
  objects = set()

  for token in doc:
    if token.dep_ in subject_tags:
      subjects.add(token.text)
    elif token.dep_ in predicate_tags:
      predicates.add(token.text)
    elif token.dep_ in object_tags:
      objects.add(token.text)

  # A sentence has "minimal meaning" if it has at least one subject, predicate, and object
  # they don't need to be dependent on each other, they just need to be present
  return bool(subjects) and bool(predicates) and bool(objects)


def tokenize_text(text, nlp):
  doc = nlp(text)
  return [token.text for token in doc]


def check_if_has_one_word_or_more(doc):
  # use spaCy to check if the has at least one token that is a word
  # (not punctuation, not a number, not a stopword, etc.)
  for token in doc:
    if token.is_alpha:
      return True
  return False
