# from pprint import pprint

"""
List of entity labels that sPacy recognizes (https://spacy.io/api/annotation#named-entities):

- PERSON: People, including fictional.
- NORP: Nationalities or religious or political groups.
- FAC: Buildings, airports, highways, bridges, etc.
- ORG: Companies, agencies, institutions, etc.
- GPE: Countries, cities, states.
- LOC: Non-GPE locations, mountain ranges, bodies of water.
- PRODUCT: Objects, vehicles, foods, etc. (Not services.)
- EVENT: Named hurricanes, battles, wars, sports events, etc.
- WORK_OF_ART: Titles of books, songs, etc.
- LAW: Named documents made into laws.
- LANGUAGE: Any named language.
- DATE: Absolute or relative dates or periods.
- TIME: Times smaller than a day.
- PERCENT: Percentage, including "%".
- MONEY: Monetary values, including unit.
- QUANTITY: Measurements, as of weight or distance.
- ORDINAL: “first”, “second”, etc.
- CARDINAL: Numerals that do not fall under another type.
"""


def anonymize_text(text, nlp):
  """
  Anonymize text by replacing all entities with their labels.

  :param text: string, The input text to be anonymized.
  :param nlp: The spacy NLP model used for entity recognition.
  :return:
  """

  doc = nlp(text)
  ents = [e for e in doc.ents if
          e.label_ in ["PERSON",
                       "ORG",
                       "NORP",
                       "TIME",
                       "DATE",
                       "CARDINAL",
                       "MONEY",
                       "FAC",
                       "QUANTITY",
                       "PERCENT",
                       "GPE"]]
  sorted_ents = sorted(ents, key=lambda e: e.start_char, reverse=True)
  for ent in sorted_ents:
    # text = text[:ent.start_char] + "[" + ent.label_ + "]" + text[ent.end_char:]
    text = text[:ent.start_char] + ent.label_ + text[ent.end_char:]
  return text


def custom_anonymize_text(text, nlp, labels=None):
  """
  Anonymize text by replacing specified entities with their labels.

  If no labels are specified, all recognized entities will be anonymized.

  :param text: The input text to be anonymized.
  :param nlp: The NLP model used for entity recognition.
  :param labels: A list of entity labels to be anonymized. If not provided, all entities will be anonymized.
  :return: Anonymized text.
  """

  # Default labels to anonymize if none are provided
  default_labels = [
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL"
  ]

  # If no labels are provided, use the default list
  if labels is None:
    labels = default_labels

  doc = nlp(text)
  ents = [e for e in doc.ents if e.label_ in labels]
  sorted_ents = sorted(ents, key=lambda e: e.start_char, reverse=True)
  for ent in sorted_ents:
    # text = text[:ent.start_char] + "[" + ent.label_ + "]" + text[ent.end_char:]
    text = text[:ent.start_char] + ent.label_ + text[ent.end_char:]
  return text


def replace_speaker_labels(text):
  for label in SPEAKER_LABELS:
    text = text.replace(label, "ENTITY")
  return text


SPEAKER_LABELS = [
  "[SPEAKER]",
  "[SPEAKER1]",
  "[SPEAKER2]",
  "[SPEAKER3]",
  "[SPEAKER4]",
  "[SPEAKER5]",
  "[SPEAKER6]",
  "[SPEAKER7]",
  "[SPEAKER8]",
  "[SPEAKER9]",
  "[SPEAKER10]",
  "[SPEAKER11]",
  "[SPEAKER12]",
  "[SPEAKER13]",
  "[SPEAKER14]",
  "[SPEAKER15]",
  "[INTERVIEWER]",
  "[INTERVIEWER1]",
  "[INTERVIEWER2]",
  "[INTERVIEWER3]",
  "[INTERVIEWER4]",
  "[INTERVIEWER5]",
  "[INTERVIEWER6]",
  "[INTERVIEWER7]",
  "[INTERVIEWER8]",
  "[INTERVIEWER9]",
  "[INTERVIEWER10]",
  "[INTERVIEWER11]",
  "[INTERVIEWER12]",
  "[INTERVIEWER13]",
  "[INTERVIEWER14]",
  "[INTERVIEWER15]",
  "[INTERVIEWED]",
  "[INTERVIEWED1]",
  "[INTERVIEWED2]",
  "[INTERVIEWED3]",
  "[INTERVIEWED4]",
  "[INTERVIEWED5]",
  "[INTERVIEWED6]",
  "[INTERVIEWED7]",
  "[INTERVIEWED8]",
  "[INTERVIEWED9]",
  "[INTERVIEWED10]",
  "[INTERVIEWED11]",
  "[INTERVIEWED12]",
  "[INTERVIEWED13]",
  "[INTERVIEWED14]",
  "[INTERVIEWED15]",
]
