"""
Stance markers and modal verbs
"""

""" =================================================
Modal verbs
================================================= """

predictive_modal = [
  "will", "shall", "would"
]

possibility_modal = [
  "may", "might", "could", "can"
]

necessity_modal = [
  "must", "should"
]

predictive_modal_phrase = [
  "go to",
]

necessity_modal_phrase = [
  "have to",
  "need to",
  "ought to",
]


def create_predictive_modal(matcher):
  pattern1 = [
    {"LEMMA": {"IN": predictive_modal}},
  ]
  matcher.add("PRED_MODALS", [pattern1])
  return matcher


def create_possibility_modal(matcher):
  pattern1 = [
    {"LEMMA": {"IN": possibility_modal}},
  ]
  matcher.add("POSS_MODALS", [pattern1])
  return matcher


def create_necessity_modal(matcher):
  pattern1 = [
    {"LEMMA": {"IN": necessity_modal}},
  ]
  matcher.add("NEC_MODALS", [pattern1])
  return matcher


def create_predictive_modal_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in predictive_modal_phrase]
  phrase_matcher.add("PRED_MODALS", None, *patterns)
  return phrase_matcher


def create_necessity_modal_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in necessity_modal_phrase]
  phrase_matcher.add("NEC_MODALS", None, *patterns)
  return phrase_matcher
