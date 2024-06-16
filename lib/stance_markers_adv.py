""" """

""" =================================================
Positive Affect
================================================= """

positive_adv = [
  "best", "better", "good", "effectively", "successfully", "fortunately", "rightly", "understandably",
  "freely", "commercially", "democratically", "enthusiastically", "environmentally", "judiciously", "mutually",
  "responsibly", "traditionally", "wisely", "fairly", "tirelessly",
]


def create_positive_adv(matcher):
  # pattern: lemma of ADV | example:
  pattern1 = [
    {"POS": "ADV", "LEMMA": {"IN": positive_adv}},
  ]
  matcher.add("POSITIVE_ADV", [pattern1])
  return matcher


def create_positive_adv_negation(matcher):
  """
  Add negation patterns of positive adverbs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: not ADV | example: "not effectively"
  pattern1 = [
    {"LOWER": "not"},
    {"LEMMA": {"IN": positive_adv}},
  ]
  matcher.add("POSITIVE_ADV_NEGATION", [pattern1])
  return matcher


""" =================================================
Negative Affect
================================================= """

negative_adv = [
  "aggressively", "unfortunately", "badly", "disturbingly", "sadly", "unjustly", "overwhelmingly", "painstakingly",
  "negatively", "arbitrarily",
]


def create_negative_adv(matcher):
  # pattern: lemma of ADV | example:
  pattern1 = [
    {"POS": "ADV", "LEMMA": {"IN": negative_adv}},
  ]
  matcher.add("NEGATIVE_ADV", [pattern1])
  return matcher


def create_negative_adv_negation(matcher):
  """
  Add negation patterns of negative adverbs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: not ADV | example: "not effectively"
  pattern1 = [
    {"LOWER": "not"},
    {"LEMMA": {"IN": negative_adv}},
  ]
  matcher.add("POSITIVE_ADV_NEGATION", [pattern1])
  return matcher


""" =================================================
Certainty
================================================= """

certainty_adv = [
  "absolutely", "clearly", "actually", "firmly", "fully", "precisely", "truly", "really", "seriously",
  "explicitly", "completely", "certainly", "definitely", "indeed", "obviously", "surely", "totally", "quickly",
  "frankly", "just", "all", "already", "always", "ever", "everywhere", "exactly", "finally", "entirely",
  "particularly", "immediately", "especially", "forever", "hence", "here", "often", "once", "personally",
  "strictly", "systemically", "then", "therefore", "thus", "too", "twice", "ultimately", "up", "well",
  "never", "anymore", "anywhere",
]

certainty_adv_phrase = [
  "of course", "without a doubt",
]


def create_certainty_adv(matcher):
  # pattern: lemma of ADV | example:
  pattern1 = [
    {"POS": "ADV", "LEMMA": {"IN": certainty_adv}},
  ]
  matcher.add("CERTAINTY_ADV", [pattern1])
  return matcher


def create_certainty_adv_negation(matcher):
  """
  Add negation patterns of positive adverbs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: not ADV | example: "not fully"
  pattern1 = [
    {"LOWER": "not"},
    {"LEMMA": {"IN": certainty_adv}},
  ]
  matcher.add("CERTAINTY_ADV_NEGATION", [pattern1])
  return matcher


def create_certainty_adv_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in certainty_adv_phrase]
  phrase_matcher.add("CERTAINTY_ADV_PHRASE", patterns)
  return phrase_matcher


""" =================================================
Doubt
================================================= """

doubt_adv = [
  "perhaps", "possibly", "probably", "eventually",
]


def create_doubt_adv(matcher):
  # pattern: lemma of ADV | example:
  pattern1 = [
    {"POS": "ADV", "LEMMA": {"IN": doubt_adv}},
  ]
  matcher.add("DOUBT_ADV", [pattern1])
  return matcher


def create_doubt_adv_negation(matcher):
  """
  Add negation patterns of doubt adverbs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: not ADV | example: "not possibly"
  pattern1 = [
    {"LOWER": "not"},
    {"LEMMA": {"IN": doubt_adv}},
  ]
  matcher.add("DOUBT_ADV_NEGATION", [pattern1])
  return matcher


""" =================================================
Emphatics
================================================= """

emphatic_adv = [
  "swiftly", "actively", "incredibly", "exceptionally", "extensively", "independently", "extremely", "profoundly",
  "significantly", "strongly", "very", "vigorously", "increasingly", "highly", "deeply", "immeasurably",
  "fundamentally", "largely", "quite", "broadly", "currently", "directly", "importantly", "equally", "far",
  "globally", "historically", "primarily", "principally", "rapidly", "severely", "overly", "fast", "over",
  "worldwide", "enough", "hard", "instinctively", "now", "overall", "privately", "repeatedly", "most",
  "furthermore", "even", "much", "outright", "philosophically", "more", "only", "simply", "still",
  "politically", "right", "so", "yet", "nevertheless", "also", "foremost", "further", "hardly", "presently",
  "simultaneously", "quick", "soon", "shortly", "quietly", "again", "anew", "recently", "slowly", "forward",
  "before", "seamlessly", "alone", "ahead", "away", "long", "longer", "down", "forth", "within", "early",
]

emphatic_adv_phrase = [
  "a lot", "at the top", "in fact", "in reality", "in particular", "in addition", "for sure",
]


def create_emphatic_adv(matcher):
  # pattern: lemma of ADV | example:
  pattern1 = [
    {"POS": "ADV", "LEMMA": {"IN": emphatic_adv}},
  ]
  matcher.add("EMPHATIC_ADV", [pattern1])
  return matcher


def create_emphatic_adv_negation(matcher):
  """
  Add negation patterns of positive adverbs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: not ADV | example: "not extensively"
  pattern1 = [
    {"LOWER": "not"},
    {"LEMMA": {"IN": emphatic_adv}},
  ]
  matcher.add("EMPHATIC_ADV_NEGATION", [pattern1])
  return matcher


def create_emphatic_adv_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in emphatic_adv_phrase]
  phrase_matcher.add("EMPHATIC_ADV_PHRASE", patterns)
  return phrase_matcher


def create_emphatic_adv_patterns(matcher):
  """
  Add patterns for adverbs that represent emphatic adverbs
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """

  # as I (HAVE) said|noted / say|note
  pattern1 = [
    {"LOWER": "as"},
    {"LOWER": {"IN": ["i", "we"]}},
    {"LEMMA": "have", "OP": "?"},
    {"LEMMA": {"IN": ["say", "note", "announce"]}}  # "indicate"
  ]
  # Pattern for "so" followed by any adverb (ADV)
  pattern2 = [
    {"LOWER": "so"},
    {"POS": "ADV"}
  ]
  matcher.add("AS_I_SAID", [pattern1])
  matcher.add("SO_ADV", [pattern2])
  return matcher


""" =================================================
Hedges
================================================= """

hedge_adv = [
  "closely", "maybe", "somewhat", "around", "mostly", "nearly", "partly", "rather", "virtually", "generally",
  "except", "alternatively", "instead", "later", "little", "pretty", "regardless", "about", "almost",
  "like", "potentially", "indirectly", "anyway", "but", "elsewhere", "however", "initially", "though",
]

hedge_adv_phrase = [
  "kind of", "sort of", "a bit", "a little", "in a way", "pretty much", "in general", "in part", "in principle",
]


def create_hedge_adv(matcher):
  # pattern: lemma of ADV | example:
  pattern1 = [
    {"POS": "ADV", "LEMMA": {"IN": hedge_adv}},
  ]
  matcher.add("HEDGE_ADV", [pattern1])
  return matcher


def create_hedge_adv_negation(matcher):
  """
  Add negation patterns of positive adverbs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: not ADV | example: "not somewhat"
  pattern1 = [
    {"LOWER": "not"},
    {"LEMMA": {"IN": hedge_adv}},
  ]
  matcher.add("HEDGE_ADV_NEGATION", [pattern1])
  return matcher


def create_hedge_adv_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in hedge_adv_phrase]
  phrase_matcher.add("HEDGE_ADV_PHRASE", patterns)
  return phrase_matcher


# Includes also all negation forms of:
# - positive advs
# - negative advs
# - certainty advs
# - doubt advs
# - emphatic advs
# - hedge advs
# - pro advs
# - con advs


""" =================================================
Pro
================================================= """

pro_adv = [
  "together",
]

pro_adv_phrase = [
  "on board",
]


def create_pro_adv(matcher):
  # pattern: lemma of ADV | example:
  pattern1 = [
    {"POS": "ADV", "LEMMA": {"IN": pro_adv}},
  ]
  matcher.add("PRO_ADV", [pattern1])
  return matcher


def create_pro_adv_negation(matcher):
  """
  Add negation patterns of positive adverbs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: not ADV | example: "not together"
  pattern1 = [
    {"LOWER": "not"},
    {"LEMMA": {"IN": pro_adv}},
  ]
  matcher.add("PRO_ADV_NEGATION", [pattern1])
  return matcher


def create_pro_adv_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in pro_adv_phrase]
  phrase_matcher.add("PRO_ADV_PHRASE", patterns)
  return phrase_matcher


""" =================================================
Con
================================================= """

con_adv = [
  "back", "detrimentally", "out", "outside", "not", "no", "off",
]

con_adv_phrase = [
  "in opposition",
]


def create_con_adv(matcher):
  # pattern: lemma of ADV | example:
  pattern1 = [
    {"POS": "ADV", "LEMMA": {"IN": con_adv}},
  ]
  matcher.add("CON_ADV", [pattern1])
  return matcher


def create_con_adv_negation(matcher):
  """
  Add negation patterns of positive adverbs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: not ADV | example: "not back"
  pattern1 = [
    {"LOWER": "not"},
    {"LEMMA": {"IN": con_adv}},
  ]
  matcher.add("CON_ADV_NEGATION", [pattern1])
  return matcher


def create_con_adv_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in con_adv_phrase]
  phrase_matcher.add("CON_ADV_PHRASE", patterns)
  return phrase_matcher


def create_con_adv_patterns(matcher):
  # ADV against | example: much against
  pattern1 = [
    {"POS": "ADV"},
    {"LEMMA": "against"},
  ]
  """# 
  pattern2 = [
    {"LOWER": "against"},
    {"LEMMA": {"IN": ["a", "an", "the"]}},
  ]"""
  # major concern | opposition | Example: "major concern" / "little more concern"
  pattern2 = [
    {"POS": "ADV"},
    {"LEMMA": {"IN": ["concern", "opposition", "disagreement", "threat"]}},
  ]
  matcher.add("ADV_AGAINST", [pattern1])
  matcher.add("ADV_CONCERN", [pattern2])
  return matcher


"""------------------------------------------------"""

advs = [
  "else", "by", "behind", "on", "overseas", "alike", "ago", "in", "as",
]
