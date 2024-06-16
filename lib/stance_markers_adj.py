""""""

""" =================================================
Positive Affect
================================================= """

positive_adj = [
  "fortunate", "happy", "hopeful", "incredible", "interested", "pleased", "powerful", "proud", "transparent",
  "remarkable", "satisfied", "well", "positive", "fellow", "effective", "bold", "extraordinary", "young",
  "competitive", "ambitious", "stable", "prosperous", "peaceful", "inclusive", "innocent", "free", "fair", "clean",
  "equitable", "affordable", "strategic", "humanitarian", "available", "wonderful", "caregiving", "nimble",
  "agile", "good", "smart", "comprehensive", "safe", "strong", "innovative", "profitable", "attractive", "excited",
  "willing", "excellent", "super", "bilateral", "loved", "substantial", "magnificent", "legal", "popular",
  "durable", "steadfast", "meaningful", "friendly", "successful", "capable", "vibrant", "responsible", "sincere",
  "rightful", "honest", "qualified", "stimulant", "fast", "kind", "bipartisan", "functional", "indispensable",
  "pretty", "independent", "intellectual", "legitimate", "live", "moral", "mutual", "practical", "antitrust",
  "productive", "proper", "resilient", "singular", "sovereign", "special", "assistive", "right", "efficient",
  "collective", "constructive", "cooperative", "dedicated", "democratic", "diplomatic", "honorable", "hospitable",
  "unashamed", "uncensored", "eligible", "interconnected", "multilateral", "multinational", "founding", "no",
  "hardworking", "transformative", "important", "essential", "vital", "necessary", "balanced", "tolerated",
]

positive_adj_phrase = [
  "up to date", "up-to-date", "top-of-the-line", "top of the line"
]


def create_positive_adj(matcher):
  # pattern: lemma of ADJ | example:
  pattern1 = [
    {"POS": "ADJ", "LEMMA": {"IN": positive_adj}},
  ]
  matcher.add("POSITIVE_ADJ", [pattern1])
  return matcher


def create_positive_adj_negation(matcher):
  """
  Add negation patterns of positive adjs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: non-/not ADJ | example: "not clear" / "non-clear"
  pattern1 = [
    {"LOWER": {"IN": ["non", "not"]}},
    {"TEXT": {"IN": ["-", "–"]}, "OP": "?"},
    {"LEMMA": {"IN": positive_adj}},
  ]
  matcher.add("POSITIVE_ADJ_NEGATION", [pattern1])
  return matcher


def create_positive_adj_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in positive_adj_phrase]
  phrase_matcher.add("POSITIVE_ADJ_PHRASE", patterns)
  return phrase_matcher


""" =================================================
Negative Affect
================================================= """

negative_adj = [
  "aggressive", "affected", "devastating", "worse", "brutal", "unconstitutional", "underserved",
  "violent", "difficult", "selfish", "catastrophic", "horrible", "exhausted", "terrorist", "guilty", "isolated",
  "indifferent", "afraid", "harmful", "illegal", "dark", "wrong", "lethal", "anti", "vicious", "tragic", "alone",
  "dangerous", "poor", "ill", "worried", "worst", "deadly", "weak", "bad", "unsafe", "authoritarian", "unilateral",
  "threatening", "extremist", "disastrous", "contagious", "toxic", "terrible", "controversial", "misguided",
  "urgent", "grim", "vulnerable", "criminal", "dire", "severe", "angry", "anxious", "evil", "jeopardized",
  "contaminated", "racist", "discriminatory", "numb", "harsh", "alarmed", "ashamed", "disturbed", "embarrassed",
  "frightened", "frightening", "odd", "unfortunate", "deliberate", "deteriorated", "killer", "malicious",
  "uncontrolled", "unrealistic", "systemic", "unauthorized", "unpopular", "unending", "unrelenting", "unwise",
  "enslaved", "despicable", "grave", "dependent", "unjustified", "unprovoked", "arduous", "phony", "malign",
  "immoral", "unfair", "rash", "unnecessary", "overdue", "infectious", "abhorrent", "turbulent", "cruel",
  "divorced", "burdensome", "colonial", "complex", "faulty", "flagrant",  "hateful", "illicit", "imperialistic",
  "precarious", "rampant", "sectarian", "negative", "troublesome", "uncivil", "tough",
]

negative_adj_phrase = [
  "in jeopardy", "under attack", "under threat", "under siege", "at risk",
]


def create_negative_adj(matcher):
  # pattern: lemma of ADJ | example:
  pattern1 = [
    {"POS": "ADJ", "LEMMA": {"IN": negative_adj}},
  ]
  matcher.add("NEGATIVE_ADJ", [pattern1])
  return matcher


def create_negative_adj_negation(matcher):
  """
  Add negation patterns of negative adjs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: non-/not ADJ | example: "not violent"
  pattern1 = [
    {"LOWER": {"IN": ["non", "not"]}},
    {"TEXT": {"IN": ["-", "–"]}, "OP": "?"},
    {"LEMMA": {"IN": negative_adj}},
  ]
  matcher.add("NEGATIVE_NEGATION_ADJ", [pattern1])
  return matcher


def create_negative_adj_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in negative_adj_phrase]
  phrase_matcher.add("NEGATIVE_ADJ_PHRASE", patterns)
  return phrase_matcher


""" =================================================
Certainty
================================================= """

certainty_adj = [
  "absolute", "convinced", "determined", "distinct", "inconceivable", "evident", "explicit", "concrete",
  "feasible", "guaranteed", "inevitable", "infallible", "obvious", "patent", "resolute", "known", "true",
  "unarguable", "unavoidable", "unconditional", "unambiguous", "unanimous", "complete", "continued", "whole",
  "real", "credible", "aware", "affirmative", "secure", "institutional", "objective", "reliable", "realistic",
  "sustainable", "attentive", "cognizant", "concerted", "conducive", "accountable", "confident",
  "direct", "entire", "final", "firm", "flat", "frank", "inherent", "material", "only", "particular", "precise",
  "ready", "same", "structural", "substantive", "sure", "tangible", "unchanging", "certain", "resolved",
]

certainty_adj_phrase = [
  "well-known", "well known"
]


def create_certainty_adj(matcher):
  # pattern: lemma of ADJ | example:
  pattern1 = [
    {"POS": "ADJ", "LEMMA": {"IN": certainty_adj}},
  ]
  matcher.add("CERTAINTY_ADJ", [pattern1])
  return matcher


def create_certainty_adj_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in certainty_adj_phrase]
  phrase_matcher.add("CERTAINTY_ADJ_PHRASE", patterns)
  return phrase_matcher


def create_certainty_adj_negation(matcher):
  """
  Add negation patterns of certainty adjs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: non-/not ADJ | example: "not convinced"/"non-convincing"
  pattern1 = [
    {"LOWER": {"IN": ["non", "not"]}},
    {"TEXT": {"IN": ["-", "–"]}, "OP": "?"},
    {"LEMMA": {"IN": certainty_adj}},
  ]
  matcher.add("CERTAINTY_NEGATION_ADJ", [pattern1])
  return matcher


""" =================================================
Doubt
================================================= """

doubt_adj = [
  "ambiguous", "confused", "uncertain", "hypothetical", "undetermined", "impossible", "remote", "distrustful",
  "covert", "usual", "possible", "untrue",
]


def create_doubt_adj(matcher):
  # pattern: lemma of ADJ | example:
  pattern1 = [
    {"POS": "ADJ", "LEMMA": {"IN": doubt_adj}},
  ]
  matcher.add("DOUBT_ADJ", [pattern1])
  return matcher


def create_doubt_adj_negation(matcher):
  """
  Add negation patterns of doubt adjs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: non-/not ADJ | example: "not ambiguous"/"non-ambiguous"
  pattern1 = [
    {"LOWER": {"IN": ["non", "not"]}},
    {"TEXT": {"IN": ["-", "–"]}, "OP": "?"},
    {"LEMMA": {"IN": doubt_adj}},
  ]
  matcher.add("DOUBT_NEGATION_ADJ", [pattern1])
  return matcher


""" =================================================
Emphatics
================================================= """

emphatic_adj = [
  "equal", "open", "vigorous", "easy", "first", "groundbreaking", "historic", "serious", "ultimate", "shared",
  "key", "major", "maximum", "pivotal", "quick", "enormous", "tremendous", "total", "targeted", "clear",
  "unparalleled", "unprecedented", "visible", "crucial", "high", "sophisticated", "reliant", "active", "big",
  "seamless", "rising", "lasting", "significant", "integral", "very", "heavy", "huge", "prominent",
  "critical", "intense", "new", "longstanding", "countless", "detailed", "common", "economical",
  "central", "decisive", "dominant", "dramatic", "early", "fundamental", "immediate", "endless", "enough",
  "massive", "primary", "recent", "sweeping", "broad", "old", "basic", "bottom", "close", "current", "deep",
  "elementary", "emotional",  "disadvantaged", "eternal", "even", "foremost", "former", "specific", "steady",
  "front", "full", "further", "general", "global", "hard", "individual", "individualized", "instant", "prompt",
  "integrated", "just", "large", "last", "long", "many", "mass", "middle", "spare", "multiplier", "light",
  "minimum", "more", "most", "much", "multiple", "narrow", "small", "tectonic", "universal", "wide", "previous",
  "ongoing", "other", "outer", "overall", "own", "parallel", "past", "persistent", "present", "upcoming",
  "pursuant", "rich", "short", "similar", "simple", "single", "worldwide", "worthwhile", "uniform", "legendary",
  "fantastic", "such", "next", "top", "solid", "focused", "landmark", "great", "consistent", "leading",
  "modern",
]

emphatic_adj_phrase = [
  "long-range", "long range",
]


def create_emphatic_adj(matcher):
  # pattern: lemma of ADJ | example:
  pattern1 = [
    {"POS": "ADJ", "LEMMA": {"IN": emphatic_adj}},
  ]
  matcher.add("EMPHATIC_ADJ", [pattern1])
  return matcher


def create_emphatic_adj_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in emphatic_adj_phrase]
  phrase_matcher.add("EMPHATIC_ADJ_PHRASE", patterns)
  return phrase_matcher


def create_emphatic_adj_negation(matcher):
  """
  Add negation patterns of emphatic adjs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: non-/not ADJ | example: "not equal"/"non-equal"
  pattern1 = [
    {"LOWER": {"IN": ["non", "not"]}},
    {"TEXT": {"IN": ["-", "–"]}, "OP": "?"},
    {"LEMMA": {"IN": emphatic_adj}},
  ]
  matcher.add("EMPHATIC_NEGATION_ADJ", [pattern1])
  return matcher


def create_emphatic_adj_patterns(matcher):
  # Pattern for "real"/"so" followed by any adjective (ADJ)
  pattern1 = [
    {"LOWER": {"IN": ["real", "so"]}},
    {"POS": "ADJ"}]
  matcher.add("REAL_SO_ADJ", [pattern1])
  return matcher


""" =================================================
Hedge
================================================= """

hedge_adj = [
  "moderate", "preventative", "reasonable", "appropriate", "convenient", "satisfactory", "relevant", "reduced",
  "some", "few", "less", "little", "likely", "potential", "various", "several", "additional", "down",
  "alternative", "low", "nuanced",
]


def create_hedge_adj(matcher):
  # pattern: lemma of ADJ | example:
  pattern1 = [
    {"POS": "ADJ", "LEMMA": {"IN": hedge_adj}},
  ]
  matcher.add("HEDGE_ADJ", [pattern1])
  return matcher


def create_hedge_adj_negation(matcher):
  """
  Add negation patterns of hedge adjs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: non-/not ADJ | example: "not moderate"/"non-moderate"
  pattern1 = [
    {"LOWER": {"IN": ["non", "not"]}},
    {"TEXT": {"IN": ["-", "–"]}, "OP": "?"},
    {"LEMMA": {"IN": hedge_adj}},
  ]
  matcher.add("HEDGE_NEGATION_ADJ", [pattern1])
  return matcher


# Includes also all negation forms of:
# - positive adjs
# - negative adjs
# - certainty adjs
# - doubt adjs
# - emphatic adjs
# - pro adjs
# - con adjs


""" =================================================
Pro 
================================================= """

pro_adj = [
  "committed", "supportive", "favorable", "helpful", "backed", "encouraged", "strengthened", "sustained",
  "defended", "advanced", "developed", "grown", "enriched", "improved", "built", "invested", "defensive",
  "equipped", "funded", "bound", "commended", "respected", "recognized", "contributed", "joined", "pledged",
  "offered", "provided", "invited", "brought", "unleashed", "reaffirmed", "welcomed", "raised", "stood",
  "prioritized", "enabled", "sponsored", "maintained", "expanded", "allowed", "accepted", "implemented",
  "created", "agreed", "included", "renewed", "coordinated", "joint",
]


def create_pro_adj(matcher):
  # pattern: lemma of ADJ | example:
  pattern1 = [
    {"POS": "ADJ", "LEMMA": {"IN": pro_adj}},
  ]
  matcher.add("PRO_ADJ", [pattern1])
  return matcher


def create_pro_adj_negation(matcher):
  """
  Add negation patterns of pro adjs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: non-/not ADJ | example: "not committed"/"non-committed"
  pattern1 = [
    {"LOWER": {"IN": ["non", "not"]}},
    {"TEXT": {"IN": ["-", "–"]}, "OP": "?"},
    {"LEMMA": {"IN": pro_adj}},
  ]
  matcher.add("PRO_NEGATION_ADJ", [pattern1])
  return matcher


def create_pro_adj_patterns(matcher):
  # "second commitment"
  pattern1 = [
    {"POS": "ADJ"},
    {"LEMMA": {"IN": ["commitment", "support", "help", "endorsement", "favor", "agreement", "effort"]}},
  ]
  # "our/my commitment"
  """pattern2 = [
    {"LOWER": {"IN": ["our", "my"]}},
    {"LEMMA": {"IN": ["commitment", "support", "help", "endorsement", "favor", "agreement", "effort"]}},
  ]"""
  matcher.add("PRO_ADJ_COMMITMENT", [pattern1])
  # matcher.add("PRO_OUR_COMMITMENT", [pattern2])
  return matcher


""" =================================================
Con
================================================= """

con_adj = [
  "opposed", "compromised", "unwilling", "vigilant", "refuted", "counter", "fought", "concerned",
  "weakened", "threatened", "exposed", "diminished", "lessened", "criticized", "wracked", "divided", "faced",
  "condemned", "denounced", "accused", "deterred", "confronted", "limited", "hurt", "imposed", "interrupted",
  "exploited", "posed", "misled", "forced", "suffered", "cut", "prevented", "deplored", "suppressed", "ended",
  "defeated", "concerning", "unacceptable", "contrary",
]

con_adj_phrase = [
  "held down", "held-down", "broken down", "broken-down", "angry at",
]


def create_con_adj(matcher):
  # pattern: lemma of ADJ | example:
  pattern1 = [
    {"POS": "ADJ", "LEMMA": {"IN": con_adj}},
  ]
  matcher.add("CON_ADJ", [pattern1])
  return matcher


def create_con_adj_negation(matcher):
  """
  Add negation patterns of con adjs to the matcher, used as hedges
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """
  # pattern: non-/not ADJ | example: "not opposed"/"non-opposed"
  pattern1 = [
    {"LOWER": {"IN": ["non", "not"]}},
    {"TEXT": {"IN": ["-", "–"]}, "OP": "?"},
    {"LEMMA": {"IN": con_adj}},
  ]
  matcher.add("CON_NEGATION_ADJ", [pattern1])
  return matcher


def create_con_adj_phrase(nlp, phrase_matcher):
  patterns = [nlp(phrase) for phrase in con_adj_phrase]
  phrase_matcher.add("CON_ADJ_PHRASE", patterns)
  return phrase_matcher


def create_con_adj_patterns(matcher):
  """
  Add patterns for adjectives that represent con adjectives
  :param matcher: spacy Matcher
  :return: spaCy Matcher with negation patterns
  """

  # major concern | opposition | Example: "major concern" / "little more concern"
  pattern1 = [
    {"POS": "ADJ"},
    {"LEMMA": {"IN": ["concern", "opposition", "disagreement", "threat"]}},
  ]
  # "a matter of"/"more than a" concern(s)
  pattern2 = [
    {"LOWER": {"IN": ["a matter of", "more than a", "list of"]}},
    {"LEMMA": "concern"},
  ]
  # "our/my concern" | opposition
  """pattern3 = [
    {"LOWER": {"IN": ["our", "my"]}},
    {"LEMMA": {"IN": ["concern", "opposition"]}},
  ]"""
  # united against
  pattern4 = [
    {"POS": "ADJ"},
    {"LEMMA": "against"},
  ]
  matcher.add("CON_ADJ_AND_NOUN", [pattern1])
  matcher.add("CON_ADJ_CONCERN", [pattern2])
  # matcher.add("CON_OUR_CONCERN", [pattern3])
  matcher.add("CON_ADJ_AGAINST", [pattern4])
  return matcher


"""------------------------------------------------"""

adjectives = [
  "united",
]
