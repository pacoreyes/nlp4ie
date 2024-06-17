""" """

""" =================================================
Positive Affect
================================================= """

positive_verb = [
  "achieve", "enjoy", "excel", "flourish", "like", "love", "please", "improve", "produce", "promise", "prosper",
  "refresh", "relish", "enrich", "alleviate", "implement", "soften", "strive", "succeed", "suit",
  "surprise", "thank", "wish", "clear", "relieve", "create", "rebuild", "build", "accomplish", "save", "renew",
  "tolerate", "preserve", "care", "rescue", "revitalize", "reform", "recover", "transform", "address",
  "modernize", "rally", "hearten", "satisfy", "forge", "immunize", "repair", "install", "reward",  "envision",
  "revamp", "win", "rid", "empower", "cherish", "thrive", "guide", "fulfil", "materialize",  "discover",
  "educate", "clean", "cure", "secure", "restore", "plow", "engage", "endure", "aspire", "reach", "qualify",
  "fix", "resolve", "streamline", "drive", "revise", "train", "serve", "stimulate", "decriminalize", "coordinate",
  "organize", "work", "count", "encourage", "strengthen", "upgrade", "enhance",
  # "make",
]


def create_positive_verb(matcher):
  # pattern: lemma of VERB | example:
  pattern1 = [
    {"POS": "VERB", "LEMMA": {"IN": positive_verb}},
  ]
  matcher.add("POSITIVE_VERB", [pattern1])
  return matcher


def create_positive_verb_negation(matcher):
  # pattern: not VERB | example: "not achieve"
  pattern2 = [
    {"LOWER": "not"},
    {"POS": "VERB", "LEMMA": {"IN": positive_verb}},
  ]
  matcher.add("POSITIVE_VERB_NEGATION", [pattern2])
  return matcher


""" =================================================
Negative Affect
================================================= """

negative_verb = [
  "aggravate", "confuse", "discourage", "disturb", "fear", "hate", "kill", "pain", "possess", "resent",
  "trouble", "worry", "die", "violate", "cause", "waste", "lose", "impose", "violent", "expose", "damage",
  "concern", "dump", "prey", "war", "stir", "endanger", "jeopardize", "destroy", "deteriorate",
  "dismantle", "overturn", "overthrow", "pose", "overwhelm", "destabilize", "cheat", "steal",
  "suffer", "scourge", "invade", "abandon", "fail", "precipitate", "imperil", "enslave", "misuse", "harm",
  "intimidate", "rage", "dethrone", "criminalize", "misrepresent", "nuclearize", "inconvenience",
  "cope", "burn", "bury", "censor", "delay", "distract", "forget", "hang", "mislead", "stumble", "traffic",
  "reshore", "sacrifice", "spiral", "stricken", "demand", "detain", "divide", "compromise", "bear", "pay",
  "spend", "loom", "tear", "wrack", "wrench", "break", "crumble", "hurt", "depend",
]


def create_negative_verb(matcher):
  # pattern: lemma of VERB | example:
  pattern1 = [
    {"POS": "VERB", "LEMMA": {"IN": negative_verb}},
  ]
  matcher.add("NEGATIVE_VERB", [pattern1])
  return matcher


def create_negative_verb_negation(matcher):
  # pattern: not VERB | example: "not confuse"
  pattern2 = [
    {"LOWER": "not"},
    {"POS": "VERB", "LEMMA": {"IN": negative_verb}},
  ]
  matcher.add("NEGATIVE_VERB_NEGATION", [pattern2])
  return matcher


""" =================================================
Certainty
================================================= """

certainty_verb = [
  "believe", "conclude", "demonstrate", "determine", "establish", "know", "note", "order", "perceive", "prove",
  "realize", "ensure", "uphold", "confirm", "rely", "recognize", "include", "reaffirm",
  "acknowledge", "affirm", "assure", "accept", "inform", "designate", "show", "aim", "attest", "convince",
  "recognise", "swear", "present", "notify", "introduce", "outline", "understand", "learn", "solidify", "announce",
  "illustrate", "express", "explain", "state", "define",
]

certainty_verb_phrase = [
  "make sure"
]


def create_certainty_verb(matcher):
  # pattern: lemma of VERB | example:
  pattern1 = [
    {"POS": "VERB", "LEMMA": {"IN": certainty_verb}},
  ]
  matcher.add("CERTAINTY_VERB", [pattern1])
  return matcher


def create_certainty_verb_negation(matcher):
  # pattern: not VERB | example: "not demonstrate"
  pattern2 = [
    {"LOWER": "not"},
    {"POS": "VERB", "LEMMA": {"IN": certainty_verb}},
  ]
  matcher.add("CERTAINTY_VERB_NEGATION", [pattern2])
  return matcher


def create_certainty_verb_phrase(nlp, matcher):
  patterns = [nlp(phrase) for phrase in certainty_verb_phrase]
  matcher.add("CERTAINTY_VERB_PHRASE", patterns)
  return matcher


def create_certainty_verb_patterns(matcher):
  # Pattern: the/this/that/it (ADJ) (NOUN) lemma of shows (that)
  """
  :param matcher: a spaCy Matcher object
  :return: a spaCy Matcher object
  """
  """pattern2 = [
    {"POS": "NOUN"},
    {"LEMMA": "show"},
    {"LOWER": "that", "OP": "?"}
  ]"""
  # have shown
  pattern4 = [
    {"LEMMA": "have"},
    {"LEMMA": "show"}
  ]
  # must be
  """pattern5 = [
    {"LEMMA": "must"},
    {"LEMMA": "be"}
  ]"""
  # matcher.add("CERTAINTY_VERB_THAT", [pattern2])
  matcher.add("CERTAINTY_HAVE_SHOWN", [pattern4])
  # matcher.add("CERTAINTY_MUST_BE", [pattern5])
  return matcher


""" =================================================
Doubt
================================================= """

doubt_verb = [
  "think", "appear", "expect", "feel", "imply", "indicate", "seem", "sense", "try", "prefer",
  "question", "waver", "distrust", "challenge", "hope", "attempt", "intend",
]


def create_doubt_verb(matcher):
  # pattern: lemma of VERB | example:
  pattern1 = [
    {"POS": "VERB", "LEMMA": {"IN": doubt_verb}},
  ]
  matcher.add("DOUBT_VERB", [pattern1])
  return matcher


def create_doubt_verb_negation(matcher):
  # pattern: not VERB | example: "not achieve"
  pattern2 = [
    {"LOWER": "not"},
    {"POS": "VERB", "LEMMA": {"IN": doubt_verb}},
  ]
  matcher.add("DOUBT_VERB_NEGATION", [pattern2])
  return matcher


""" =================================================
Emphatics
================================================= """

emphatic_verb = [
  "focus", "highlight", "prioritize", "multiply", "increase", "deepen", "repeat", "insist",
  "emphasize", "harden", "continue", "boost", "redouble", "emerge", "complete", "close", "take", "center",
  "target", "point", "maintain", "grow", "amplify", "augment", "extend", "consolidate",  "transition",
  "elevate", "bring", "unleash", "raise", "leverage", "prioritise", "mobilize", "rise", "expand", "disrupt",
  "exploit", "force", "accelerate", "arise", "concentrate", "condition", "plead", "converge", "connect", "lead",
  "escalate", "evolve", "exaggerate", "exceed", "excite", "generate", "intensify", "enforce", "spur", "keep",
  "rededicate", "redefine", "push", "jump", "undertake", "consume", "integrate", "broaden", "begin", "declare",
  "value", "open", "pursue", "acquire", "articulate", "bridge", "broker", "reignite", "double", "anticipate",
  "earn", "impact", "lift", "ground", "orient", "enter", "identify", "exercise", "become", "prepare"
]

emphatic_verb_phrase = [
  "fan the flames", "call for", "make clear", "point out"
]


def create_emphatic_verb(matcher):
  # pattern: lemma of VERB | example:
  pattern1 = [
    {"POS": "VERB", "LEMMA": {"IN": emphatic_verb}},
  ]
  matcher.add("EMPHATIC_VERB", [pattern1])
  return matcher


def create_emphatic_verb_negation(matcher):
  # pattern: not VERB | example: "not achieve"
  pattern2 = [
    {"LOWER": "not"},
    {"POS": "VERB", "LEMMA": {"IN": emphatic_verb}},
  ]
  matcher.add("EMPHATIC_VERB_NEGATION", [pattern2])
  return matcher


def create_emphatic_verb_phrase(nlp, matcher):
  patterns = [nlp(phrase) for phrase in emphatic_verb_phrase]
  matcher.add("EMPHATIC_VERB_PHRASE", patterns)
  return matcher


def create_emphatic_verb_patterns(matcher):
  # Pattern for "do" followed by any verb (VERB)
  pattern_do_verb = [
    {"LEMMA": "do"},
    {"POS": "VERB"}
  ]
  matcher.add('DO_VERB', [pattern_do_verb])
  return matcher


""" =================================================
Hedges
================================================= """

hedge_verb = [
  "lower", "weaken", "diminish", "reduce", "lessen", "dilute", "avoid", "cut", "creep", "hide",
  "near", "decline", "degrade", "deter", "undermine", "impair", "restrict", "hinder", "suppress",
  "interrupt", "isolate", "freeze", "rot", "underscore", "grind",
]

hedge_verb_phrase = [
  "hold down", "might be",
]


def create_hedge_verb(matcher):
  # pattern: lemma of VERB | example:
  pattern1 = [
    {"POS": "VERB", "LEMMA": {"IN": hedge_verb}},
  ]
  matcher.add("HEDGE_VERB", [pattern1])
  return matcher


def create_hedge_verb_negation(matcher):
  # pattern: not VERB | example: "not achieve"
  pattern2 = [
    {"LOWER": "not"},
    {"POS": "VERB", "LEMMA": {"IN": hedge_verb}},
  ]
  matcher.add("HEDGE_VERB_NEGATION", [pattern2])
  return matcher


def create_hedge_verb_phrase(nlp, matcher):
  patterns = [nlp(phrase) for phrase in hedge_verb_phrase]
  matcher.add("HEDGE_VERB_PHRASE", patterns)
  return matcher


# Includes also all negation forms of:
# - positive verbs
# - negative verbs
# - certainty verbs
# - doubt verbs
# - emphatic verbs
# - hedge verbs
# - pro verbs
# - con verbs

""" =================================================
Pro
================================================= """

pro_verb = [
  "commit", "support", "endorse", "favor", "pledge", "help", "assist", "approve", "promote", "back", "commend",
  "champion", "bolster", "advocate", "equip", "fund", "agree", "contribute", "offer", "respect", "supply",
  "protect", "defend", "guarantee", "facilitate", "foster", "nurture", "cultivate", "give", "provide", "enable",
  "join", "ally", "benefit", "stand", "collaborate", "sponsor", "partner", "finance", "praise", "allow", "bind",
  "welcome", "invite", "seek", "associate", "cohost", "cooperate", "incentivise", "recommend", "propose", "trust",
  "position", "negotiate", "permit", "pump", "rekindle", "unite", "combine",  "coddle", "further", "safeguard",
  "guard", "invest", "answer", "respond", "carry", "embrace", "insure", "accord", "aid", "find", "applaud",
  "sustain", "reinforce", "advance", "progress", "develop",
]

pro_verb_phrase = [
  "look forward", "stand for", "believe in", "pursuit of"
]


def create_pro_verb(matcher):
  # pattern: lemma of VERB | example:
  pattern1 = [
    {"POS": "VERB", "LEMMA": {"IN": pro_verb}},
  ]
  matcher.add("PRO_VERB", [pattern1])
  return matcher


def create_pro_verb_negation(matcher):
  # pattern: not VERB | example: "not support"
  pattern2 = [
    {"LOWER": "not"},
    {"POS": "VERB", "LEMMA": {"IN": pro_verb}},
  ]
  matcher.add("PRO_VERB_NEGATION", [pattern2])
  return matcher


def create_pro_verb_phrase(nlp, matcher):
  patterns = [nlp(phrase) for phrase in pro_verb_phrase]
  matcher.add("PRO_VERB_PHRASE", patterns)
  return matcher


def create_pro_verb_patterns(matcher):
  # made a commitment
  pattern1 = [
    {"LEMMA": "make"},
    {"LOWER": "a", "OP": "?"},
    {"LEMMA": "commitment"}
  ]
  """"# pursuit of
  pattern3 = [
    {"LEMMA": "pursuit of"}
  ]"""
  # have (an) interest
  pattern2 = [
    {"POS": "VERB", "LEMMA": "have"},
    {"LOWER": {"IN": ["a", "an", "the"]}, "OP": "?"},
    {"POS": "NOUN", "LEMMA": "interest"}
  ]
  # recognize the importance
  pattern3 = [
    {"LEMMA": "recognize"},
    {"LOWER": "the", "OP": "?"},
    {"POS": "NOUN", "LEMMA": "importance"}
  ]
  # fight for
  pattern4 = [
    {"POS": "VERB", "LEMMA": "fight"},
    {"LOWER": "for"}
  ]
  matcher.add("PRO_MAKE_COMMITMENT", [pattern1])
  # matcher.add("PRO_PURSUIT_OF", [pattern3])
  matcher.add("PRO_HAVE_INTEREST", [pattern2])
  matcher.add("PRO_RECOGNIZE_IMPORTANCE", [pattern3])
  matcher.add("PRO_FIGHT_FOR", [pattern4])
  return matcher


""" =================================================
Con
================================================= """

con_verb = [
  "fight", "oppose", "defy", "disagree", "resist", "object", "dissent", "refute", "reject", "dispute", "contradict",
  "counter", "combat", "attack", "struggle", "blame", "denounce", "battle", "destruct", "stop", "finish", "end",
  "disapprove", "accuse", "condemn", "tackle", "clash", "confront", "strike", "face", "refuse", "interfere",
  "buck", "prevent", "disallow", "argue", "arrest", "beat", "deplore", "muzzle", "prosecute", "haunt", "press",
  "eliminate", "outlaw", "pit", "punish", "shut", "suspend", "defeat", "devote", "dedicate", "control", "regulate",
  "discontinue", "exclude", "affect", "wage", "threaten", "deal", "limit", "curtail", "deny",
]

con_verb_phrase = [
  "cannot", "can not", "will not", "won't", "do not", "break down", "take on",
  "spark concern",
]


def create_con_verb(matcher):
  # pattern: lemma of VERB | example:
  pattern1 = [
    {"POS": "VERB", "LEMMA": {"IN": con_verb}},
  ]
  matcher.add("CON_VERB", [pattern1])
  return matcher


def create_con_verb_negation(matcher):
  # pattern: not VERB | example: "not oppose"
  pattern2 = [
    {"LOWER": "not"},
    {"POS": "VERB", "LEMMA": {"IN": con_verb}},
  ]
  matcher.add("CON_VERB_NEGATION", [pattern2])
  return matcher


def create_con_verb_phrase(nlp, matcher):
  patterns = [nlp(phrase) for phrase in con_verb_phrase]
  matcher.add("CON_VERB_PHRASE", patterns)
  return matcher


def create_con_verb_patterns(matcher):
  # Pattern: concern I/we have | example: "concern I have"
  """pattern1 = [
    {"LEMMA": "concern"},
    {"LOWER": {"IN": ["i", "we"]}},
    {"POS": "VERB"},
  ]"""
  # VERB concern | example: "raised concern", "sparked (a) concern", "strengthen the opposition"
  pattern2 = [
    {"POS": "VERB"},
    {"LOWER": {"IN": ["a", "an", "the", "more of a"]}, "OP": "?"},
    {"POS": "NOUN", "LEMMA": {"IN": ["concern", "opposition", "disagreement", "threat"]}},
  ]
  # fight/struggle/vote (it) against
  pattern3 = [
    {"POS": "VERB"},
    {"POS": "PRON", "OP": "?"},
    {"LOWER": "against"}
  ]
  # fight (BE) against = fighting against | fight/struggle/effort/aggression/war/prejudices against
  pattern4 = [
    {"POS": "NOUN", "LEMMA": {"IN": ["fight", "struggle", "effort", "aggression", "war", "prejudice", "sanction"]}},
    {"LEMMA": "be", "OP": "?"},
    {"LOWER": "against"},
  ]
  # war on
  """pattern5 = [
    {"LOWER": "war on"},
  ]"""
  # must/should not/never be/continue
  """pattern6 = [
    {"LEMMA": {"IN": ["must", "should"]}},
    {"LOWER": {"IN": ["not", "never", "no"]}},
    {"POS": "VERB"}
  ]"""
  """#
  pattern7 = [
    {"POS": "AUX"},
    {"POS": "VERB", "LEMMA": "fight"},
    {"POS": "NOUN"}
  ]"""
  # must (ADV) fight | example: "must actively fight"
  """pattern8 = [
    {"LEMMA": {"IN": ["must", "should"]}},
    {"POS": "ADV", "OP": "?"},
    {"POS": "VERB", "LEMMA": {"IN": ["fight"]}},
  ]"""
  # fight poverty
  pattern9 = [
    {"POS": "VERB", "LEMMA": {"IN": ["fight", "struggle", "battle", "combat"]}},
    {"POS": "NOUN"}
  ]
  # matcher.add("PRO_CONCERN_VERB", [pattern1])
  matcher.add("PRO_VERB_CONCERN", [pattern2])
  matcher.add("PRO_VERB_AGAINST", [pattern3])
  matcher.add("PRO_NOUN_AGAINST", [pattern4])
  # matcher.add("PRO_WAR_ON", [pattern5])
  # matcher.add("PRO_MUST_NOT_BE", [pattern6]) **
  # matcher.add("PRO_HAVE_FIGHT", [pattern7])
  # matcher.add("PRO_MUST_FIGHT", [pattern8])
  matcher.add("PRO_FIGHT_NOUN", [pattern9])
  return matcher


"""--------------------------------------------------------------------------------------"""

verbs = [
  "employ", "decide", "deliver", "design", "desire", "discuss", "elect", "explore", "export", "involve",
  "act", "base", "replace", "proceed", "plan", "speak", "rule", "add", "afford", "appeal",
  "examine", "track", "want", "belong", "remove", "compete", "spill", "throw", "launch", "listen", "live",
  "look", "make", "manage", "map", "mark", "match", "meet", "move", "obtain", "operate", "play", "own",
  "purchase", "recruit", "reflect", "relate", "represent", "return", "run", "see", "share", "shift",
  "start", "study", "submit", "treat", "use", "view", "vote", "worship", "wrap", "distribute", "ask",
  "await", "be", "breathe", "call", "change", "claim", "come", "command", "conduct", "consider", "consult",
  "cross", "culminate", "date", "describe", "do", "drink", "eat", "exist",
  "experience", "fan", "fly", "function", "get", "go", "happen", "harbor", "have", "head", "hire", "hold",
  "house", "import", "last", "lay", "layer", "leave", "lend", "let", "lie", "march", "mean", "mention", "mount",
  "necessitate", "need", "occur", "practice", "predict", "outpace", "outsource", "pass", "put", "ramp", "range",
  "read", "reap", "reapply", "receive", "refer", "regard", "remain", "remember", "request", "require", "reside",
  "resolve", "result", "resume", "retire", "root", "say", "seal", "send", "set", "sign", "sit", "spread", "stay",
  "step", "substitute", "surround", "talk", "tax", "tell", "test", "tip", "total", "touch", "trade", "turn",
  "weather", "witness",

  "gather",
]
