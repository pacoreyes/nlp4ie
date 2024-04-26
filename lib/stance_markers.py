"""-------------------Positive Affect Features-------------------"""

positive_affect_adj = [
  "amaze",
  "amazing",
  "amused",
  "amusing",
  "appropriate",
  "astonished",
  "astonishing",
  "compel",
  "content",
  "convenient",
  "credible",
  "curious",
  "delighted",
  "delightful",
  "eager",
  "enchanted",
  "fascinate",
  "fascinating",
  "fitting",
  "fortunate",
  "funny",
  "glad",
  "great",
  "happy",
  "hopeful",
  "incredible",
  "indispensable",
  "inevitable",
  "interested",
  "interesting",
  "ironic",
  "jubilant",
  "keen",
  "lucky",
  "merciful",
  "natural",
  "nice",
  "noteworthy",
  "outstanding",
  "overjoyed",
  "please",
  "powerful",
  "predictable",
  "preferable",
  "proper",
  "proof",
  "proud",
  "refresh",
  "relieve",
  "remarkable",
  "satisfy",
  "self-evident",
  "significant",
  "surprise",
  "surprising",
  "thankful",
  "understandable",
  "unexpected"
]

positive_affect_adv = [
  'absolutely', 'actually', 'always', 'amazingly', 'amusingly', 'apparently',
  'appropriately', 'assuredly', 'astonishingly', 'categorically', 'certainly',
  'clearly', 'completely', 'comprehensively', 'conclusively', 'considerably',
  'consistently', 'conspicuously', 'constantly', 'conveniently', 'convincingly',
  'credibly', 'crucially', 'curiously', 'decisively', 'definitely', 'definitively',
  'deservedly', 'distinctively', 'no doubt', 'doubtlessly', 'enchantingly', 'entirely',
  'especially', 'essentially', 'evidently', 'exceptionally', 'exhaustively',
  'extensively', 'extremely', 'firmly', 'forcefully', 'fortunately', 'fully',
  'fundamentally', 'funnily', 'genuinely', 'happily', 'highly', 'honestly',
  'hopefully', 'impressively', 'incredibly', 'indeed', 'indispensably', 'inevitably',
  'interestingly', 'ironically', 'largely', 'luckily', 'mainly', 'manifestly',
  'markedly', 'meaningfully', 'mercifully', 'mostly', 'naturally', 'necessarily',
  'never', 'nevertheless', 'nonetheless', 'notably', 'noticeably', 'obviously',
  'particularly', 'perfectly', 'persuasively', 'plainly', 'precisely', 'predictably',
  'preferably', 'profoundly', 'prominently', 'quite', 'radically', 'really',
  'refreshingly', 'reliably', 'remarkably', 'rightly', 'rigorously', 'safely',
  'securely', 'significantly', 'sizably', 'strikingly', 'successfully', 'surely',
  'surprisingly', 'thankfully', 'thoroughly', 'totally', 'truly', 'unaccountably',
  'unambiguously', 'unarguably', 'unavoidably', 'undeniably', 'understandably',
  'undoubtedly', 'unequivocally', 'unexpectedly', 'uniquely', 'unmistakably',
  'unquestionably', 'vastly', 'vitally'
]

positive_affect_verb = [
  'ache for', 'amaze', 'amuse', 'as i said', 'as i say', 'astonish', 'conclude', 'confirm', 'corroborate', 'delight',
  'demonstrate', 'enhance', 'enjoy', 'establish', 'fancy', 'find that', 'found that', 'highlight', 'hope', 'interest',
  'know', 'like', 'like i say', 'like i said', 'long for', 'love', 'please', 'prefer', 'prove', 'refresh', 'relish',
  'seek', 'show', 'suit', 'surprise', 'thrill', 'uphold', 'want', 'wish', 'yearn'
]

"""-------------------Negative Affect Features-------------------"""

negative_affect_adj = [
  'afraid', 'aggrieve', 'alarm', 'alarming', 'annoyed', 'annoying', 'ashamed', 'concerned', 'confuse', 'depressed',
  'disappointed', 'disappointing', 'disgust', 'disgusting', 'dismay', 'dissatisfy', 'distress', 'distressed',
  'disturb', 'embarrass', 'embarrassing', 'frightened', 'frightening', 'furious', 'hopeless', 'horrible',
  'impatient', 'improper', 'indignant', 'irritate', 'mad', 'odd', 'overwhelm', 'perplex', 'perturbed', 'puzzle',
  'puzzling', 'regretful', 'regrettable', 'sad', 'scare', 'scary', 'shocked', 'silly', 'strange', 'suspicious',
  'terrible', 'tragic', 'unfortunate', 'unhappy', 'unnatural', 'upset', 'upsetting', 'worried', 'worrisome'
]

negative_affect_adv = [
  'alarmingly', 'annoyingly', 'ashamedly', 'depressingly', 'disappointingly', 'disgustingly', 'disturbingly',
  'embarrassedly', 'frighteningly', 'impatiently', 'oddly', 'perplexingly', 'regretfully', 'regrettably', 'sadly',
  'shockingly', 'strangely', 'suspiciously', 'tragically', 'unfortunately', 'unhappily', 'unluckily', 'unnaturally'
]

negative_affect_verb = [
  'aggravate', 'agitate', 'alarms', 'annoy', 'begrudge', 'bother', "can't stand", 'cannot stand', 'confuse',
  'deign', 'despise', 'detest', 'disappoint', 'discourage', 'disgust', 'dislike', 'dismay', 'distress', 'disturb',
  'dread', 'embarrass', 'embarrasse', 'envy', 'fear', 'frighten', 'hate', 'horrify', 'irritate', 'kill', 'loathe',
  'overwhelm', 'pain', 'perplex', 'perplexe', 'perturb', 'puzzle', 'regret', 'resent', 'rub', 'sadden', 'scare',
  'scorn', 'shock', 'slay', 'trouble', 'upset', 'worry'
]


"""-------------------Certainty Lexicons-------------------"""

certainty_adj = [
  'absolute', 'adamant', 'apparent', 'assure', 'categorical', 'certain', 'clear-cut', 'convince',
  'decisive', 'definite', 'determined', 'distinct', 'evident', 'explicit', 'feasible', 'flawless', 'foolproof',
  'guarantee', 'impeccable', 'impossible', 'inconceivable', 'incontestable', 'incontrovertible', 'indisputable',
  'indubitable', 'inevitable', 'inexorable', 'infallible', 'irrefutable', 'irresistible', 'irrevocable',
  'manifest', 'obvious', 'patent', 'plain', 'possible', 'predestine', 'resolute', 'sure', 'surefire', 'true',
  'unambiguous', 'unarguable', 'unassailable', 'unavoidable', 'unconditional', 'undeniable', 'undoubted',
  'unimpeachable', 'unmistakable', 'unqualified', 'unquestionable', 'untrue', 'unwavering', 'unyielding', 'well-known'
]

certainty_adv = [
  "actually", "admittedly", "assuredly", "avowedly", "certainly", "clearly", "decisively", "decidedly", "definitely",
  "evidently", "in actuality", "incontestably", "incontrovertibly", "indeed", "indisputably", "indubitably",
  "in fact", "in reality", "in certainty", "with certainty", "irrefutably", "manifestly", "obviously", "of course",
  "patently", "plainly", "reality", "surely", "unambiguously", "unarguably", "undeniably", "undoubtedly",
  "unmistakably", "unquestionably", "with certainty", "without doubt", "without question",
]

certainty_adv_to_be = [
  "certain", "clear", "realistic", "be sure"
]

certainty_verbs = [
  "ascertain", "calculate", "conclude", "deduce", "demonstrate", "determine", "discern", "establish",
  "know", "note", "perceive", "prove", "realize"
]

emphatics = [
  "a lot", "for sure", "just", "really", "more", "most"
]

predictive_modals = [
  "will", "would", "shall"
]

"""-------------------Doubt Lexicons-------------------"""

doubt_adj = [
  'allege', 'arguable', 'conceivable', 'disputable', 'doubtful', 'dubious', 'imaginable', 'improbable', 'indefinite',
  'likely', 'possible', 'presumable', 'probable', 'questionable', 'repute', 'suppose', 'uncertain', 'unclear',
  'unlikely', 'unsure'
]

doubt_adv = [
  'allegedly', 'apparently', 'arguably', 'conceivably', 'formally', 'hypothetically', 'ideally', 'likely',
  'officially', 'ostensibly', 'outwardly', 'perchance', 'perhaps', 'possibly', 'presumably', 'purportedly',
  'reportedly', 'reputedly', 'seemingly', 'superficially', 'supposedly', 'technically', 'theoretically'
]

doubt_verbs = [
  'appear', 'assume', 'conjecture', 'disbelieve', 'doubt', 'estimate', 'expect', 'feel', 'gather', 'guess',
  'hypothesise', 'hypothesize', 'imagine', 'imply', 'indicate', 'infer', 'postulate', 'presume', 'seem', 'sense',
  'speculate', 'suppose', 'surmise', 'suspect', 'think', 'wonder if'
]

hedges = [
  "almost", "at about", "kind of", "maybe", "more or less", "something like", "sort of"
]

possibility_modals = [
  "can", "may", "might", "could"
]

necessity_modals = [
  "ought", "should", "must"
]
