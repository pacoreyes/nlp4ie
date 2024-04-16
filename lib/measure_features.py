
def measure_sentence_length(_sentences):
  # Make a list of the length of all sentences, excluding punctuation
  return [len([token for token in sent if token.is_alpha]) for sent in _sentences]


# Feature 2: Word length, a list of the length of all words, excluding punctuation
def measure_word_length(_sentences):
  # List the length of all words in these sentences
  return [len(token) for sent in _sentences for token in sent if token.is_alpha]


# Feature 3: Sentence complexity, a list of the number of adverbial clauses found in each sentence
def measure_sentence_complexity(_sentences):
  # Make a list of the number of adverbial clauses found in each sentence
  return [len([token for token in sent if token.dep_ in
               ['advcl', 'relcl', 'acl', 'ccomp', 'xcomp']]) for sent in _sentences]


# Feature 4: Personal pronoun use, a list of the number of personal pronouns found in each sentence
def measure_personal_pronoun_use(_sentences):
  # List of personal pronouns in English
  personal_pronouns = ['i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it',
                       'its', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs']
  # Make a list of the number of personal pronouns in each sentence
  personal_pronoun_use_count = [len([token for token in sent if token.text.lower() in personal_pronouns]) for sent in
                                _sentences]
  return personal_pronoun_use_count


# Feature 5: Passive voice use, a list of the number of passive voice occurrences found in each sentence
def measure_passive_voice_use(_sentences):
  # Make a list of the number of passive voice occurrences in each sentence
  return [len([tok for tok in sent if tok.dep_ == 'auxpass']) for sent in _sentences]


# Feature 6: Nominalization use, a list of the number of nominalizations found in each sentence
def measure_nominalization_use(_sentences):
  # Define common nominalization suffixes
  nominalization_suffixes = ('tion', 'ment', 'ness', 'ity', 'age',
                             'ance', 'ence', 'hood', 'ship', 'ty',
                             'cy', 'al', 'ure', 'er', 'ing',
                             'sion', 'ation', 'ibility', 'ana', 'acy',
                             'ama', 'ant', 'dom', 'edge', 'ee',
                             'eer', 'ery', 'ese', 'ess', 'ette',
                             'fest', 'ful', 'iac', 'ian', 'ie',
                             'ion', 'ism', 'ist', 'ite', 'itude',
                             'ium', 'let', 'ling', 'man', 'woman',
                             'mania', 'or', 'th', 'tude')
  # Make a list of the number of nominalizations in each sentence
  return [len([token for token in sent if token.pos_ == 'NOUN'
                                    and any(token.text.lower().endswith(suffix)
                                            for suffix in nominalization_suffixes)]) for sent in _sentences]


# Feature 7: Lexical density, a list of the number of lexical words found in each sentence
def measure_lexical_density(_sentences):
  # Make a list of the number of lexical words in each sentence
  return [len([token for token in sent if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']])
                           for sent in _sentences]


# Feature 8: Interjection use, a list of the number of interjections found in each sentence
def measure_interjections_use(_sentences):
  # Make a list of the number of interjections in each sentence
  return [len([token for token in sent if token.pos_ == 'INTJ']) for sent in _sentences]


# Feature 9: Modal verb use, a list of the number of modal verbs found in each sentence
def measure_modal_verbs_use(_sentences):
  # Make a list of the number of modal verbs in each sentence
  return [len([token for token in sent if token.tag_ == 'MD']) for sent in _sentences]


# Feature 10: Discourse markers use, a list of the number of discourse markers found in each sentence
def measure_discourse_markers_use(_sentences):
  # Predefined list of common discourse markers
  discourse_markers = [
    'accordingly', 'actually', 'after', 'after all', 'afterward', 'afterwards', 'all in all', 'also', 'although',
    'always', 'anyway', 'apparently', 'as', 'as a matter of fact', 'as a result', 'as if', 'as long as', 'as soon as',
    'as though', 'as well as', 'assuming that', 'at first', 'at last', 'at least', 'at present', 'at the same time',
    'basically', 'because', 'before', 'being that', 'besides', 'but', 'by the time', 'by the way', 'chiefly', 'clearly',
    'commonly', 'consequently', 'considering that', 'despite', 'due to', 'during', 'either', 'especially',
    'essentially', 'even if', 'even so', 'even supposing', 'even though', 'eventually', 'every', 'evidently', 'except',
    'except for', 'except that', 'exclusively', 'exclusively', 'finally', 'first', 'first of all', 'for example',
    'for instance', 'for one thing', 'for the most part', 'for the time being', 'for this purpose', 'for this reason',
    'formerly', 'forthwith', 'fortunately', 'frankly', 'frequently', 'further', 'furthermore', 'generally',
    'given that', 'granting that', 'hence', 'henceforth', 'honestly', 'however', 'if only', 'immediately',
    'in addition', 'in any case', 'in brief', 'in case', 'in conclusion', 'in contrast', 'in fact', 'in general',
    'in my opinion', 'in order that', 'in order to', 'in other words', 'in particular', 'in short', 'in spite of',
    'in sum', 'in summary', 'in that case', 'in the beginning', 'in the end', 'in the first place', 'in the meantime',
    'in the meanwhile', 'in the same way', 'in the second place', 'in the third place', 'in this case', 'in truth',
    'in view of', 'incidentally', 'including', 'indeed', 'individually', 'initially', 'instantly', 'instead',
    'interestingly', 'just', 'largely', 'last', 'lastly', 'lately', 'later', 'lest', 'like', 'likewise', 'mainly',
    'markedly', 'meanwhile', 'merely', 'moreover', 'most', 'most importantly', 'mostly', 'much namely', 'naturally',
    'neither', 'never', 'nevertheless', 'next', 'nonetheless', 'nor', 'normally', 'not only', 'not just',
    'not to mention',
    'notably', 'notwithstanding', 'now', 'now that', 'nowadays', 'obviously', 'occasionally', 'of course', 'often',
    'on balance', 'on condition that', 'on the contrary', 'on the other hand', 'on the whole', 'once', 'only',
    'ordinarily', 'originally', 'otherwise', 'overall', 'particularly', 'permanently', 'personally', 'plainly', 'plus',
    'presently', 'presently', 'presumably', 'previously', 'primarily', 'privately', 'probably', 'promptly', 'properly',
    'provided that', 'publicly', 'quickly', 'rarely', 'rather', 'readily', 'really', 'recently', 'regardless',
    'regularly', 'relatively', 'remarkably', 'respectively', 'secondly', 'seeing that', 'seemingly', 'seldom',
    'separately', 'seriously', 'seriously', 'shortly', 'shortly', 'significantly', 'similarly', 'similarly', 'simply',
    'since', 'slightly', 'slowly', 'so', 'so far', 'so long as', 'so that', 'sometimes', 'soon', 'specifically',
    'still', 'straightaway', 'strangely', 'strongly', 'subsequently', 'successfully', 'such as', 'suddenly',
    'supposedly', 'supposing', 'surely', 'surprisingly', 'technically', 'temporarily', 'that is way', 'then',
    'thereafter', 'thereby', 'therefore', 'thereupon', 'these days', 'thirdly', 'though', 'thus', 'till',
    'to begin with', 'to conclude', 'to date', 'to illustrate', 'to sum up', 'to tell the truth', 'to that end',
    'to this end', 'to this end', 'typically', 'ultimately', 'undoubtedly', 'unfortunately', 'unless', 'unlike',
    'until', 'until now', 'up to', 'up to now', 'up to the present time', 'usually', 'utterly', 'virtually', 'well',
    'what is more', 'whatever', 'when', 'whenever', 'whereas', 'whereby', 'whereupon', 'wherever', 'whether',
    'whichever', 'while', 'whilst', 'whoever', 'with this in mind', 'with this intention', 'yet'
  ]
  # Make a list of the number of discourse markers per sentence
  return [sum(token.lemma_.lower() in discourse_markers for token in sent) for sent in _sentences]
