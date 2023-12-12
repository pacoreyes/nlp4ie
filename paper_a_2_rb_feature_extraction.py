"""
This script extract linguistic features from the texts in the corpus.

The features are:
- Sentence length, a list of the length of all sentences, excluding punctuation
- Word length, a list of the length of all words, excluding punctuation
- Sentence complexity, a list of the number of adverbial clauses found in each sentence
- Personal pronoun usage, a list of the number of personal pronouns found in each sentence
- Passive voice usage, a list of the number of passive voice found in each sentence
- Nominalization usage, a list of the number of nominalization found in each sentence
- Lexical density, a list of the number of lexical words found in each sentence
- Interjection usage, a list of the number of interjections found in each sentence
- Modal verb usage, a list of the number of modal verbs found in each sentence
- Discourse markers usage, a list of the number of discourse markers found in each sentence
"""
import spacy

from utils.utils import save_row_to_jsonl_file, load_jsonl_file, empty_json_file
from utils.linguistic_utils import check_if_has_one_word_or_more

# load transformer model
nlp = spacy.load("en_core_web_trf")

# Initialize a JSONL file for the dataset
# empty_json_file("shared_data/paper_a_2_feature_extraction.jsonl")
empty_json_file("shared_data/dataset_1_2_features.jsonl")


"""All functions evaluate based on sentence occurrences, not token or text occurrences."""


# Feature 1: Sentence length, a list of the length of all sentences, excluding punctuation
def measure_sentence_length(_doc):
  # Get a list of sentences filtering only those with one word
  sentences = list(sent for sent in _doc.sents if check_if_has_one_word_or_more(sent))
  # Make a list of the length of all sentences, excluding punctuation
  sentence_length_count = [len([token for token in sent if token.is_alpha]) for sent in sentences]
  return sentence_length_count


# Feature 2: Word length, a list of the length of all words, excluding punctuation
def measure_word_length(_doc):
  # Get a list of sentences filtering only those with one word
  sentences = list(sent for sent in _doc.sents if check_if_has_one_word_or_more(sent))
  # Make a list of the length of all words in all sentences, excluding punctuation
  word_length_count = [len(token) for sent in sentences for token in sent if token.is_alpha]
  return word_length_count


# Feature 3: Sentence complexity, a list of the number of adverbial clauses found in each sentence
def measure_sentence_complexity(_doc):
  # Get a list of sentences filtering only those with one word
  sentences = list(sent for sent in _doc.sents if check_if_has_one_word_or_more(sent))
  # Make a list of the number of adverbial clauses found in each sentence
  sentence_complexity_count = [len([token for token in sent if token.dep_ in
                                    ['advcl', 'relcl', 'acl', 'ccomp', 'xcomp']]) for sent in sentences]
  return sentence_complexity_count


"""def measure_pronoun_use(doc):
  # Get a list of sentences filtering only those with one word
  sentences = list(sent for sent in doc.sents if check_if_has_one_word_or_more(sent))
  # Make a list of the number of pronouns in each sentence
  pronoun_use_count = [len([token for token in sent if token.pos_ == 'PRON']) for sent in sentences]
  return pronoun_use_count"""


# Feature 4: Personal pronoun use, a list of the number of personal pronouns found in each sentence
def measure_personal_pronoun_use(_doc):
  # Get a list of sentences filtering only those with one word
  sentences = list(sent for sent in _doc.sents if check_if_has_one_word_or_more(sent))
  # List of personal pronouns in English
  personal_pronouns = ['i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it',
                       'its', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs']
  # Make a list of the number of personal pronouns in each sentence
  personal_pronoun_use_count = [len([token for token in sent if token.text.lower() in personal_pronouns]) for sent in
                                sentences]
  return personal_pronoun_use_count


# Feature 5: Passive voice use, a list of the number of passive voice occurrences found in each sentence
def measure_passive_voice_use(_doc):
  # Get a list of sentences filtering only those with one word
  sentences = list(sent for sent in _doc.sents if check_if_has_one_word_or_more(sent))
  # Make a list of the number of passive voice occurrences in each sentence
  passive_voice_use_count = [len([tok for tok in sent if tok.dep_ == 'auxpass']) for sent in sentences]
  return passive_voice_use_count


# Feature 6: Nominalization use, a list of the number of nominalizations found in each sentence
def measure_nominalization_use(_doc):
  # Get a list of sentences filtering only those with one word
  sentences = list(sent for sent in _doc.sents if check_if_has_one_word_or_more(sent))
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
  nominatilzation_use_count = [len([token for token in sent if token.pos_ == 'NOUN'
                                    and any(token.text.lower().endswith(suffix)
                                            for suffix in nominalization_suffixes)]) for sent in sentences]
  return nominatilzation_use_count


# Feature 7: Lexical density, a list of the number of lexical words found in each sentence
def measure_lexical_density(_doc):
  # Get a list of sentences filtering only those with one word
  sentences = list(sent for sent in _doc.sents if check_if_has_one_word_or_more(sent))
  # Make a list of the number of lexical words in each sentence
  lexical_density_count = [len([token for token in sent if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']])
                           for sent in sentences]
  return lexical_density_count


# Feature 8: Interjection use, a list of the number of interjections found in each sentence
def measure_interjections_use(_doc):
  # Get a list of sentences filtering only those with one word
  sentences = list(sent for sent in _doc.sents if check_if_has_one_word_or_more(sent))
  # Make a list of the number of interjections in each sentence
  interjection_use_count = [len([token for token in sent if token.pos_ == 'INTJ']) for sent in sentences]
  return interjection_use_count


# Feature 9: Modal verb use, a list of the number of modal verbs found in each sentence
def measure_modal_verbs_use(_doc):
  # Get a list of sentences filtering only those with one word
  sentences = list(sent for sent in _doc.sents if check_if_has_one_word_or_more(sent))
  # Make a list of the number of modal verbs in each sentence
  modal_verb_use_count = [len([token for token in sent if token.tag_ == 'MD']) for sent in sentences]
  return modal_verb_use_count


# Feature 10: Discourse markers use, a list of the number of discourse markers found in each sentence
def measure_discourse_markers_use(_doc):
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
  # Get a list of sentences filtering only those with one word
  sentences = list(sent for sent in _doc.sents if check_if_has_one_word_or_more(sent))
  # Make a list of the number of discourse markers per sentence
  discourse_marker_use_count = [sum(token.lemma_.lower() in discourse_markers for token in sent) for sent in sentences]
  return discourse_marker_use_count


corpus = load_jsonl_file("shared_data/dataset_1_1_raw.jsonl")
# corpus = load_jsonl_file("shared_data/paper_a_4_corpus_sliced_dataset.jsonl")

# Container for the morphosyntactic data
ms_data = []

# Create dictionaries for each class of text
text_classes = {i: [] for i in range(3)}
for item in corpus:
  text_classes[item["discourse_type"]].append(item)

for idx, texts_per_class in text_classes.items():
  print("Class: ", idx)

  for idx2, item in enumerate(texts_per_class):
    # convert list of strings to a single string
    text = " ".join(item["text"])
    # text = item["text"]
    # create a spaCy document with the preprocessed text
    doc = nlp(text)

    # measure sentence length
    sentence_lengths = measure_sentence_length(doc)
    # measure word length
    word_lengths = measure_word_length(doc)
    # measure sentence complexity
    sentence_complexity = measure_sentence_complexity(doc)
    # measure pronouns use
    # pronoun_use = measure_pronoun_use(doc)
    # measure personal pronouns use
    personal_pronoun_use = measure_personal_pronoun_use(doc)
    # measure passive voice use
    passive_voice_use = measure_passive_voice_use(doc)
    # measure nominalization use
    nominalization_use = measure_nominalization_use(doc)
    # measure lexical density
    lexical_density = measure_lexical_density(doc)
    # measure interjection use
    interjection_use = measure_interjections_use(doc)
    # measure modal verb use
    modal_verb_use = measure_modal_verbs_use(doc)
    # measure discourse markers use
    discourse_markers_use = measure_discourse_markers_use(doc)

    """def convert_list_of_nums_to_str(list_of_ints):
      str_list = [str(i) for i in list_of_ints]
      return ','.join(str_list)"""

    slots = {
      "id": item["id"],
      "discourse_type": item["discourse_type"],
      # "discourse_type": item["label"],
      "sentence_length": sentence_lengths,
      "word_length": word_lengths,
      "sentence_complexity": sentence_complexity,
      "personal_pronoun_use": personal_pronoun_use,
      "passive_voice_use": passive_voice_use,
      "nominalization_use": nominalization_use,
      "lexical_density": lexical_density,
      "interjection_use": interjection_use,
      "modal_verb_use": modal_verb_use,
      "discourse_markers_use": discourse_markers_use
    }
    # store the slots in a JSONL file
    save_row_to_jsonl_file(slots, "shared_data/dataset_1_2_features.jsonl")
    ms_data.append(slots)

    print(f"{idx2+1}/{len(texts_per_class)} Added datapoint {item['id']} to JSONL file")

  print()
