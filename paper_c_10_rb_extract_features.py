# from pprint import pprint

import spacy
from tqdm import tqdm
from afinn import Afinn
# from textblob import TextBlob

from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file
from lib.semantic_frames import api_get_frames, encode_sentence_frames

# Load datasets
dataset_training = load_jsonl_file("shared_data/dataset_3_1_training.jsonl")
dataset_validation = load_jsonl_file("shared_data/dataset_3_2_validation.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_3_3_test.jsonl")

# Dataset not seen by the model, used for testing
dataset_for_test = load_jsonl_file("datasets/3/bootstrap_1/dataset_3_7_test_anonym.jsonl")


output_features = "shared_data/dataset_3_10_features.jsonl"

# Empty JSONL file
empty_json_file(output_features)

# Join datasets
dataset = dataset_training + dataset_validation + dataset_test + dataset_for_test

# oppose_sentences = [datapoint for datapoint in dataset if datapoint["label"] == "oppose"]
# support_sentences = [datapoint for datapoint in dataset if datapoint["label"] == "support"]

# Initialize spacy model
nlp = spacy.load("en_core_web_trf")

# Initialize Afinn sentiment analyzer
afinn = Afinn()


def measure_sentence_length(_doc):
  return len([token for token in doc])


def measure_word_length(_doc):
  return [len(token.text) for token in _doc if not token.is_punct]


def measure_sentence_complexity(_doc):
  return len([token for token in _doc if token.dep_ in ['advcl', 'relcl', 'acl', 'ccomp', 'xcomp']])


def measure_passive_voice_use(_doc):
  return len([tok for tok in _doc if tok.dep_ == 'auxpass'])


def measure_nominalization_use(_doc):
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
  return len([token for token in _doc if token.pos_ == 'NOUN'
                                    and any(token.text.lower().endswith(suffix)
                                            for suffix in nominalization_suffixes)])


def measure_personal_pronoun_use(_doc):
  personal_pronouns = ['i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it',
                       'its', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs']
  # Make a list of the number of personal pronouns in each sentence
  return len([token for token in _doc if token.text.lower() in personal_pronouns])


def measure_lexical_density(_doc):
  return len([token for token in _doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']])


def measure_modal_verbs_use(_doc):
  return len([token for token in _doc if token.tag_ == 'MD'])


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
  # Make a list of the number of discourse markers per sentence
  return sum(token.lemma_.lower() in discourse_markers for token in _doc)


def measure_negation_use(_doc):
  return len([token for token in _doc if token.dep_ == "neg"])


def measure_adjective_polarity(_doc, pole):
  # Get the polarity words
  polarity_words = []
  for token in _doc:
    # Get sentiment score
    score = afinn.score(token.text)
    # blob = TextBlob(token.text)
    # score = blob.sentiment.polarity
    if pole == "pos":
      if token.pos_ == "ADJ" and token.ent_type_ == "" and score > 0:
        polarity_words.append(score)
        # print(token.text, score)
    elif pole == "neg":
      if token.pos_ == "ADJ" and token.ent_type_ == "" and score < 0:
        polarity_words.append(score)
        # print(token.text, score)
  return sum(polarity_words)


def measure_adverb_polarity(_doc, pole):
  # Get the polarity words
  polarity_words = []
  for token in _doc:
    # Get sentiment score
    score = afinn.score(token.text)
    # blob = TextBlob(token.text)
    # score = blob.sentiment.polarity
    if pole == "pos":
      if token.pos_ == "ADV" and token.ent_type_ == "" and score > 0:
        polarity_words.append(score)
    elif pole == "neg":
      if token.pos_ == "ADV" and token.ent_type_ == "" and score < 0:
        polarity_words.append(score)
  return sum(polarity_words)


def measure_verb_polarity(_doc, pole):
  # Get the polarity words
  polarity_words = []
  for token in _doc:
    # Get sentiment score
    score = afinn.score(token.text)
    # blob = TextBlob(token.text)
    # score = blob.sentiment.polarity
    if pole == "pos":
      if token.pos_ == "VERB" and token.ent_type_ == "" and score > 0:
        polarity_words.append(score)
    elif pole == "neg":
      if token.pos_ == "VERB" and token.ent_type_ == "" and score < 0:
        polarity_words.append(score)
  return sum(polarity_words)


print("Extracting Stance features from dataset...")
for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  present_frames_response = api_get_frames(datapoint["text"], "localhost", "5001", "all")
  encoded_frames = encode_sentence_frames(present_frames_response)

  doc = nlp(datapoint["text"])
  label = datapoint["label"]

  neg_adj_polarity = measure_adjective_polarity(doc, "neg")
  pos_adj_polarity = measure_adjective_polarity(doc, "pos")

  row = {
    "text": datapoint["text"],
    "label": label,
    "id": datapoint["id"],
    "sentence_length": measure_sentence_length(doc),
    "word_length": measure_word_length(doc),
    "sentence_complexity": measure_sentence_complexity(doc),
    "passive_voice_use": measure_passive_voice_use(doc),
    "personal_pronouns": measure_personal_pronoun_use(doc),
    "nominalization_use": measure_nominalization_use(doc),
    "lexical_density": measure_lexical_density(doc),
    "modal_verbs_use": measure_modal_verbs_use(doc),
    "discourse_markers_use": measure_discourse_markers_use(doc),
    "negation_use": measure_negation_use(doc),
    "pos_adj_polarity": pos_adj_polarity,
    "neg_adj_polarity": neg_adj_polarity,
    "pos_adv_polarity": measure_adverb_polarity(doc, "pos"),
    "neg_adv_polarity": measure_adverb_polarity(doc, "neg"),
    "pos_verb_polarity": measure_verb_polarity(doc, "pos"),
    "neg_verb_polarity": measure_verb_polarity(doc, "neg"),
    "semantic_frames": encoded_frames
  }
  save_row_to_jsonl_file(row, output_features)

print("Finished!")
