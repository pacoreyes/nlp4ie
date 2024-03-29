from pprint import pprint
import random
# from collections import Counter
import statistics

import spacy
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# from db import firestore_db
from lib.utils import load_jsonl_file, save_json_file, load_json_file
# from lib.linguistic_utils import check_if_has_one_word_or_more
from lib.measure_features import (
  measure_sentence_length, measure_word_length, measure_sentence_complexity, measure_passive_voice_use,
  measure_lexical_density, measure_nominalization_use, measure_personal_pronoun_use, measure_interjections_use,
  measure_modal_verbs_use, measure_discourse_markers_use)
from lib.ner_processing import custom_anonymize_text

if spacy.prefer_gpu():
    print("spaCy is using GPU!")
else:
    print("GPU not available, spaCy is using CPU instead.")

nlp = spacy.load("en_core_web_trf")

features_map = {
  "sentence_length": "Sentence Length",
  "word_length": "Word Length",
  "sentence_complexity": "Sentence Complexity",
  "passive_voice_freq": "Passive Voice Freq",
  "lexical_freq": "Lexical Word Freq",
  "nominalization_freq": "Nominalization Freq",
  "personal_pronoun_freq": "Personal Pronoun Freq",
  "interjection_freq": "Interjection Freq",
  "modal_verb_freq": "Modal Verb Freq",
  "discourse_marker_freq": "Discourse Marker Freq"
}


def save_txt_file(content, file_path):
  with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)


def read_text_from_file(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
  return content


neutral_verbs = [
  "run", "observe", "manage", "organize", "develop", "maintain", "compare", "describe",
  "report", "examine", "determine", "record", "address", "identify", "access", "create",
  "follow", "measure", "discuss", "review", "conduct", "analyze", "present", "apply",
  "utilize", "suggest", "consider", "calculate", "synthesize",
  "quantify", "evaluate", "interpret", "formulate", "validate", "investigate", "correlate", "optimize",
  "integrate", "specify", "simulate", "modify", "fabricate", "illustrate", "derive", "adapt", "enumerate",
  "diagnose", "explore", "implement", "innovate", "elucidate", "catalog", "classify", "deconstruct",
  "replicate", "benchmark", "standardize", "characterize"
]

neutral_verbs = list(set(neutral_verbs))

neutral_adverbs = [
  "objectively", "clearly", "publicly", "officially", "directly", "generally", "globally",
  "nationally", "locally", "effectively", "efficiently", "equally", "precisely", "simply",
  "specifically", "historically", "legally", "politically", "practically", "formally",
  "objectively", "clearly", "publicly", "officially", "directly", "generally", "globally",
  "nationally", "locally", "effectively", "efficiently", "equally", "precisely", "simply",
  "specifically", "historically", "legally", "politically", "practically", "formally",
  "methodically", "analytically", "systematically", "empirically", "theoretically",
  "technically", "experimentally", "quantitatively", "qualitatively", "innovatively",
  "strategically", "academically", "conceptually", "environmentally", "sustainably",
  "ethically", "culturally", "intuitively", "diagnostically", "holistically",
  "statistically", "comprehensively", "dynamically", "virtually", "procedurally"
]

neutral_adverbs = list(set(neutral_adverbs))

neutral_nouns = [
  'candy', 'road', 'table', 'window', 'book', 'chair', 'tree', 'water', 'light', 'phone', 'paper', 'car',
  'glass', 'music', 'art', 'food', 'house', 'garden', 'computer', 'river', 'mountain', 'city', 'village',
  'cloud', 'rain', 'snow', 'sand', 'forest', 'beach', 'bridge', 'candy', 'road', 'table', 'window', 'book',
  'chair', 'tree', 'water', 'light', 'phone', 'paper', 'car',
  'glass', 'music', 'art', 'food', 'house', 'garden', 'computer', 'river', 'mountain', 'city', 'village',
  'cloud', 'rain', 'snow', 'sand', 'forest', 'beach', 'bridge',
  'lake', 'island', 'field', 'moon', 'star', 'planet', 'ocean', 'flower', 'rock', 'hill', 'valley', 'tree',
  'grass', 'leaf', 'bird', 'fish', 'animal', 'sky', 'sun', 'tea', 'coffee', 'cake', 'bicycle', 'train',
  'airplane', 'ship', 'pen', 'pencil', 'notebook', 'basket', 'fence', 'path', 'stream', 'cave', 'cliff', 'desert'
]

neutral_nouns = list(set(neutral_nouns))

bias_words = [
  'family', 'well', 'go', 'year', 'look', 'like', 'life', 'time', 'good', 'america', 'ask', 'come',
  'government', 'man', 'state', 'states', 'want', 'president', 'problem', 'american', 'kind', 'tell',
  'think', 'try', 'world', 'today', 'country', 'child', 'fact', 'people', 'day', 'believe', 'need', 'thing',
  'united', 'great', 'say', 'important', 'nation', 'help', 'deal', 'get', 'lot', 'new', 'right', 'school',
  'entity', 'mean', 'know', 'million', 'thank', 'take', 'issue', 'americans', 'way', 'congress', 'sure',
  'let', 'talk', 'question', 'work', 'happen', 'job'
]

# ref_coll_adversarial = firestore_db.collection("adversarial")

dataset_test = load_jsonl_file("shared_data/dataset_1_6_1b_test.jsonl")

bias_words = set(bias_words)
# print(neutral_nouns)

IGNORE_LIST = [5557, 5742, 5847, 3381, 3405, 3701]
SETUP = False

# Divide the dataset into two classes
speech_class = [datapoint for datapoint in dataset_test
                if datapoint["label"] == "monologic" and datapoint["id"] not in IGNORE_LIST]
interview_class = [datapoint for datapoint in dataset_test
                   if datapoint["label"] == "dialogic" and datapoint["id"] not in IGNORE_LIST]

speech_class = speech_class[:30]
interview_class = interview_class[:30]

dataset = speech_class + interview_class

DATASET_PATH = "shared_data/adversarial"

if SETUP:
  for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):

    text = datapoint["text"]
    label = datapoint["label"]

    doc = nlp(text)
    sentences = list(sent for sent in doc.sents)

    # Measure features
    features = {
      "sentence_length": measure_sentence_length(sentences),
      "word_length": measure_word_length(sentences),
      "sentence_complexity": measure_sentence_complexity(sentences),
      "passive_voice_freq": measure_passive_voice_use(sentences),
      "lexical_word_freq": measure_lexical_density(sentences),
      "nominalization_freq": measure_nominalization_use(sentences),
      "personal_pronoun_freq": measure_personal_pronoun_use(sentences),
      "interjections_freq": measure_interjections_use(sentences),
      "modal_verbs_freq": measure_modal_verbs_use(sentences),
      "discourse_marker_freq": measure_discourse_markers_use(sentences)
    }

    # Calculate mean for each feature
    for feature, values in features.items():
      features[feature] = statistics.mean(values)

    slots = {
      "id": datapoint["id"],
      "text": text,
      "label": label,
      "features": features
    }

    if label == "monologic":
      # Save text in a TXT file
      save_txt_file("", f"{DATASET_PATH}/speech/{datapoint['id']}.txt")
      save_json_file(slots, f"{DATASET_PATH}/speech/{datapoint['id']}.json")
    if label == "dialogic":
      # Save text in a TXT file
      save_txt_file("", f"{DATASET_PATH}/interview/{datapoint['id']}.txt")
      save_json_file(slots, f"{DATASET_PATH}/interview/{datapoint['id']}.json")

# ------------------------------
else:
  text_id = "5531"
  text_class = "speech"

  source_data = load_json_file(f"{DATASET_PATH}/{text_class}/{text_id}.json")
  manipulated_text = read_text_from_file(f"{DATASET_PATH}/{text_class}/{text_id}.txt")

  # text = source_data["text"]

  pprint(manipulated_text)

  modified_tokens = []

  doc = nlp(manipulated_text)

  # Replace bias words with neutral words
  for token in doc:
    token_replaced = False
    # print(token.text, token.pos_, token.tag_)
    if token.lemma_ in bias_words and not token.is_stop and not token.is_punct:
      if token.pos_ == "NOUN":
        random_neutral_noun = random.choice(neutral_nouns)
        modified_tokens.append(random_neutral_noun + token.whitespace_)
        token_replaced = True
        # print(random_neutral_noun)
      elif token.pos_ == "VERB":
        random_neutral_verb = random.choice(neutral_verbs)
        modified_tokens.append(random_neutral_verb + token.whitespace_)
        token_replaced = True
      elif token.pos_ == "ADV":
        random_neutral_adverb = random.choice(neutral_adverbs)
        modified_tokens.append(random_neutral_adverb + token.whitespace_)
        token_replaced = True
    if not token_replaced:
      modified_tokens.append(token.text_with_ws)

  modified_text = "".join(modified_tokens)

  # Anonymize the text
  modified_text = custom_anonymize_text(modified_text, nlp)

  doc = nlp(modified_text)
  sentences = list(sent for sent in doc.sents)

  # Measure features
  features = {
    "sentence_length": measure_sentence_length(sentences),
    "word_length": measure_word_length(sentences),
    "sentence_complexity": measure_sentence_complexity(sentences),
    "passive_voice_freq": measure_passive_voice_use(sentences),
    "lexical_word_freq": measure_lexical_density(sentences),
    "nominalization_freq": measure_nominalization_use(sentences),
    "personal_pronoun_freq": measure_personal_pronoun_use(sentences),
    "interjections_freq": measure_interjections_use(sentences),
    "modal_verbs_freq": measure_modal_verbs_use(sentences),
    "discourse_marker_freq": measure_discourse_markers_use(sentences)
  }
  # Calculate mean for each feature
  for feature, values in features.items():
    features[feature] = statistics.mean(values)

  source_data["manipulated_text"] = modified_text
  source_data["manipulated_features"] = features

  save_json_file(source_data, f"{DATASET_PATH}/{text_class}/{text_id}.json")

  print()
  pprint(modified_text)
  print(f"Class: {text_class}")

  pprint(source_data["features"])
  pprint(source_data["manipulated_features"])

  """# Use min-max scaling to normalize the features
  combined_data = pd.DataFrame([source_data["features"], source_data["manipulated_features"]])

  scaler = MinMaxScaler()
  combined_data_scaled = scaler.fit_transform(combined_data)"""

  before_manipulation = source_data["features"]
  after_manipulation = source_data["manipulated_features"]

  features = list(before_manipulation.keys())
  feature_names = list(features_map.values())

  before_values = [before_manipulation[feature] for feature in features]
  after_values = [after_manipulation[feature] for feature in features]

  # Plotting
  x = range(len(features))
  plt.figure(figsize=(12, 8))
  plt.bar(x, before_values, width=0.4, label='Before Treatment', color='blue', align='center')
  plt.bar(x, after_values, width=0.4, label='After Treatment', color='red', align='edge')
  plt.xlabel('Feature')
  plt.ylabel('Frequency')
  plt.title('Comparison of Features Before and After Treatment')
  plt.xticks(x, feature_names, rotation='vertical')
  plt.legend()
  plt.tight_layout()
  plt.show()
