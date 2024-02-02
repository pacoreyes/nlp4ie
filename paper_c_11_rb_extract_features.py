from pprint import pprint
from collections import Counter

import spacy
from tqdm import tqdm
from afinn import Afinn
from textblob import TextBlob

from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file

# Load datasets
dataset_training = load_jsonl_file("shared_data/dataset_3_1_training.jsonl")
dataset_validation = load_jsonl_file("shared_data/dataset_3_2_validation.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_3_3_test.jsonl")

output_features = "shared_data/dataset_3_10_features.jsonl"

# Empty JSONL file
empty_json_file(output_features)

# Join datasets
dataset = dataset_training + dataset_validation + dataset_test

# oppose_sentences = [datapoint for datapoint in dataset if datapoint["label"] == "oppose"]
# support_sentences = [datapoint for datapoint in dataset if datapoint["label"] == "support"]

# Initialize spacy model
nlp = spacy.load("en_core_web_trf")

# Initialize Afinn sentiment analyzer
afinn = Afinn()


def measure_sentence_length(_doc):
  return len([token for token in doc])


# Feature 2: Word length, a list of the length of all words, excluding punctuation
def measure_word_length(_doc):
  return [len(token.text) for token in _doc if not token.is_punct]


def measure_adjective_polarity(_doc, pole):
  # Get the polarity words
  polarity_words = set()
  for token in _doc:
    # Get sentiment score
    # score = afinn.score(token.text)
    blob = TextBlob(token.text)
    score = blob.sentiment.polarity
    if pole == "pos":
      if token.pos_ == "ADJ" and token.ent_type_ == "" and score > 0:
        polarity_words.add(token.text)
    elif pole == "neg":
      if token.pos_ == "ADJ" and token.ent_type_ == "" and score < 0:
        polarity_words.add(token.text)
  return len(polarity_words)


def measure_adverb_polarity(_doc, pole):
  # Get the polarity words
  polarity_words = set()
  for token in _doc:
    # Get sentiment score
    # score = afinn.score(token.text)
    blob = TextBlob(token.text)
    score = blob.sentiment.polarity
    if pole == "pos":
      if token.pos_ == "ADV" and token.ent_type_ == "" and score > 0:
        polarity_words.add(token.text)
    elif pole == "neg":
      if token.pos_ == "ADV" and token.ent_type_ == "" and score < 0:
        polarity_words.add(token.text)
  return len(polarity_words)


def measure_verb_polarity(_doc, pole):
  # Get the polarity words
  polarity_words = set()
  for token in _doc:
    # Get sentiment score
    # score = afinn.score(token.text)
    blob = TextBlob(token.text)
    score = blob.sentiment.polarity
    if pole == "pos":
      if token.pos_ == "VERB" and token.ent_type_ == "" and score > 0:
        polarity_words.add(token.text)
    elif pole == "neg":
      if token.pos_ == "VERB" and token.ent_type_ == "" and score < 0:
        polarity_words.add(token.text)
  return len(polarity_words)


def measure_personal_pronoun_use(_doc):
  personal_pronouns = ['i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it',
                       'its', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs']
  # Make a list of the number of personal pronouns in each sentence
  return len([token for token in _doc if token.text.lower() in personal_pronouns])


print("Extracting Stance features from dataset...")
for sentence in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  doc = nlp(sentence["text"])
  label = sentence["label"]

  neg_adj_polarity = measure_adjective_polarity(doc, "neg")
  pos_adj_polarity = measure_adjective_polarity(doc, "pos")

  row = {
    "text": sentence["text"],
    "label": label,
    "id": sentence["id"],
    "sentence_length": measure_sentence_length(doc),
    "word_length": measure_word_length(doc),
    "pos_adj_polarity": pos_adj_polarity,
    "neg_adj_polarity": neg_adj_polarity,
    "pos_adv_polarity": measure_adverb_polarity(doc, "pos"),
    "neg_adv_polarity": measure_adverb_polarity(doc, "neg"),
    "pos_verb_polarity": measure_verb_polarity(doc, "pos"),
    "neg_verb_polarity": measure_verb_polarity(doc, "neg"),
    "personal_pronouns": measure_personal_pronoun_use(doc)
  }
  save_row_to_jsonl_file(row, output_features)

print("Finished!")
