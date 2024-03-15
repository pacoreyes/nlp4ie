from collections import Counter
from pprint import pprint

import spacy
from tqdm import tqdm

from lib.utils import load_jsonl_file

# Load spaCy's language model
nlp = spacy.load("en_core_web_trf")

# Load the datasets
dataset_train = load_jsonl_file("shared_data/dataset_1_6_1b_train.jsonl")
dataset_validation = load_jsonl_file("shared_data/dataset_1_6_1b_validation.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_1_6_1b_test.jsonl")

dataset = dataset_train + dataset_validation + dataset_test

# Split the datasets into two classes
speech_class = [datapoint for datapoint in dataset if datapoint["label"] == "monologic"]
interview_class = [datapoint for datapoint in dataset if datapoint["label"] == "dialogic"]

texts_by_class = {"speech": speech_class, "interview": interview_class}

word_freq_by_class = {}
# for class_label, datapoints in tqdm(texts_by_class.items(), desc=f"Processing {len(dataset)} datapoints"):
for class_label, datapoints in texts_by_class.items():
  print("---")
  tokens = []
  for datapoint in tqdm(datapoints, desc=f"Processing {len(datapoints)} datapoints"):
    # Process each text with spaCy
    doc = nlp(datapoint["text"].lower())
    # tokens.extend([token.text for token in doc if not token.is_stop and not token.is_punct])
    tokens.extend([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
  word_freq_by_class[class_label] = Counter(tokens)

# Output the most common words for each class
for class_label, word_freq in word_freq_by_class.items():
  print(f"Most common words in class {class_label}:")
  pprint(word_freq.most_common(50))
