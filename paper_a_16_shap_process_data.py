# from pprint import pprint

import spacy
import pandas as pd
from tqdm import tqdm

from lib.lexicons import nominalization_suffixes, discourse_markers, personal_pronouns, bias_words, interjections
from lib.utils import save_row_to_jsonl_file, empty_json_file

# Check if spaCy is using GPU
if spacy.prefer_gpu():
  print("spaCy is using GPU!")
else:
  print("GPU not available, spaCy is using CPU instead.")


# Load a spaCy model
nlp = spacy.load("en_core_web_lg")


""" ##########################
Step 1: load datasets
########################## """

speech_data_path = "shared_data/dataset_1_9_speech_anonym_shap_features_plot_bar.csv"
interview_data_path = "shared_data/dataset_1_9_interview_anonym_shap_features_plot_bar.csv"

speech_data = pd.read_csv(speech_data_path, index_col=None)
interview_data = pd.read_csv(interview_data_path, index_col=None)

output_file = "shared_data/dataset_1_10_shap_features_anonym.jsonl"
empty_json_file(output_file)

# speech_data = speech_data[:500]

""" ##########################
Step 2: process Speech data
########################## """

all_data = [
  {"name": "speech", "data": speech_data},
  {"name": "interview", "data": interview_data}
]

extracted_data = []

for _class in all_data:

  # initialize clusters

  interjection_freq = []
  nominalization_freq = []
  discourse_marker_freq = []
  modal_verb_freq = []
  # passive_voice_freq = []
  personal_pronoun_freq = []
  lexical_word_freq = []
  interrogations = []

  for index, feature in tqdm(_class["data"].iterrows(),
                             desc=f"Processing {_class['name']} data", total=len(_class["data"])):

    doc = nlp(str(feature["Feature"]))
    token = doc[0]
    value = float(feature["Value"])

    # print(token)
    # for token in doc:
    """if token.text == "ive":
      print(token.text.lower())"""

    if token.text.lower() == "entity" or token.lemma_ in bias_words:
      continue

    # Check if token is a word by checking if it has a vector representation
    if token.has_vector and not token.is_punct:
      # words.append(token.text)

      # 2.1 Extract interjections
      if token.pos_ == 'INTJ' and token.text.lower() in interjections:
        # print(token.text)
        # interjection_freq.append(feature.tolist())
         interjection_freq.append(value)

      # 2.2 Extract nominalizations
      if any(token.text.lower().endswith(suffix) for suffix in nominalization_suffixes):
        # nominalization_freq.append(feature.tolist())
        nominalization_freq.append(value)

      # 2.3 Extract discourse markers
      if token.text.lower() in discourse_markers:
        # print(token.text)
        # discourse_marker_freq.append(feature.tolist())
         discourse_marker_freq.append(value)

      # 2.4 Extract modal verbs
      if token.tag_ == 'MD':
        # modal_verb_freq.append(feature.tolist())
        modal_verb_freq.append(value)

      # 2.6 Extract personal pronouns
      if token.text.lower() in personal_pronouns:
        # pprint(feature.tolist())
        # personal_pronoun_freq.append(feature.tolist())
        personal_pronoun_freq.append(value)

      # 2.7 Extract lexical words
      if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
        # lexical_word_freq.append(feature.tolist())
        lexical_word_freq.append(value)

    # Then it's a subword
    elif not token.has_vector and not token.is_punct:
      # subwords.append(token.text)
      if any(token.text.lower().endswith(suffix) for suffix in nominalization_suffixes):
        # nominalization_freq.append(feature.tolist())
         nominalization_freq.append(value)

    # Collect question marks
    elif token.is_punct:
      if token.text == "?":
        interrogations.append(feature.tolist())

  row = {
    "label": _class["name"],
    "interjection_freq": interjection_freq,
    "nominalization_freq": nominalization_freq,
    "discourse_marker_freq": discourse_marker_freq,
    "modal_verb_freq": modal_verb_freq,
    "personal_pronoun_freq": personal_pronoun_freq,
    "lexical_word_freq": lexical_word_freq,
  }
  save_row_to_jsonl_file(row, output_file)
  extracted_data.append(row)
