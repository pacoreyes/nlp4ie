import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

from lib.utils import load_jsonl_file, save_jsonl_file, empty_json_file


# Initialize label map and class names
LABEL_MAP = {"monologic": 0, "dialogic": 1}

dataset_train = load_jsonl_file("shared_data/dataset_1_5_1a_train_features.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_1_5_1a_test_features.jsonl")

output_dataset_train = "shared_data/dataset_1_5_1a_train_features_transformed.jsonl"
output_dataset_test = "shared_data/dataset_1_5_1a_test_features_transformed.jsonl"

output_paths = [output_dataset_train, output_dataset_test]

for path in output_paths:
  empty_json_file(path)

features = {
  "sentence_length": "Sentence Length",
  "word_length": "Word Length",
  "sentence_complexity": "Sentence Complexity",
  "passive_voice_d": "Passive Voice Frequency",
  "lexical_d": "Lexical Word Frequency",
  "nominalization_d": "Nominalization Frequency",
  "personal_pronoun_d": "Personal Pronoun Frequency",
  "interjection_d": "Interjection Frequency",
  "modal_verb_d": "Modal Verb Frequency",
  "discourse_marker_d": "Discourse Marker Frequency"
}

# Load the training dataset into a DataFrame
df_train = pd.json_normalize(dataset_train, max_level=0)


def calculate_statistics(column):
  """Calculate basic statistics for a given column."""
  stats = {
    'mean': np.mean(column),
    'median': np.median(column),
    'mode': np.argmax(np.bincount(column)) if column else np.nan,
    'min': np.min(column),
    'max': np.max(column),
    'range': np.max(column) - np.min(column),
    'std_dev': np.std(column),
    'variance': np.var(column),
    'skewness': skew(np.array(column)),
    'kurtosis': kurtosis(column),
  }
  return stats


