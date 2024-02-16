"""
This script calculates the descriptive statistics for the features extracted from the measurements of linguistic
features extracted from the texts.
"""

from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from typing import List
from pprint import pprint

from lib.utils import load_jsonl_file, write_to_google_sheet
from db import spreadsheet_5


# Load the data
dataset_train = load_jsonl_file("shared_data/dataset_1_5_1a_train_features.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_1_5_1a_test_features.jsonl")

# Load the data of features into a DataFrame
df = pd.DataFrame(dataset_train + dataset_test)

# Split the dataframe into monologic and dialogic
df_monologic = df[df['label'] == "monologic"]
df_dialogic = df[df['label'] == "dialogic"]

pprint(df_monologic.head())
pprint(df_dialogic.head())


def calculate_statistics(column: List[int]) -> dict:
  """Calculate basic statistics for a given column."""
  stats = {
    'mean': np.mean(column),
    'median': np.median(column),
    'mode': np.argmax(np.bincount(column)) if column else np.nan,
    'min': np.min(column),
    'max': np.max(column),
    'std_dev': np.std(column),
    'variance': np.var(column),
    'skewness': skew(np.array(column)),
    'kurtosis': kurtosis(column),
  }
  return stats


# List of features
ms_features = [
  'sentence_length',
  'word_length',
  'sentence_complexity',
  'personal_pronoun_d',
  'passive_voice_d',
  'nominalization_d',
  'lexical_d',
  'interjection_d',
  'modal_verb_d',
  'discourse_marker_d'
]

# Initialize dictionaries to store the results
monologic_stats = {}
dialogic_stats = {}

# Convert DataFrame to JSON serializable (list of dictionaries)
# features = df.to_dict(orient='records')

# Calculate the statistics for each feature
for feature in tqdm(ms_features, desc="Calculating statistics", total=len(ms_features)):
  monologic_stats[feature] = calculate_statistics([item for sublist in df_monologic[feature] for item in sublist])
  dialogic_stats[feature] = calculate_statistics([item for sublist in df_dialogic[feature] for item in sublist])

# Convert the dictionaries into dataframes and reset index
df_monologic_stats = pd.DataFrame(monologic_stats).transpose().reset_index()
df_dialogic_stats = pd.DataFrame(dialogic_stats).transpose().reset_index()

# Rename the new column containing the feature names to 'features'
df_monologic_stats = df_monologic_stats.rename(columns={"index": "features"})
df_dialogic_stats = df_dialogic_stats.rename(columns={"index": "features"})

# Convert the dataframes into JSON serializable (list of dictionaries)
monologic_stats = df_monologic_stats.to_dict(orient='records')
dialogic_stats = df_dialogic_stats.to_dict(orient='records')

# Assuming dialogic_stats is a list of dictionaries from df_dialogic_stats.to_dict(orient='records')
monologic_stats = [[item[key] for key in item] for item in monologic_stats]
dialogic_stats = [[item[key] for key in item] for item in dialogic_stats]

# Store the monologic dataframe on a Google Sheet
write_to_google_sheet(spreadsheet_5, "monologic_desc_stat", monologic_stats)

# Store the dialogic dataframe on a Google Sheet
write_to_google_sheet(spreadsheet_5, "dialogic_desc_stat", dialogic_stats)

print("Descriptive statistics for monologic")
pprint(df_monologic_stats)
print()
print("Descriptive statistics for dialogic")
pprint(df_dialogic_stats)
