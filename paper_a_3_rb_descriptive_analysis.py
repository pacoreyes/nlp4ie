"""
This script calculates the descriptive statistics for the features extracted from the measurements of linguistic
features extracted from the texts.
"""
from pprint import pprint

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from gspread_dataframe import set_with_dataframe

from lib.utils import load_jsonl_file
from db import spreadsheet_5


# Load the data
ms_data = load_jsonl_file("shared_data/dataset_1_2_features.jsonl")

# Load the data of features into a DataFrame
df = pd.DataFrame(ms_data)

# Split the dataframe into monologic and dialogic
df_monologic = df[df['discourse_type'] == "monologic"]
df_dialogic = df[df['discourse_type'] == "dialogic"]


def calculate_statistics(column):
  """Calculate basic statistics for a given column."""
  stats = {
    'mean': np.mean(column),
    'median': np.median(column),
    'mode': np.argmax(np.bincount(column)) if column else np.nan,
    'min': np.min(column),
    'max': np.max(column),
    'std_dev': np.std(column),
    'variance': np.var(column),
    'skewness': skew(column),
    'kurtosis': kurtosis(column),
    # 'iqr': np.subtract(*np.percentile(column, [75, 25]))
  }
  return stats


# List of features
ms_features = [
  'sentence_length',
  'word_length',
  'sentence_complexity',
  'personal_pronoun_use',
  'passive_voice_use',
  'nominalization_use',
  'lexical_density',
  'interjection_use',
  'modal_verb_use',
  'discourse_markers_use'
]

# Initialize dictionaries to store the results
monologic_stats = {}
dialogic_stats = {}

# Store the dataframe on a Google Sheet
sheet = spreadsheet_5.worksheet("feature_extraction")
# Append DataFrame to worksheet
set_with_dataframe(sheet, df)

# Calculate the statistics for each feature
for feature in ms_features:
  monologic_stats[feature] = calculate_statistics([item for sublist in df_monologic[feature] for item in sublist])
  dialogic_stats[feature] = calculate_statistics([item for sublist in df_dialogic[feature] for item in sublist])

# Convert the dictionaries into dataframes and reset index
df_monologic_stats = pd.DataFrame(monologic_stats).transpose().reset_index()
df_dialogic_stats = pd.DataFrame(dialogic_stats).transpose().reset_index()

# Rename the new column containing the feature names to 'features'
df_monologic_stats = df_monologic_stats.rename(columns={"index": "features"})
df_dialogic_stats = df_dialogic_stats.rename(columns={"index": "features"})

print("Descriptive statistics for monologic")
pprint(df_monologic_stats)
print()
print("Descriptive statistics for dialogic")
pprint(df_dialogic_stats)

# Store the monologic dataframe on a Google Sheet
sheet = spreadsheet_5.worksheet("monologic_stats")
# Append DataFrame to worksheet
set_with_dataframe(sheet, df_monologic_stats)

# Store the dialogic dataframe on a Google Sheet
sheet = spreadsheet_5.worksheet("dialogic_stats")
# Append DataFrame to worksheet
set_with_dataframe(sheet, df_dialogic_stats)
