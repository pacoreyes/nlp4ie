from pprint import pprint

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from lib.utils import load_jsonl_file

features = {
  # "sentence_length": "Sentence Length",
  "word_length": "Word Length",
  "sentence_complexity": "Sentence Complexity",
  "passive_voice_d": "Passive Voice Freq",
  "lexical_d": "Lexical Word Freq",
  "nominalization_d": "Nominalization Freq",
  "personal_pronoun_d": "Personal Pronoun Freq",
  "interjection_d": "Interjection Freq",
  "modal_verb_d": "Modal Verb Freq",
  "discourse_marker_d": "Discourse Marker Freq"
}

# Make list of keys and list of values
ms_features = list(features.keys())
feature_names = list(features.values())

# Load the data
dataset_train = load_jsonl_file("shared_data/dataset_1_5_1a_train_features_transformed.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_1_5_1a_test_features_transformed.jsonl")

dataset = dataset_train

# Split the dataset into 2 classes
class_speech = [datapoint for datapoint in dataset if datapoint["label"] == "monologic"]
class_interview = [datapoint for datapoint in dataset if datapoint["label"] == "dialogic"]

# Load each class into a DataFrame
df_speech = pd.DataFrame(class_speech)
df_interview = pd.DataFrame(class_interview)

# Separate the dataset into two groups based on the labels
df_speech = df_speech.drop(columns=['id', 'label', 'sentence_length'])
df_interview = df_interview.drop(columns=['id', 'label', 'sentence_length'])

# Initialize a dictionary to store t-test results
t_test_results = {}

# Conduct t-tests for each feature
for feature in ms_features:
  speech_mean = df_speech[feature].mean()
  interview_mean = df_interview[feature].mean()
  speech_sd = df_speech[feature].std()
  interview_sd = df_interview[feature].std()
  stat, p_value = stats.ttest_ind(df_speech[feature], df_interview[feature], equal_var=False)  # Welch's t-test
  t_test_results[features[feature]] = {
    'statistic': stat,
    'p_value': p_value,
    'speech_mean': speech_mean.round(2),
    'interview_mean': interview_mean.round(2),
    'speech_sd': speech_sd.round(2),
    'interview_sd': interview_sd.round(2)
  }

# Convert the results dictionary to a DataFrame for easier analysis
t_test_results_df = pd.DataFrame.from_dict(t_test_results, orient='index').sort_values(by='p_value')

# Adding a "reject t-test" column based on the original p-values and a typical alpha level of 0.05
t_test_results_df['reject_t_test'] = t_test_results_df['p_value'] < 0.05

# pprint(t_test_results_df)

# Extracting p-values from the results
p_values = t_test_results_df['p_value'].values

# Apply the Bonferroni correction to the p-values
bonferroni_corrected = multipletests(p_values, method='bonferroni')

# Applying Benjamini-Hochberg FDR correction
fdr_bh_corrected = multipletests(p_values, method='fdr_bh')

# Adding the corrected p-values and rejection decisions to the DataFrame
t_test_results_df['p_value_bonferroni'] = bonferroni_corrected[1]
t_test_results_df['reject_bonferroni'] = bonferroni_corrected[0]
t_test_results_df['p_value_fdr_bh'] = fdr_bh_corrected[1]
t_test_results_df['reject_fdr_bh'] = fdr_bh_corrected[0]

# Calculate the total number of columns
total_columns = len(t_test_results_df.columns)

# Iterate over the DataFrame in steps of 4 columns
for start in range(0, total_columns, 4):
  end = start + 4
  # Print the subset of the DataFrame with the current 4 columns
  print(t_test_results_df.iloc[:, start:end])

html_data = t_test_results_df.to_html()

# Save the HTML data to a file
html_file_path = "shared_data/paper_a_7_t_test_results.html"
with open(html_file_path, "w") as file:
    file.write(html_data)

print(f"T-test scores saved in HTML format to '{html_file_path}'")
