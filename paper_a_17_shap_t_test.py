from pprint import pprint

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import shapiro
from scipy.stats import skew
import numpy as np
import matplotlib.pyplot as plt

from lib.utils import load_jsonl_file

features = {
  "interjection_freq": "Interjection Freq",
  "nominalization_freq": "Nominalization Freq",
  "discourse_marker_freq": "Discourse Marker Freq",
  # "modal_verb_freq": "Modal Verb Freq",
  "personal_pronoun_freq": "Personal Pronoun Freq",
  # "lexical_word_freq": "Lexical Word Freq",
}

# Significance level
ALPHA = 0.05

# Make list of keys and list of values
ms_features = list(features.keys())
feature_names = list(features.values())

# Load the data
dataset = load_jsonl_file("shared_data/dataset_1_10_shap_features.jsonl")

# Split the dataset into 2 classes
class_speech = dataset[0]
class_interview = dataset[1]

# Drop the 'label' attribute
class_speech.pop("label")
class_interview.pop("label")

df_speech = pd.DataFrame()
df_speech['interjection_freq'] = pd.Series(class_speech['interjection_freq'])
df_speech['nominalization_freq'] = pd.Series(class_speech['nominalization_freq'])
df_speech['discourse_marker_freq'] = pd.Series(class_speech['discourse_marker_freq'])
# df_speech['modal_verb_freq'] = pd.Series(class_speech['modal_verb_freq'])
df_speech['personal_pronoun_freq'] = pd.Series(class_speech['personal_pronoun_freq'])
# df_speech['lexical_word_freq'] = pd.Series(class_speech['lexical_word_freq'])

df_interview = pd.DataFrame()
df_interview['interjection_freq'] = pd.Series(class_interview['interjection_freq'])
df_interview['nominalization_freq'] = pd.Series(class_interview['nominalization_freq'])
df_interview['discourse_marker_freq'] = pd.Series(class_interview['discourse_marker_freq'])
# df_interview['modal_verb_freq'] = pd.Series(class_interview['modal_verb_freq'])
df_interview['personal_pronoun_freq'] = pd.Series(class_interview['personal_pronoun_freq'])
# df_interview['lexical_word_freq'] = pd.Series(class_interview['lexical_word_freq'])

all_data = [
  {"name": "speech", "data": df_speech},
  {"name": "interview", "data": df_interview}
]

for class_data in all_data:
  print(f"\nClass: {class_data['name'].capitalize()}\n")

  for key, data in class_data["data"].items():
    # Run Shapiro-Wilk test for normality
    shapiro_test_result = shapiro(data)
    if shapiro_test_result.pvalue > ALPHA:
      print(f"• {key}: log-transformed data is normally distributed (or not significantly different from normal).")
    else:
      print(f"• {key}: reject the null hypothesis - the log-transformed data is not normally distributed.")


# Initialize a dictionary to store t-test results
t_test_results = {}

# Conduct t-tests for each feature
for feature in ms_features:
  # Apply log transformation to the data
  df_speech[feature] = np.log(df_speech[feature])
  df_interview[feature] = np.log(df_interview[feature])

  # Plot Box plot for each feature including both classes
  plt.figure(figsize=(6, 5))
  bplot = plt.boxplot(
    [df_speech[feature], df_interview[feature]],
    labels=['Speech', 'Interview'],
    # notch=True,
    widths=0.3,
    patch_artist=True
  )
  for element in ['boxes']:  # , 'whiskers', 'fliers', 'means', 'caps'
    plt.setp(bplot[element], color='#0CA37F')

  for box in bplot['boxes']:
    box.set_edgecolor('black')  # Set edge color to blue
    box.set_linewidth(1)  # Set line width to 2

  # Set fill color for boxes
  for patch in bplot['boxes']:
    patch.set_facecolor('#0CA37F')

  plt.setp(bplot['medians'], color='red', linewidth=2)
  plt.tick_params(axis='y', which='major', labelsize=17)
  plt.tick_params(axis='x', which='major', labelsize=15)

  # Set the background grid
  plt.grid(True, linestyle='--', linewidth=0.5, color='grey')

  # plt.title(f'Box plot of {features[feature]}')
  plt.ylabel('Log-transformed frequency')
  plt.savefig(f"images/paper_a_21_{feature}_boxplots_anonym.png")
  plt.close()

  # Calculate mean and standard deviation of the log-transformed data
  speech_mean = df_speech[feature].mean()
  interview_mean = df_interview[feature].mean()
  speech_sd = df_speech[feature].std()
  interview_sd = df_interview[feature].std()
  # Perform Welch's t-test
  stat, p_value = stats.ttest_ind(df_speech[feature], df_interview[feature], equal_var=False)
  print(stat, p_value)
  t_test_results[features[feature]] = {
    'statistic': stat.round(1),
    'p_value': p_value.round(3),
    'speech_mean': speech_mean.round(2),
    'interview_mean': interview_mean.round(2),
    'speech_sd': speech_sd.round(2),
    'interview_sd': interview_sd.round(2)
  }

# Convert the results dictionary to a DataFrame for easier analysis
t_test_results_df = pd.DataFrame.from_dict(t_test_results, orient='index').sort_values(by='p_value')

# Adding a "reject t-test" column based on the original p-values and a typical alpha level of 0.05
t_test_results_df['reject_t_test'] = t_test_results_df['p_value'] < 0.05

pprint(t_test_results_df)

p_values = [result['p_value'] for result in t_test_results.values()]

# Apply the Bonferroni correction
_, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

# Update the results dictionary with corrected p-values
for (feature, result), pval_corrected in zip(t_test_results.items(), pvals_corrected):
    t_test_results[feature]['p_value_corrected'] = pval_corrected

# Convert the results dictionary to a DataFrame for easier analysis
t_test_results_df = pd.DataFrame.from_dict(t_test_results, orient='index').sort_values(by='p_value_corrected')

# Adding a "reject t-test" column based on the corrected p-values
t_test_results_df['reject_t_test'] = t_test_results_df['p_value_corrected'] < 0.05

pprint(t_test_results_df)

"""t_test_data = []

for class_data in all_data:
  print(f"\nClass: {class_data['name'].capitalize()}\n")

  mean_values = []

  for key, data in class_data["data"].items():
    data_values = np.array(data)
    # Calculate skewness of original data
    original_skewness = skew(data_values)
    log_transformed_values = np.log(data)"""
    

"""
    # Calculate mean of log-transformed values
    mean_values.append({
      key: np.mean(log_transformed_values),
    })
    # Calculate skewness of log-transformed data
    log_skewness = skew(log_transformed_values)
    # Run Shapiro-Wilk test for normality
    shapiro_test_result = shapiro(log_transformed_values)
    if shapiro_test_result.pvalue > ALPHA:
      print(f"{key}: log-transformed data is normally distributed (or not significantly different from normal).")
    else:
      print(f"{key}: reject the null hypothesis - the log-transformed data is not normally distributed.")
    print(original_skewness, log_skewness)

  t_test_data.append({
    "class": _class["name"],
    "mean_values": mean_values
  })

speech_data = t_test_data[0]["mean_values"]
interview_data = t_test_data[1]["mean_values"]

speach_means = {
  "interjection_freq": speech_data[0]["interjection_freq"],
  "nominalization_freq": speech_data[1]["nominalization_freq"],
  "discourse_marker_freq": speech_data[2]["discourse_marker_freq"],
  "modal_verb_freq": speech_data[3]["modal_verb_freq"],
  "personal_pronoun_freq": speech_data[4]["personal_pronoun_freq"],
  "lexical_word_freq": speech_data[5]["lexical_word_freq"],
}

interview_means = {
  "interjection_freq": interview_data[0]["interjection_freq"],
  "nominalization_freq": interview_data[1]["nominalization_freq"],
  "discourse_marker_freq": interview_data[2]["discourse_marker_freq"],
  "modal_verb_freq": interview_data[3]["modal_verb_freq"],
  "personal_pronoun_freq": interview_data[4]["personal_pronoun_freq"],
  "lexical_word_freq": interview_data[5]["lexical_word_freq"],
}

pprint(speach_means)
pprint(interview_means)

for feature in ms_features:
  # print(speach_means[feature], interview_means[feature])
  stat, p_value = stats.ttest_ind(speach_means[feature], interview_means[feature], equal_var=False)  # Welch's t-test
"""