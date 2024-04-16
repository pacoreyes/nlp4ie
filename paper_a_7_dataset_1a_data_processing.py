"""
This script calculates the descriptive statistics for the features extracted from the measurements of linguistic
features extracted from the texts.
"""
import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, shapiro
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from lib.utils import load_jsonl_file, save_jsonl_file
from lib.visualizations import plot_correlation_heatmap_double

# Load the data
dataset_train = load_jsonl_file("shared_data/dataset_1_5_1a_train_features.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_1_5_1a_test_features.jsonl")

# Load the data of features into a DataFrame
df = pd.DataFrame(dataset_train + dataset_test)

# Split the dataframe into speech and interview
df_speech = df[df['label'] == "monologic"]
df_interview = df[df['label'] == "dialogic"]


def calculate_statistics(column):
  """Calculate basic statistics for a given column."""
  return {
    'mean': f"{np.mean(column):.2f}",
    'median': np.median(column),
    'mode': np.argmax(np.bincount(column)) if column else np.nan,
    'min': np.min(column),
    'max': np.max(column),
    'range': np.max(column) - np.min(column),
    'std_dev': f"{np.std(column):.2f}",
    'variance': f"{np.var(column):.2f}",
    'skewness': f"{skew(np.array(column)):.2f}",
    'kurtosis': f"{kurtosis(column):.2f}",
  }


features = {
  "sentence_length": "Sentence Length",
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

# Initialize dictionaries to store the results
speech_stats = {}
interview_stats = {}

# Calculate the statistics for each feature
for feature in tqdm(ms_features, desc="Calculating statistics", total=len(ms_features)):
  speech_stats[feature] = calculate_statistics([item for sublist in df_speech[feature] for item in sublist])
  interview_stats[feature] = calculate_statistics([item for sublist in df_interview[feature] for item in sublist])

# Convert the dictionaries into dataframes and reset index
df_speech_stats = pd.DataFrame(speech_stats).transpose().reset_index()
df_interview_stats = pd.DataFrame(interview_stats).transpose().reset_index()

# Rename the new column containing the feature names to 'features'
df_speech_stats = df_speech_stats.rename(columns={"index": "features"})
df_interview_stats = df_interview_stats.rename(columns={"index": "features"})

# Convert the dataframes into JSON serializable (list of dictionaries)
speech_stats = df_speech_stats.to_dict(orient='records')
interview_stats = df_interview_stats.to_dict(orient='records')

# Assuming dialogic_stats is a list of dictionaries from df_dialogic_stats.to_dict(orient='records')
speech_stats = [[item[key] for key in item] for item in speech_stats]
interview_stats = [[item[key] for key in item] for item in interview_stats]

# Save starts to Excel in two sheets
with pd.ExcelWriter("shared_data/paper_a_descriptive_statistics.xlsx") as writer:
  df_speech_stats.to_excel(writer, sheet_name="speech")
  df_interview_stats.to_excel(writer, sheet_name="interview")

print("Descriptive statistics for speech")
pprint(df_speech_stats)
print()
print("Descriptive statistics for interview")
pprint(df_interview_stats)

# Convert the data to HTML tables

html_table = df_speech_stats.to_html()

# Save the HTML table to a file
with open("shared_data/paper_a_1_speech_stats_table.html", "w") as file:
  file.write(html_table)

html_table = df_interview_stats.to_html()

# Save the HTML table to a file
with open("shared_data/paper_a_2_interview_stats_table.html", "w") as file:
  file.write(html_table)

""" ############################## """


def aggregate_features(_df):
  aggregated_data = []
  for _, row in _df.iterrows():
    aggregated_row = {}
    for _feature in ms_features:
      aggregated_row[_feature] = np.mean(row[_feature]) if row[_feature] else np.nan
    aggregated_data.append(aggregated_row)
  return pd.DataFrame(aggregated_data)


df_speech_aggregated = aggregate_features(df_speech)
df_interview_aggregated = aggregate_features(df_interview)

# Now, calculate the correlation matrix for each

correlation_matrix_speech = df_speech_aggregated.corr(method='pearson')
correlation_matrix_interview = df_interview_aggregated.corr(method='pearson')

# Round the correlation matrices to three decimal places
correlation_matrix_speech = correlation_matrix_speech.round(3)
correlation_matrix_interview = correlation_matrix_interview.round(3)

# Convert correlation matrices to HTML tables
html_table = correlation_matrix_speech.to_html()

# Save the HTML table to a file
with open("shared_data/paper_a_3_speech_correlation_matrix_table.html", "w") as file:
  file.write(html_table)

html_table = correlation_matrix_interview.to_html()

# Save the HTML table to a file
with open("shared_data/paper_a_4_interview_correlation_matrix_table.html", "w") as file:
  file.write(html_table)

# Output the correlation matrices
print("Correlation matrix for speech data:")
print(correlation_matrix_speech)
print("\nCorrelation matrix for interview data:")
print(correlation_matrix_interview)

correlation_matrix_speech.to_json("shared_data/paper_a_5_correlation_matrix_speech.json")
correlation_matrix_interview.to_json("shared_data/paper_a_6_correlation_matrix_interview.json")

# Plot the correlation matrices
plot_correlation_heatmap_double(
  correlation_matrix_speech,
  correlation_matrix_interview,
  'Correlation Matrix of Linguistic features for the Speech Class',
  'Correlation Matrix of Linguistic features for the Interview Class',
  feature_names,
  "images/paper_a_7_correlation_matrix_heatmap.png")

""" ############################## """


def replace_with_mean(_df, _ms_features):
  # copy the dataframe
  df_flatten = _df.copy()
  # Flatten the list of values in each cell to a single mean value
  for column in _ms_features:
    df_flatten[column] = df_flatten[column].apply(lambda x: np.mean([float(item) for item in x]))
  return df_flatten


# Apply the function to both DataFrames
df_speech_mean = replace_with_mean(df_speech.copy(), ms_features)
df_interview_mean = replace_with_mean(df_interview.copy(), ms_features)

print("Generating plots for all features in speech class...")
f, axes = plt.subplots(len(features), 3, figsize=(20, 4 * len(ms_features)), sharex=False)
for i, feature in enumerate(ms_features):
  skewness_speech = df_speech_mean[feature].skew()
  # print(f"- Skewness of {feature_names[i]}: {skewness_speech}")

  # Histogram and Density Plot
  sns.histplot(df_speech_mean[feature], color="b", ax=axes[i, 0], kde=True)
  axes[i, 0].set_title('Histogram - ' + feature_names[i])
  axes[i, 0].set_xlabel(f"Skewness: {skewness_speech:.2f}")

  # Q-Q Plot
  stats.probplot(df_speech_mean[feature], dist="norm", plot=axes[i, 1])
  axes[i, 1].set_title('QQ Plot - ' + feature_names[i])

  # Boxplot
  sns.boxplot(x=df_speech_mean[feature], ax=axes[i, 2])
  axes[i, 2].set_title('Boxplot - ' + feature_names[i])

plt.tight_layout()
plt.savefig(f"images/paper_a_8_speech_all_features_before_treatment.png")
plt.close()
print("Generated plots for all features in speech class")
print("--------------------------------------------------------------")


print("Generating plots for all features in interview class...")
f, axes = plt.subplots(len(features), 3, figsize=(20, 4 * len(ms_features)), sharex=False)
for i, feature in enumerate(ms_features):
  skewness_speech = df_interview_mean[feature].skew()
  # print(f"- Skewness of {feature_names[i]}: {skewness_speech}")

  # Histogram and Density Plot
  sns.histplot(df_interview_mean[feature], color="r", ax=axes[i, 0], kde=True)
  axes[i, 0].set_title('Histogram - ' + feature_names[i])
  axes[i, 0].set_xlabel(f"Skewness: {skewness_speech:.2f}")

  # Q-Q Plot
  stats.probplot(df_interview_mean[feature], dist="norm", plot=axes[i, 1])
  axes[i, 1].set_title('QQ Plot - ' + feature_names[i])

  # Boxplot
  sns.boxplot(x=df_interview_mean[feature], ax=axes[i, 2])
  axes[i, 2].set_title('Boxplot - ' + feature_names[i])

plt.tight_layout()
plt.savefig(f"images/paper_a_9_interview_all_features_before_treatment.png")
plt.close()
print("Generated plots for all features in interview class")
print("--------------------------------------------------------------")


print("Generating comparative plots for all features from both classes...")
num_features = len(ms_features)
num_rows = num_features  # Each feature gets a row with 2 plots

# Create a large figure to contain all the plots
fig, axs = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows))

for i, feature in enumerate(ms_features):
  # Adjust the index if num_rows > 1, otherwise keep it simple for a single feature
  if num_features > 1:
      ax_hist, ax_box = axs[i, 0], axs[i, 1]
  else:
      ax_hist, ax_box = axs[0], axs[1]

  fig.suptitle(f'Comparative Distribution by Class', fontsize=16)

  # Histograms
  ax_hist.hist(df_speech_mean[feature], bins=20, alpha=0.5, label='Speech')
  ax_hist.hist(df_interview_mean[feature], bins=20, alpha=0.5, label='Interview')
  ax_hist.set_title(f'Histogram - {feature_names[i]}')
  ax_hist.set_xlabel(feature_names[i])
  ax_hist.set_ylabel('Frequency')
  ax_hist.legend()

  # Box plots
  data_to_plot = [df_speech_mean[feature], df_interview_mean[feature]]
  ax_box.boxplot(data_to_plot, patch_artist=True, labels=['Speech', 'Interview'])
  ax_box.set_title(f'Box Plot - {feature_names[i]}')
  ax_box.set_ylabel(feature_names[i])

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Save the entire figure as a PNG file
plt.savefig('images/paper_a_10_comparative_features.png')
# Optionally, you can display the figure in the notebook as well
# plt.show()
plt.close()

print("Generated plots for all features from both classes")
print("--------------------------------------------------------------")

save_jsonl_file(
  df_speech_mean.to_dict(orient='records'),
  "shared_data/dataset_1_5_1a_train_speech_features_transformed.jsonl")

save_jsonl_file(
  df_interview_mean.to_dict(orient='records'),
  "shared_data/dataset_1_5_1a_train_interview_features_transformed.jsonl")

""" ############################## """

# Set the aesthetics for the plots
sns.set(style="whitegrid")


# Define a function to create box plots for each feature
def plot_feature_boxplots(_df, _title, _filename, _feature_names):
  plt.figure(figsize=(20, 10))
  _df.drop(['id', 'label'], axis=1).plot(kind='box', vert=False)
  # rename column names to feature names using the mapping "features"
  plt.yticks(ticks=range(1, len(_feature_names) + 1), labels=_feature_names)

  plt.title(_title)
  plt.savefig(_filename)
  plt.close()


# Plot box plots for speeches
plot_feature_boxplots(
  df_speech_mean,
  'Box Plot of Features for Speech',
  'images/paper_a/paper_a_12_speech_feature_boxplots.png',
  feature_names)

# Plot box plots for interviews
plot_feature_boxplots(
  df_interview_mean,
  'Box Plot of Features for Interview',
  'images/paper_a/paper_a_13_interview_feature_boxplots.png',
  feature_names)

""" ############################## """

# Calculate the variance for each feature for interviews and speeches separately
var_interviews = df_speech_mean.drop(columns=['id', 'label']).var()
var_speeches = df_interview_mean.drop(columns=['id', 'label']).var()

# Prepare data for plotting
var_df = pd.DataFrame({'Speech': var_speeches, 'Interview': var_interviews})

# Plotting the variance for each class separately
plt.figure(figsize=(12, 7))
var_df.plot(kind='bar', color={'Speech': '#6EC173', 'Interview': '#147E3A'})
plt.title('Variance of Linguistic Features by Class')
plt.ylabel('Variance')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right', labels=feature_names, ticks=range(0, len(feature_names)))
plt.legend(title='Class')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('images/paper_a_14_variance_by_class.png')

""" ############################## """

# Initialize a dictionary to hold features that may require normalization
features_require_normalization = {'speech': [], 'interview': []}


# Function to test a single dataframe and class name
def test_normality(df, class_name):
    for feature in ms_features:
        stat, p = shapiro(df[feature])
        if p <= 0.05:  # Using alpha = 0.05
            print(f"{class_name} - {feature}: p-value = {p}. May require normalization.")
            features_require_normalization[class_name].append(feature)
        else:
            print(f"{class_name} - {feature}: p-value = {p}. Normal distribution **.")


# Run the Shapiro-Wilk test for each class
test_normality(df_speech_mean, 'speech')
test_normality(df_interview_mean, 'interview')

# Print the summary of features that may require normalization
print("\nFeatures that may require normalization:")
pprint(features_require_normalization)

# Make a list of all the ids in the dataset
train_ids = [item['id'] for item in dataset_train]
test_ids = [item['id'] for item in dataset_test]

all_datapoints = [df_speech_mean, df_interview_mean]
all_datapoints = pd.concat(all_datapoints, ignore_index=True)
print(all_datapoints)

# Filter the dataset_train and dataset_test based on the ids
dataset_train = []
for idx, item in all_datapoints.iterrows():
  if item['id'] in train_ids:
    dataset_train.append(item)

# Convert to dataframe
dataset_train_df = pd.DataFrame(dataset_train)

pprint(dataset_train_df)

dataset_test = []
for idx, item in all_datapoints.iterrows():
  if item['id'] in test_ids:
    dataset_test.append(item)

# Convert to dataframe
dataset_test_df = pd.DataFrame(dataset_test)

# Applying Min-Max scaling
# scaler = MinMaxScaler()
min_max_scaler = preprocessing.MinMaxScaler()

# Initializing the StandardScaler
# scaler = StandardScaler()

dataset_train_df_scaled = dataset_train_df.copy()
dataset_train_df_scaled[ms_features] = min_max_scaler.fit_transform(dataset_train_df[ms_features])
# dataset_train_df_scaled[ms_features] = scaler.fit_transform(dataset_train_df[ms_features])
dataset_train_scaled = dataset_train_df_scaled.to_dict(orient='records')

dataset_test_df_scaled = dataset_test_df.copy()
dataset_test_df_scaled[ms_features] = min_max_scaler.transform(dataset_test_df[ms_features])
# dataset_test_df_scaled[ms_features] = scaler.transform(dataset_test_df[ms_features])
dataset_test_scaled = dataset_test_df_scaled.to_dict(orient='records')

# Save the train data to a JSONL file
save_jsonl_file(
  dataset_train_scaled,
  "shared_data/dataset_1_5_1a_train_features_transformed.jsonl")

# Save the test data to a JSONL file
save_jsonl_file(
  dataset_test_scaled,
  "shared_data/dataset_1_5_1a_test_features_transformed.jsonl")

# Remove temp files
os.remove("shared_data/dataset_1_5_1a_train_speech_features_transformed.jsonl")
os.remove("shared_data/dataset_1_5_1a_train_interview_features_transformed.jsonl")
