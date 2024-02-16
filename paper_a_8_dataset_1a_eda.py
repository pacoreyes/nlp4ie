"""
This script performs exploratory data analysis (EDA) on the dataset.
It performs the following steps:
  1. Observe the distribution of each feature in each class
  2. Test the normality of each feature in each class
  3. Apply log transformation and IQR outlier removal to each feature in each class
  4. Tests the normality of each feature in each class AFTER transformation
  5. Observes the distribution of each feature in each class AFTER transformation
  6. Saves the transformed dataset to a JSONL file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, boxcox
from collections import defaultdict

from lib.utils import load_jsonl_file, save_jsonl_file, empty_json_file

dataset_train = load_jsonl_file("shared_data/dataset_1_5_1a_train_features.jsonl")
length_train_dataset = len(dataset_train)
dataset_test = load_jsonl_file("shared_data/dataset_1_5_1a_test_features.jsonl")

output_dataset_train = "shared_data/dataset_1_5_1a_train_features_transformed.jsonl"
output_dataset_test = "shared_data/dataset_1_5_1a_test_features_transformed.jsonl"

dataset = dataset_train + dataset_test

features = ["sentence_length", "word_length", "sentence_complexity", "personal_pronoun_d",
            "passive_voice_d", "nominalization_d", "lexical_d", "interjection_d", "modal_verb_d",
            "discourse_marker_d"]

# Convert the data into a DataFrame
df = pd.json_normalize(dataset, max_level=0)

# copy the dataframe
df_flatten = df.copy()

# Flatten the list of values in each cell to a single mean value
for feature in features:
  df_flatten[feature] = df_flatten[feature].apply(lambda x: np.mean([float(item) for item in x]))

# Divide dataset into two classes: monologic and dialogic
df_monologic = df_flatten[df_flatten["label"] == "monologic"].copy()
df_dialogic = df_flatten[df_flatten["label"] == "dialogic"].copy()

""" ##########################################################
1. Observe the distribution of each feature in each class
###########################################################"""

# 1. Monologic Texts, make a visualization for each feature
print("Generating plots for all features in monologic texts...")
f, axes = plt.subplots(len(features), 3, figsize=(20, 4 * len(features)), sharex=False)
for i, feature in enumerate(features):
  skewness_monologic = df_monologic[feature].skew()
  print(f"- Skewness of {feature}: {skewness_monologic}")

  # Histogram and Density Plot
  sns.histplot(df_monologic[feature], color="b", ax=axes[i, 0], kde=True)
  axes[i, 0].set_title('Histogram - ' + feature)
  axes[i, 0].set_xlabel(f"Skewness: {skewness_monologic:.2f}")

  # Q-Q Plot
  stats.probplot(df_monologic[feature], dist="norm", plot=axes[i, 1])
  axes[i, 1].set_title('QQ Plot - ' + feature)

  # Boxplot
  sns.boxplot(x=df_monologic[feature], ax=axes[i, 2])
  axes[i, 2].set_title('Boxplot - ' + feature)

plt.tight_layout()
plt.savefig(f"images/paper_a_1_monologic_all_features_before_treatment.png")
plt.close()
print("Generated plots for all features in monologic texts")
print("--------------------------------------------------------------")

# 2. Dialogic Texts, make a visualization for each feature
print("Generating plots for all features in dialogic texts...")
f, axes = plt.subplots(len(features), 3, figsize=(20, 4 * len(features)), sharex=False)

for i, feature in enumerate(features):
  skewness_dialogic = df_dialogic[feature].skew()
  print(f"- Skewness of {feature}: {skewness_dialogic}")

  # Histogram and Density Plot
  sns.histplot(df_dialogic[feature], color="r", ax=axes[i, 0], kde=True)
  axes[i, 0].set_title('Histogram - ' + feature)
  axes[i, 0].set_xlabel(f"Skewness: {skewness_dialogic:.2f}")

  # Q-Q Plot
  stats.probplot(df_dialogic[feature], dist="norm", plot=axes[i, 1])
  axes[i, 1].set_title('QQ Plot - ' + feature)

  # Boxplot
  sns.boxplot(x=df_dialogic[feature], ax=axes[i, 2])
  axes[i, 2].set_title('Boxplot - ' + feature)

plt.tight_layout()
plt.savefig(f"images/paper_a_2_dialogic_all_features_before_treatment.png")
plt.close()
print("Generated plots for all features in dialogic texts")
print("--------------------------------------------------------------")

""" ##########################################################
2. Test for Normality using Shapiro-Wilk test 
#######################################################"""

print("Testing for normality of each feature in each class...")
# Initialize the result dictionary
normality_tests = defaultdict(dict)

# Apply the Shapiro-Wilk test to each feature
for feature in features:
  print(f"\n- Testing for normality of {feature}...")
  _, p_value_monologic = shapiro(df_monologic[feature].apply(np.mean))
  _, p_value_dialogic = shapiro(df_dialogic[feature].apply(np.mean))

  # Store the p-values for each feature and class
  normality_tests[feature]['monologic'] = p_value_monologic
  normality_tests[feature]['dialogic'] = p_value_dialogic

# Create a DataFrame from the dictionary
normality_tests_df = pd.DataFrame(normality_tests)


# evaluate if any feature is normally distributed
def is_normal(p_value, alpha=0.05):
  if p_value > alpha:
    return True
  else:
    return False


# collect features that are normally distributed by both classes
not_normal_features_monologic = []
not_normal_features_dialogic = []

for feature in features:
  if not is_normal(normality_tests_df[feature]['monologic']):
    not_normal_features_monologic.append(feature)
  if not is_normal(normality_tests_df[feature]['dialogic']):
    not_normal_features_dialogic.append(feature)

print("Features that are not normally distributed by both classes:")
print(f"Monologic: {not_normal_features_monologic}")
print(f"Dialogic: {not_normal_features_dialogic}")
print()

""" ##########################################################
3. Apply transformations to features that are not normally distributed
   - Apply transformations to make the data more normal
   - Apply outlier handling to make the data more robust
#######################################################"""

print("Applying transformations to features that are not normally distributed...")


# Function to adjust outliers
def adjust_outliers(values):
  q1 = np.percentile(values, 25)
  q3 = np.percentile(values, 75)
  iqr = q3 - q1
  lower_bound = q1 - 1.5 * iqr
  upper_bound = q3 + 1.5 * iqr
  return np.clip(values, lower_bound, upper_bound)


# Initialize a MinMaxScaler
# scaler = MinMaxScaler()

# Initialize a new dataframe to store processed values
df_processed = df_flatten.copy()

for feature in features:
  # Apply log transformation to the "non-normal" feature (column)
  feature_transformed, _ = boxcox(df_processed[feature].values + 1)
  df_processed[feature] = feature_transformed
  print("- Applied log transformation to " + feature + "...")
  # Apply the IQR method to adjust outliers
  df_processed[feature] = adjust_outliers(df_processed[feature].values)
  print("- Applied IQR method to adjust outliers in " + feature + "...")
  # Apply Min-Max scaling to the transformed feature (column)
  # df_processed[feature] = scaler.fit_transform(df_processed[feature].values.reshape(-1, 1))

# Print a random sample of the pre- and post-transformation data
print()
print(df_flatten["personal_pronoun_d"].head(10))
print(df_processed["personal_pronoun_d"].head(10))
print()

""" ##########################################################
4. Test for Normality using Shapiro-Wilk test - After treatment
#######################################################"""

print("Testing for normality of each feature in each class AFTER transformation...")

# Divide dataset into two classes: monologic and dialogic
del df_monologic
del df_dialogic
df_monologic = df_processed[df_processed["label"] == "monologic"].copy()
df_dialogic = df_processed[df_processed["label"] == "dialogic"].copy()

# Initialize the result dictionary
normality_tests = defaultdict(dict)

# Apply the Shapiro-Wilk test to each feature
for feature in features:
  _, p_value_monologic = shapiro(df_monologic[feature].apply(np.mean))
  _, p_value_dialogic = shapiro(df_dialogic[feature].apply(np.mean))

  # Store the p-values for each feature and class
  normality_tests[feature]['monologic'] = p_value_monologic
  normality_tests[feature]['dialogic'] = p_value_dialogic

# Create a DataFrame from the dictionary
normality_tests_df = pd.DataFrame(normality_tests)

# collect features that are normally distributed by both classes
not_normal_features_monologic = []
not_normal_features_dialogic = []

for feature in features:
  if not is_normal(normality_tests_df[feature]['monologic']):
    not_normal_features_monologic.append(feature)
  if not is_normal(normality_tests_df[feature]['dialogic']):
    not_normal_features_dialogic.append(feature)

print("Features that are not normally distributed by both classes AFTER transformation:")
print(f"Monologic: {not_normal_features_monologic}")
print(f"Dialogic: {not_normal_features_dialogic}")
print()

""" ##########################################################
5. Observe the distribution of each feature in each class AFTER transformation
#######################################################"""

# 1. Monologic Texts, make a visualization for each feature
print("Generating plots for all features in dialogic texts AFTER transformation...")
f, axes = plt.subplots(len(features), 3, figsize=(20, 4 * len(features)), sharex=False)
for i, feature in enumerate(features):
  skewness_monologic = df_monologic[feature].skew()
  print(f"- Skewness of {feature}: {skewness_monologic}")

  # Histogram and Density Plot
  sns.histplot(df_monologic[feature], color="b", ax=axes[i, 0], kde=True)
  axes[i, 0].set_title('Histogram - ' + feature)
  axes[i, 0].set_xlabel(f"Skewness: {skewness_monologic:.2f}")

  # Q-Q Plot
  stats.probplot(df_monologic[feature], dist="norm", plot=axes[i, 1])
  axes[i, 1].set_title('QQ Plot - ' + feature)

  # Boxplot
  sns.boxplot(x=df_monologic[feature], ax=axes[i, 2])
  axes[i, 2].set_title('Boxplot - ' + feature)

plt.tight_layout()
plt.savefig(f"images/paper_a_3_monologic_all_features_after_treatment.png")
plt.close()
print("Generated plots for all features in monologic texts")
print("--------------------------------------------------------------")

# 2. Dialogic Texts, make a visualization for each feature
print("Generating plots for all features in dialogic texts AFTER transformation...")
f, axes = plt.subplots(len(features), 3, figsize=(20, 4 * len(features)), sharex=False)
for i, feature in enumerate(features):
  skewness_dialogic = df_dialogic[feature].skew()
  print(f"- Skewness of {feature}: {skewness_dialogic}")

  # Histogram and Density Plot
  sns.histplot(df_dialogic[feature], color="r", ax=axes[i, 0], kde=True)
  axes[i, 0].set_title('Histogram - ' + feature)
  axes[i, 0].set_xlabel(f"Skewness: {skewness_dialogic:.2f}")

  # Q-Q Plot
  stats.probplot(df_dialogic[feature], dist="norm", plot=axes[i, 1])
  axes[i, 1].set_title('QQ Plot - ' + feature)

  # Boxplot
  sns.boxplot(x=df_dialogic[feature], ax=axes[i, 2])
  axes[i, 2].set_title('Boxplot - ' + feature)

plt.tight_layout()
plt.savefig(f"images/paper_a_4_dialogic_all_features_after_treatment.png")
plt.close()
print("Generated plots for all features in dialogic texts")
print("--------------------------------------------------------------")

""" ##########################################################
6. Save the processed dataset to JSONL file
#######################################################"""

merged_dataset = []
for index, datapoint in df_processed.iterrows():
  row = {
    "id": datapoint["id"],
    "label": datapoint["label"],
    "sentence_length": datapoint["sentence_length"],
    "word_length": datapoint["word_length"],
    "sentence_complexity": datapoint["sentence_complexity"],
    "personal_pronoun_d": datapoint["personal_pronoun_d"],
    "passive_voice_d": datapoint["passive_voice_d"],
    "nominalization_d": datapoint["nominalization_d"],
    "lexical_d": datapoint["lexical_d"],
    "interjection_d": datapoint["interjection_d"],
    "modal_verb_d": datapoint["modal_verb_d"],
    "discourse_marker_d": datapoint["discourse_marker_d"]
  }
  merged_dataset.append(row)

# Split the dataset into train and test using the length of the original train dataset
train_dataset = merged_dataset[:length_train_dataset]
test_dataset = merged_dataset[length_train_dataset:]

# Empty the JSONL file
empty_json_file(output_dataset_train)
empty_json_file(output_dataset_test)

save_jsonl_file(train_dataset, output_dataset_train)
save_jsonl_file(test_dataset, output_dataset_test)

print()
print("Saved the processed dataset to JSONL file")
