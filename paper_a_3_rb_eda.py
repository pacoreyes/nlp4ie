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
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.stats import probplot
from scipy import stats
# from gspread_dataframe import set_with_dataframe
# from db import spreadsheet_2
from scipy.stats import shapiro, boxcox
from collections import defaultdict
# from pprint import pprint
# from sklearn.preprocessing import PowerTransformer, StandardScaler

from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file

ms_data = load_jsonl_file("shared_data/paper_a_2_feature_extraction_sliced.jsonl")

# Convert the data into a DataFrame
df = pd.json_normalize(ms_data, max_level=0)

features = ["sentence_length", "word_length", "sentence_complexity", "personal_pronoun_use",
            "passive_voice_use", "nominalization_use", "lexical_density", "interjection_use", "modal_verb_use",
            "discourse_markers_use"]

# copy the dataframe
df_flatten = df.copy()

# Flatten the list of values in each cell to a single mean value
for feature in features:
  df_flatten[feature] = df_flatten[feature].apply(lambda x: np.mean([float(i) for i in x]))

# Divide dataset into two classes: monologic and dialogic
df_monologic = df_flatten[df_flatten["discourse_type"] == 0].copy()
df_dialogic = df_flatten[df_flatten["discourse_type"] == 1].copy()

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
plt.savefig(f"shared_images/paper_a_1_monologic_all_features_before_treatment.png")
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
plt.savefig(f"shared_images/paper_a_2_dialogic_all_features_before_treatment.png")
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
  print(f"- Testing for normality of {feature}...")
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
  df_processed[feature], _ = boxcox(df_processed[feature] + 1)
  print("- Applied log transformation to " + feature + "...")
  # Apply the IQR method to adjust outliers
  df_processed[feature] = adjust_outliers(df_processed[feature].values)
  print("- Applied IQR method to adjust outliers in " + feature + "...")
  # Apply Min-Max scaling to the transformed feature (column)
  # df_processed[feature] = scaler.fit_transform(df_processed[feature].values.reshape(-1, 1))

# Print a random sample of the pre- and post-transformation data
print()
print(df_flatten["personal_pronoun_use"].head(10))
print(df_processed["personal_pronoun_use"].head(10))
print()

""" ##########################################################
4. Test for Normality using Shapiro-Wilk test - After treatment
#######################################################"""

print("Testing for normality of each feature in each class AFTER transformation...")

# Divide dataset into two classes: monologic and dialogic
del df_monologic
del df_dialogic
df_monologic = df_processed[df_processed["discourse_type"] == 0].copy()
df_dialogic = df_processed[df_processed["discourse_type"] == 1].copy()

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
plt.savefig(f"shared_images/paper_a_3_monologic_all_features_after_treatment.png")
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
plt.savefig(f"shared_images/paper_a_4_dialogic_all_features_after_treatment.png")
plt.close()
print("Generated plots for all features in dialogic texts")
print("--------------------------------------------------------------")

""" ##########################################################
6. Save the processed dataset to JSONL file
#######################################################"""

# Empty the JSONL file
empty_json_file('shared_data/paper_a_3_feature_extraction_sliced_transformed.jsonl')

for index, row in df_processed.iterrows():
  slots = {
    "id": row["id"],
    "discourse_type": row["discourse_type"],
    "sentence_length": row["sentence_length"],
    "word_length": row["word_length"],
    "sentence_complexity": row["sentence_complexity"],
    # "pronoun_use": row["pronoun_use"],
    "personal_pronoun_use": row["personal_pronoun_use"],
    "passive_voice_use": row["passive_voice_use"],
    "nominalization_use": row["nominalization_use"],
    "lexical_density": row["lexical_density"],
    "interjection_use": row["interjection_use"],
    "modal_verb_use": row["modal_verb_use"],
    "discourse_markers_use": row["discourse_markers_use"]
  }
  # corpus.append(slots)
  save_row_to_jsonl_file(slots, 'shared_data/paper_a_3_feature_extraction_sliced_transformed.jsonl')
print()
print("Saved the processed dataset to JSONL file")
