from pprint import pprint

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import seaborn as sns

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


data = pd.read_json("shared_data/dataset_1_5_1a_train_features.jsonl", lines=True)

# Step 1: Calculate mean values of features

# Calculate mean values of list-based features
features_to_summarize = [
    'sentence_length', 'word_length', 'sentence_complexity',
    'personal_pronoun_d', 'passive_voice_d', 'nominalization_d',
    'lexical_d', 'interjection_d', 'modal_verb_d', 'discourse_marker_d'
]

for feature in features_to_summarize:
    data[feature + '_mean'] = data[feature].apply(lambda x: sum(x) / len(x) if x else 0)

# Check the unique values of the 'label' column
label_unique_values = data['label'].unique()

"""# Display the unique values of the target variable and a sample of the summarized dataset
print(label_unique_values)
print(data.head())"""

# Step 2: Check for Outliers

summarized_features = [
    'sentence_length_mean', 'word_length_mean', 'sentence_complexity_mean',
    'personal_pronoun_d_mean', 'passive_voice_d_mean', 'nominalization_d_mean',
    'lexical_d_mean', 'interjection_d_mean', 'modal_verb_d_mean',
    'discourse_marker_d_mean'
]

# Detect outliers using IQR
outliers_summary = {}
for feature in summarized_features:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    outliers_summary[feature] = len(outliers)

# Plot box plots for each summarized feature for visual inspection of outliers
fig, axs = plt.subplots(len(summarized_features), 1, figsize=(10, 20))
for i, feature in enumerate(summarized_features):
    sns.boxplot(x=data[feature], ax=axs[i])
    axs[i].set_title(feature)

plt.tight_layout()
# plt.show()
plt.savefig("images/_paper_a_outliers_boxplot.png")

# Display the summary of outliers detected
print("Here's a summary of the number of outliers detected in each feature:")
pprint(outliers_summary)

# Step 3: inspect the distribution of features with outliers to determine the most appropriate transformation

# Separate the dataset based on the binary target variable classes
data_class_1 = data[data['label'] == label_unique_values[0]]  # First class
data_class_2 = data[data['label'] == label_unique_values[1]]  # Second class

# Plot histograms for each summarized feature by class
fig, axs = plt.subplots(len(summarized_features), 2, figsize=(15, 40), sharex='row', sharey='row')

for i, feature in enumerate(summarized_features):
    sns.histplot(data_class_1[feature], kde=True, ax=axs[i, 0], color="skyblue", label=label_unique_values[0])
    axs[i, 0].set_title(f"{feature} - {label_unique_values[0]}")
    axs[i, 0].legend()

    sns.histplot(data_class_2[feature], kde=True, ax=axs[i, 1], color="red", label=label_unique_values[1])
    axs[i, 1].set_title(f"{feature} - {label_unique_values[1]}")
    axs[i, 1].legend()

plt.tight_layout()
plt.show()


