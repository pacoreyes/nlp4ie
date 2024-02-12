from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lib.utils import load_jsonl_file

# Initialize the threshold for feature pruning
PRUNING_THRESHOLD = 0.8

# Initialize the class names
CLASS_NAMES = ["monologic", "dialogic"]

# Set the seed for reproducibility
np.random.seed(42)


# Function to find strongly correlated pairs
def find_strongly_correlated_pairs(corr_matrix, _features, threshold=0.5):
  strongly_correlated_pairs = []
  for i in range(len(_features)):
    for j in range(i + 1, len(_features)):  # i+1 to avoid self-correlation and duplicates
      if abs(corr_matrix.iloc[i, j]) > threshold:
        strongly_correlated_pairs.append((_features[i], _features[j], corr_matrix.iloc[i, j]))
  return strongly_correlated_pairs


dataset_train = load_jsonl_file("shared_data/dataset_1_5_1a_train_features_transformed.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_1_5_1a_test_features_transformed.jsonl")

features_names = ["Sentence length", "Word length", "Sentence complexity", "Personal pronoun density",
                  "Passive voice density", "Nominalization density", "Lexical density", "Interjection density",
                  "Modal verb density", "Discourse markers density"]

features = ["sentence_length", "word_length", "sentence_complexity",
            "personal_pronoun_d", "passive_voice_d", "nominalization_d", "lexical_d",
            "interjection_d", "modal_verb_d", "discourse_markers_d"]

# Convert the data into a DataFrame
df = pd.json_normalize(dataset_train, max_level=0)

# Divide dataset into two classes: monologic and dialogic
df_monologic = df[df["label"] == "monologic"].copy()
df_dialogic = df[df["label"] == "dialogic"].copy()

# Remove the "id" and "label" columns
df_monologic.drop(["id", "label"], axis=1, inplace=True)
df_dialogic.drop(["id", "label"], axis=1, inplace=True)

"""
##########################################
 Correlation analysis for both classes 
##########################################
"""

# Calculate correlation matrices
print("Analyzing correlation coefficients for both classes...")
corr_monologic = df_monologic.corr(method="pearson")
corr_dialogic = df_dialogic.corr(method="pearson")

# Set a color map
# colors = ["lightseagreen", "white", "orangered"]  # https://matplotlib.org/stable/gallery/color/named_colors.html
# cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

print("Plotting correlation matrices...")
# Create a figure with two subplots
fig1, ax = plt.subplots(ncols=2, figsize=(20, 8))

# Heatmap for monologic texts
sns.heatmap(corr_monologic, cmap='Greens', annot=True, ax=ax[0])
ax[0].set_title('Correlation Matrix of Linguistic features for the Monologic Class')
ax[0].set_xticklabels(features_names)
ax[0].set_yticklabels(features_names)

# Heatmap for dialogic texts
sns.heatmap(corr_dialogic, cmap='Greens', annot=True, ax=ax[1])
ax[1].set_title('Correlation Matrix of Linguistic features for the Dialogic Class')
ax[1].set_xticklabels(features_names)
ax[1].set_yticklabels(features_names)
plt.tight_layout()
plt.savefig('images/paper_a_5_correlation_heatmap.png')

# Save correlation matrices in two sheets of a single Excel file
with pd.ExcelWriter('shared_data/paper_a_correlation_matrices.xlsx') as writer:
  corr_monologic.to_excel(writer, sheet_name='monologic')
  corr_dialogic.to_excel(writer, sheet_name='dialogic')

# print(corr_monologic)
# print(corr_dialogic)
print()
print("\nGenerated correlation matrices.\n")

# Create a list of strongly correlated feature pairs for monologic and dialogic classes
monologic_pairs = find_strongly_correlated_pairs(corr_monologic, features, PRUNING_THRESHOLD)
dialogic_pairs = find_strongly_correlated_pairs(corr_dialogic, features, PRUNING_THRESHOLD)

print("Strongly correlated pairs for monologic texts:")
pprint(monologic_pairs)
print("\nStrongly correlated pairs for dialogic texts:")
pprint(dialogic_pairs)

"""# Create a list of the remaining features after pruning
remaining_features_monologic = [f for f in features if f not in [pair[0] for pair in monologic_pairs]]
remaining_features_dialogic = [f for f in features if f not in [pair[0] for pair in dialogic_pairs]]

print("\nRemaining features for monologic texts:")
pprint(remaining_features_monologic)
print("\nRemaining features for dialogic texts:")
pprint(remaining_features_dialogic)"""

# Initialize sets for the features to be pruned for each class
prune_features_monologic = set()
prune_features_dialogic = set()

# Populate the sets with features that are part of strongly correlated pairs
for pair in monologic_pairs:
    prune_features_monologic.update([pair[0], pair[1]])

for pair in dialogic_pairs:
    prune_features_dialogic.update([pair[0], pair[1]])

# Determine the remaining features by subtracting the pruned features from the total set of features
remaining_features_monologic = [feature for feature in features if feature not in list(prune_features_monologic)]
remaining_features_dialogic = [feature for feature in features if feature not in list(prune_features_dialogic)]

print("Remaining features for monologic texts:")
pprint(remaining_features_monologic)
print("\nRemaining features for dialogic texts:")
pprint(remaining_features_dialogic)


def find_weakly_correlated_pairs(corr_matrix, features, threshold):
  weakly_correlated_pairs = []
  for i in range(len(features)):
    for j in range(i + 1, len(features)):  # Ensure we don't repeat pairs or compare features with themselves
      # Access the correlation value using .iloc for DataFrame
      corr_value = corr_matrix.iloc[i, j]
      if abs(corr_value) < threshold:
        weakly_correlated_pairs.append((features[i], features[j], corr_value))
  return weakly_correlated_pairs


# Function to find the highest positive and negative correlation values
def find_extreme_correlations(pairs):
    highest_positive = float('-inf')
    lowest_negative = float('inf')
    for _, _, corr_value in pairs:
        if corr_value > 0:
            highest_positive = max(highest_positive, corr_value)
        elif corr_value < 0:
            lowest_negative = min(lowest_negative, corr_value)
    return highest_positive, lowest_negative


# Use the function to find weakly correlated pairs for both classes
weakly_correlated_pairs_monologic = find_weakly_correlated_pairs(corr_monologic, features, PRUNING_THRESHOLD)
weakly_correlated_pairs_dialogic = find_weakly_correlated_pairs(corr_dialogic, features, PRUNING_THRESHOLD)

# Get the highest positive and negative correlation values for each class
highest_monologic_pos_threshold, lowest_monologic_neg_threshold = find_extreme_correlations(weakly_correlated_pairs_monologic)
highest_dialogic_pos_threshold, lowest_dialogic_neg_threshold = find_extreme_correlations(weakly_correlated_pairs_dialogic)

print("Monologic texts - Highest positive correlation:", highest_monologic_pos_threshold)
print("Monologic texts - Highest negative correlation:", lowest_monologic_neg_threshold)
print("Dialogic texts - Highest positive correlation:", highest_dialogic_pos_threshold)
print("Dialogic texts - Highest negative correlation:", lowest_dialogic_neg_threshold)


""" ##########################################
  Feature Pruning
########################################## """

# Features to remove from the monologic class based on correlation analysis
features_to_prune_monologic = ["sentence_length", "lexical_d", "nominalization_d"]
# Features to remove from the dialogic class based on correlation analysis
features_to_prune_dialogic = ["sentence_length", "lexical_d", "nominalization_d", "sentence_complexity",
                              "discourse_markers_d"]


monologic_features_names = [
  "Word length",
  "Sentence complexity",
  "Personal pronoun density",
  "Passive voice density",
  "Interjection density",
  "Modal verb density",
  "Discourse markers density"
  ]

monologic_features = [
  'word_length',
  'sentence_complexity',
  'personal_pronoun_d',
  'passive_voice_d',
  'interjection_d',
  'modal_verb_d',
  'discourse_markers_d'
]

dialogic_features_names = [
  "Word length",
  "Personal pronoun density",
  "Passive voice density",
  "Interjection density",
  "Modal verb density",
]

dialogic_features = [
  'word_length',
  'personal_pronoun_d',
  'passive_voice_d',
  'interjection_d',
  'modal_verb_d',
]

"""pprint(df_monologic.head())
pprint(df_dialogic.head())"""

# Remove pruned features
df_monologic = df_monologic.drop(columns=features_to_prune_monologic)
df_dialogic = df_dialogic.drop(columns=features_to_prune_dialogic)

print("Analyzing correlation coefficients for both classes...")
corr_monologic = df_monologic.corr()
corr_dialogic = df_dialogic.corr()

print(corr_monologic)
print(corr_dialogic)

print("Plotting correlation matrices...")
# Create a figure with two subplots
fig1, ax = plt.subplots(ncols=2, figsize=(20, 8))


# Heatmap for monologic texts
sns.heatmap(corr_monologic, cmap='Greens', annot=True, ax=ax[0])
ax[0].set_title('Correlation Matrix of Linguistic features for the Monologic Class (after feature pruning)')
ax[0].set_xticklabels(monologic_features_names)
ax[0].set_yticklabels(monologic_features_names)

# Heatmap for dialogic texts
sns.heatmap(corr_dialogic, cmap='Greens', annot=True, ax=ax[1])
ax[1].set_title('Correlation Matrix of Linguistic features for the Dialogic Class (after feature pruning)')
ax[1].set_xticklabels(dialogic_features_names)
ax[1].set_yticklabels(dialogic_features_names)

plt.tight_layout()
plt.savefig('images/paper_a_5_correlation_heatmap_after_pruning.png')

# Save correlation matrices in two sheets of a single Excel file
with pd.ExcelWriter('shared_data/paper_a_correlation_matrices_after_pruning.xlsx') as writer:
  corr_monologic.to_excel(writer, sheet_name='monologic')
  corr_dialogic.to_excel(writer, sheet_name='dialogic')

# print(corr_monologic)
# print(corr_dialogic)
print()
print("\nGenerated correlation matrices after pruning.\n")


""" ##########################################
 Rules, Weights and Thresholds 
########################################## """

# Threshold values for correlation coefficients
"""MONOLOGIC_POSITIVE_THRESHOLD = 0.85
MONOLOGIC_NEGATIVE_THRESHOLD = -0.85
DIALOGIC_POSITIVE_THRESHOLD = 0.85
DIALOGIC_NEGATIVE_THRESHOLD = -0.85"""

"""MONOLOGIC_POSITIVE_THRESHOLD = 0.77
MONOLOGIC_NEGATIVE_THRESHOLD = -0.63
DIALOGIC_POSITIVE_THRESHOLD = 0.82
DIALOGIC_NEGATIVE_THRESHOLD = -0.16"""

"""MONOLOGIC_POSITIVE_THRESHOLD = 0.79
MONOLOGIC_NEGATIVE_THRESHOLD = -0.49
DIALOGIC_POSITIVE_THRESHOLD = 0.75
DIALOGIC_NEGATIVE_THRESHOLD = -0.079"""

"""for i in range(len(corr_monologic)):
  for j in range(i + 1, len(corr_monologic)):
    if abs(corr_monologic.iloc[i, j]) >= highest_monologic_pos_threshold or corr_monologic.iloc[
      i, j] <= lowest_monologic_neg_threshold:
      monologic_pairs.append((corr_monologic.index[i], corr_monologic.columns[j], corr_monologic.iloc[i, j]))

for i in range(len(corr_dialogic)):
  for j in range(i + 1, len(corr_dialogic)):
    if abs(corr_dialogic.iloc[i, j]) >= highest_dialogic_pos_threshold or corr_dialogic.iloc[
      i, j] <= lowest_dialogic_neg_threshold:
      dialogic_pairs.append((corr_dialogic.index[i], corr_dialogic.columns[j], corr_dialogic.iloc[i, j]))

pprint(monologic_pairs)
pprint(dialogic_pairs)"""
"""
#######################################
  Testing the Rule-Based Model (RBM)
#######################################

The first step is loading original data flattening Feature Values for Testing the Model.
We will simulates a real-world scenario where new, unseen data is input into the model.
Then, the data is flattened to a single mean value for each feature because
it’s simple and keeps the data scale relatively consistent, although it may lose potential informative variability 
in the data.


# Load the dataset
data = pd.read_json("shared_data/paper_a_2_feature_extraction.jsonl", lines=True)

# From the whole dataset, use the test_ids to select the test set
data = data[data["id"].isin(test_ids)].copy()

# print(data.head())

# Flatten the list of values in each cell to a single mean value
for feature in features:
  data[feature] = data[feature].apply(lambda x: np.mean([float(_i) for _i in x]))

# Calculate summary statistics for each feature for each text
datapoint = data.copy()
for feature in data.columns[2:]:
  datapoint[feature + "_mean"] = datapoint[feature].apply(np.mean)
  percentiles = datapoint[feature].apply(lambda lst: [np.percentile(lst, p) for p in [25, 50, 75]])
  for i, percentile in enumerate([25, 50, 75]):
    datapoint[feature + f"_p{percentile}"] = [p[i] for p in percentiles]

# Drop the original feature columns
datapoint = datapoint.drop(columns=data.columns[2:])


# Function to calculate the range of values of feature B for texts where feature A is in a given percentile
def calculate_feature_ranges(_df, feature_a, feature_b, _percentiles=None):
  if _percentiles is None:
    _percentiles = [25, 50, 75]
  ranges = {}
  for p in _percentiles:
    # Get the texts where feature A is in the given percentile
    p_value = np.percentile(_df[feature_a + "_mean"], p)
    mask = (_df[feature_a + "_mean"] <= p_value)
    # Calculate the range of feature B for these texts
    min_a = _df[mask][feature_b + "_mean"].min()
    max_b = _df[mask][feature_b + "_mean"].max()
    ranges[p] = [min_a, max_b]
  return ranges


# Establish the rules for each class and for each pair of strongly correlated features
rules = {"monologic": {}, "dialogic": {}}

for pairs, class_label in zip([monologic_pairs, dialogic_pairs], ["monologic", "dialogic"]):
  df_class = datapoint[datapoint["discourse_type"] == (class_label == "dialogic")]
  for pair in pairs:
    feature_A, feature_B, _ = pair
    rules[class_label][f"{feature_A}-{feature_B}"] = calculate_feature_ranges(df_class, feature_A, feature_B)


# Function to predict the class of a new text
def predict_discourse_type(_text_features):
  # Initialize the scores
  scores = {"monologic": 0, "dialogic": 0}

  # For each class and each pair of strongly correlated features
  for _class_label in ["monologic", "dialogic"]:
    for _pair in rules[_class_label]:
      # Extract the names of the features
      feature_a, feature_b = _pair.split("-")
      # Calculate the mean value of each feature for the text
      mean_a = np.mean(_text_features[feature_a])
      mean_b = np.mean(_text_features[feature_b])
      # For each percentile of the distribution of feature A
      for p in [25, 50, 75]:
        # If the mean value of feature A is in the given percentile
        if mean_a <= np.percentile(_text_features[feature_a], p):
          # If the mean value of feature B is within the range established by the rule
          if rules[_class_label][_pair][p][0] <= mean_b <= rules[_class_label][_pair][p][1]:
            # Increase the score of the class
            scores[_class_label] += 1

  # Return the class with the highest score
  return max(scores, key=scores.get)


# Predict the class for each text in the dataset
predictions = []
for i in range(len(data)):
  text_features = {feature: data.iloc[i][feature] for feature in data.columns[2:]}
  predictions.append(predict_discourse_type(text_features))

# Convert the class labels to binary values for the calculation of metrics
actual = data["discourse_type"].apply(lambda x: 1 if x else 0).values
predictions_binary = [1 if p == "dialogic" else 0 for p in predictions]
# print(f"Predictions: {predictions}")

# Calculate accuracy, precision, recall, F1 score, the AUC ROC, and Matthews correlation coefficient
accuracy = accuracy_score(actual, predictions_binary)
precision = precision_score(actual, predictions_binary)
recall = recall_score(actual, predictions_binary)
f1 = f1_score(actual, predictions_binary)
roc_auc = roc_auc_score(actual, predictions_binary)
mcc = matthews_corrcoef(actual, predictions_binary)
# Create the confusion matrix
cm = confusion_matrix(actual, predictions_binary)
cm_df = pd.DataFrame(cm, index=["monologic", "dialogic"], columns=["monologic", "dialogic"])

# Plot confusion matrix
print("Plotting confusion matrices...")
plot_confusion_matrix(actual, predictions_binary,
                      class_names,
                      "paper_a_5_rb_model_confusion_matrix.png",
                      "Confusion Matrix for Rule-based Model",
                      values_fontsize=22)

print(f"\nThresholds: {MONOLOGIC_POSITIVE_THRESHOLD}/{MONOLOGIC_NEGATIVE_THRESHOLD} "
      f"- {DIALOGIC_POSITIVE_THRESHOLD}/{DIALOGIC_NEGATIVE_THRESHOLD}\n")
print("Model: Rule-Based Model (RBM)")
print(f"- Accuracy: {accuracy:.3f}")
print(f"- Precision: {precision:.3f}")
print(f"- Recall: {recall:.3f}")
print(f"- F1-Score: {f1:.3f}")
print(f"- AUC ROC: {roc_auc:.3f}")
print(f"- Matthews Correlation Coefficient (MCC): {mcc:.3f}")
print(f"- Confusion Matrix:\n{cm_df}")"""

"""

Thresholds: 0.77/-0.63 - 0.82/-0.16

Model: Rule-Based Model (RBM)
- Accuracy: 0.94
- Precision: 0.94
- Recall: 0.94
- F1-Score: 0.94
- AUC ROC: 0.94
- Matthews Correlation Coefficient: 0.88
- Confusion Matrix:
           monologic  dialogic
monologic         47         3
dialogic           3        47

----------- last version oct 2023 ----------------

Feature pruning threshold: 0.8

Thresholds: 0.79/-0.78 - 0.75/-0.28

Model: Rule-Based Model (RBM)
- Accuracy: 0.90
- Precision: 0.83
- Recall: 1.00
- F1-Score: 0.91
- AUC ROC: 0.90
- Matthews Correlation Coefficient (MCC): 0.82
- Confusion Matrix:
           monologic  dialogic
monologic          8         2
dialogic           0        10

----------- last version oct 2023 ----------------

Thresholds: 0.79/-0.49 - 0.75/-0.079

Model: Rule-Based Model (RBM)
- Accuracy: 0.900
- Precision: 0.833
- Recall: 1.000
- F1-Score: 0.909
- AUC ROC: 0.900
- Matthews Correlation Coefficient (MCC): 0.816
- Confusion Matrix:
           monologic  dialogic
monologic          8         2
dialogic           0        10

"""
