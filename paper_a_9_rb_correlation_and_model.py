from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
  confusion_matrix, matthews_corrcoef, roc_auc_score

from lib.utils import load_jsonl_file
from lib.visualizations import plot_confusion_matrix

class_names = ["monologic", "dialogic"]

# Set the seed for reproducibility
np.random.seed(42)

ms_data = load_jsonl_file("shared_data/paper_a_3_feature_extraction_transformed.jsonl")

# Split data into training and test sets ensuring balance
ids_type_0 = [text["id"] for text in ms_data if text["discourse_type"] == 0]
ids_type_1 = [text["id"] for text in ms_data if text["discourse_type"] == 1]

# Calculate the number of samples for each discourse type
num_samples_0 = int(len(ms_data) * 0.1)
num_samples_1 = int(len(ms_data) * 0.1)

# Randomly sample ids from each group
test_ids_type_0 = np.random.choice(ids_type_0, size=num_samples_0, replace=False)
test_ids_type_1 = np.random.choice(ids_type_1, size=num_samples_1, replace=False)

# Concatenate the sampled ids to get the final test ids
test_ids = np.concatenate((test_ids_type_0, test_ids_type_1))

# remove from ms_data the datapoints included in the test set
ms_data = [text for text in ms_data if text["id"] not in test_ids]

"""features_names = ["Sentence length", "Word length", "Sentence complexity", "Personal pronoun use",
                  "Passive voice use", "Nominalization use", "Lexical density", "Interjection use", "Modal verb use",
                  "Discourse markers use"]

features = ["sentence_length", "word_length", "sentence_complexity",
            "personal_pronoun_use", "passive_voice_use", "nominalization_use", "lexical_density",
            "interjection_use", "modal_verb_use", "discourse_markers_use"]"""


features_names = [
  "Personal pronoun use",
  "Passive voice use",
  "Interjection use",
  "Modal verb use",
  "Discourse markers use"
  ]

features = [
  "personal_pronoun_use",
  "passive_voice_use",
  "interjection_use",
  "modal_verb_use",
  "discourse_markers_use"
  ]


# Convert the data into a DataFrame
df = pd.json_normalize(ms_data, max_level=0)

"""
##########################################
Features pruning
##########################################
"""
# Features to remove based on correlation analysis
features_to_remove = ["sentence_length", "lexical_density", "nominalization_use", "sentence_complexity", "word_length"]

# Remove identified features
df = df.drop(columns=features_to_remove)


# ----------------------------

# Divide dataset into two classes: monologic and dialogic
df_monologic = df[df["discourse_type"] == 0].copy()
df_dialogic = df[df["discourse_type"] == 1].copy()

# Remove the "id" and "discourse_type" columns
df_monologic.drop(["id", "discourse_type"], axis=1, inplace=True)
df_dialogic.drop(["id", "discourse_type"], axis=1, inplace=True)

"""
##########################################
 Correlation analysis for both classes 
##########################################
"""

# Calculate correlation matrices
print("Analyzing correlation coefficients for both classes...")
corr_monologic = df_monologic.corr()
corr_dialogic = df_dialogic.corr()

# Set a color map
# colors = ["lightseagreen", "white", "orangered"]  # https://matplotlib.org/stable/gallery/color/named_colors.html
# cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

print("Plotting correlation matrices...")
# Create a figure with two subplots
fig1, ax = plt.subplots(ncols=2, figsize=(20, 8))

# Heatmap for monologic texts
sns.heatmap(corr_monologic, cmap='Greens', annot=True, ax=ax[0])
ax[0].set_title('Correlation Matrix of Linguistic features for the Monologic Class (after feature pruning)')
ax[0].set_xticklabels(features_names)
ax[0].set_yticklabels(features_names)

# Heatmap for dialogic texts
sns.heatmap(corr_dialogic, cmap='Greens', annot=True, ax=ax[1])
ax[1].set_title('Correlation Matrix of Linguistic features for the Dialogic Class (after feature pruning)')
ax[1].set_xticklabels(features_names)
ax[1].set_yticklabels(features_names)
plt.tight_layout()
plt.savefig('shared_images/paper_a_5_correlation_heatmap.png')

# Save correlation matrices in two sheets of a single Excel file
with pd.ExcelWriter('shared_data/paper_a_correlation_matrices.xlsx') as writer:
  corr_monologic.to_excel(writer, sheet_name='monologic')
  corr_dialogic.to_excel(writer, sheet_name='dialogic')

# print(corr_monologic)
# print(corr_dialogic)
print()
print("\nGenerated correlation matrices.\n")

# from the correlation matrices, create a list of strongly correlated feature pairs for monologic and dialogic classes
monologic_pairs = []
dialogic_pairs = []

"""
##########################################
 Rules, Weights and Thresholds 
##########################################
"""

# Threshold values for correlation coefficients
"""MONOLOGIC_POSITIVE_THRESHOLD = 0.85
MONOLOGIC_NEGATIVE_THRESHOLD = -0.85
DIALOGIC_POSITIVE_THRESHOLD = 0.85
DIALOGIC_NEGATIVE_THRESHOLD = -0.85"""

"""MONOLOGIC_POSITIVE_THRESHOLD = 0.77
MONOLOGIC_NEGATIVE_THRESHOLD = -0.63
DIALOGIC_POSITIVE_THRESHOLD = 0.82
DIALOGIC_NEGATIVE_THRESHOLD = -0.16"""

MONOLOGIC_POSITIVE_THRESHOLD = 0.79
MONOLOGIC_NEGATIVE_THRESHOLD = -0.49
DIALOGIC_POSITIVE_THRESHOLD = 0.75
DIALOGIC_NEGATIVE_THRESHOLD = -0.079

for i in range(len(corr_monologic)):
  for j in range(i + 1, len(corr_monologic)):
    if abs(corr_monologic.iloc[i, j]) >= MONOLOGIC_POSITIVE_THRESHOLD or corr_monologic.iloc[
      i, j] <= MONOLOGIC_NEGATIVE_THRESHOLD:
      monologic_pairs.append((corr_monologic.index[i], corr_monologic.columns[j], corr_monologic.iloc[i, j]))

for i in range(len(corr_dialogic)):
  for j in range(i + 1, len(corr_dialogic)):
    if abs(corr_dialogic.iloc[i, j]) >= DIALOGIC_POSITIVE_THRESHOLD or corr_dialogic.iloc[
      i, j] <= DIALOGIC_NEGATIVE_THRESHOLD:
      dialogic_pairs.append((corr_dialogic.index[i], corr_dialogic.columns[j], corr_dialogic.iloc[i, j]))

pprint(monologic_pairs)
pprint(dialogic_pairs)

"""
#######################################
  Testing the Rule-Based Model (RBM)
#######################################

The first step is loading original data flattening Feature Values for Testing the Model.
We will simulates a real-world scenario where new, unseen data is input into the model.
Then, the data is flattened to a single mean value for each feature because
it’s simple and keeps the data scale relatively consistent, although it may lose potential informative variability 
in the data.
"""

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
print(f"- Confusion Matrix:\n{cm_df}")

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
