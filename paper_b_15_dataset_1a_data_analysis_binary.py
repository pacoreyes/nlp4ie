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
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import chi2
from sklearn import preprocessing

from lib.utils import load_jsonl_file, save_jsonl_file
from lib.visualizations import plot_correlation_heatmap_double

# Load the data
dataset_train = load_jsonl_file("shared_data/dataset_2_3_1a_train_features_aggregated.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_2_3_1a_test_features_aggregated.jsonl")

class_names = ["Support", "Oppose"]

features = {
  "positive_affect": "Positive Affect",
  "negative_affect": "Negative Affect",
  "epistemic_certainty": "Epistemic Certainty",
  "epistemic_doubt": "Epistemic Doubt",
  "emphatic": "Emphatics",
  "hedge": "Hedges",
  "pro": "Pro Polarity",
  "con": "Con Polarity",
  "modal_verb": "Modal Verb",
}

# Make list of keys and list of values
ms_features = list(features.keys())
feature_names = list(features.values())

# Load the data of features into a DataFrame
df_train_dataset = pd.DataFrame(dataset_train)

# Drop the "semantic_frames" column
# df_train_dataset = df_train_dataset.drop(columns=['pro', "con"], axis=1)

# Split the df into two dataframes: one for support and one for oppose
df_support = df_train_dataset[df_train_dataset['label'] == "support"]
df_oppose = df_train_dataset[df_train_dataset['label'] == "oppose"]

# Extract the feature columns from the dataframes
df_support = df_support[ms_features]
df_oppose = df_oppose[ms_features]

corr_support_matrix = df_support.corr(method='pearson')
corr_oppose_matrix = df_oppose.corr(method='pearson')

"""print(corr_support_matrix)
print(corr_oppose_matrix)"""

# Round the correlation matrices to three decimal places
corr_matrix_support = corr_support_matrix.round(3)
corr_matrix_oppose = corr_oppose_matrix.round(3)

# Convert correlation matrices to HTML tables
html_table = corr_matrix_support.to_html()

# Save the HTML table to a file
with open("shared_data/paper_b_3_support_correlation_matrix_table.html", "w") as file:
  file.write(html_table)

html_table = corr_matrix_oppose.to_html()

# Save the HTML table to a file
with open("shared_data/paper_b_4_oppose_correlation_matrix_table.html", "w") as file:
  file.write(html_table)

# Output the correlation matrices
print("Correlation matrix for support data:")
print(corr_matrix_support)
print("\nCorrelation matrix for oppose data:")
print(corr_matrix_oppose)

corr_matrix_support.to_json("shared_data/paper_b_5_correlation_matrix_support.json")
corr_matrix_oppose.to_json("shared_data/paper_b_6_correlation_matrix_oppose.json")

# Plot the correlation matrices
plot_correlation_heatmap_double(
  corr_matrix_support,
  corr_matrix_oppose,
  'Correlation Matrix of Linguistic features for the Support Class',
  'Correlation Matrix of Linguistic features for the Oppose Class',
  feature_names,
  "images/paper_b/paper_b_7_correlation_matrix_heatmap.png")


""""2. Bar plots of the features for the support and oppose classes"""

# Create bar graphs for individual feature distribution
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
axes = axes.flatten()

for i, (feature, pretty_name) in enumerate(features.items()):
  ax = axes[i]
  df_train_dataset[feature].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
  ax.set_title(f'Distribution of {pretty_name}')
  ax.set_xlabel('Values')
  ax.set_ylabel('Count')
  ax.set_xticklabels(['0', '1'], rotation=0)
  # Add a legend to the plot with the class labels
  ax.legend(title='Label', labels=['Support', 'Oppose'])

plt.tight_layout()
plt.savefig("images/paper_b/paper_b_8_feature_distribution.png")

"""
# Setting up the plots for individual feature distribution and cross-tabulation with the class labels

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
feature_columns = ['positive_affect', 'negative_affect', 'epistemic_certainty', 'epistemic_doubt', 
                   'emphatic', 'hedge', 'modal_verb']

# Plot histograms for each binary feature
for i, feature in enumerate(feature_columns):
    row, col = divmod(i, 3)
    sns.countplot(x=feature, hue='label', data=data, ax=axes[row, col], palette='viridis')
    axes[row, col].set_title(f'Distribution of {feature} by Class')
    axes[row, col].set_xlabel('')
    axes[row, col].set_ylabel('Count')
    axes[row, col].legend(title='Class')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
"""

# Create normalized stacked bar charts for cross-tabulation
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
axes = axes.flatten()

for i, (feature, pretty_name) in enumerate(features.items()):
  ax = axes[i]
  crosstab = pd.crosstab(df_train_dataset['label'], df_train_dataset[feature], normalize='index')
  crosstab.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'orange'])
  ax.set_title(f'Normalized Co-occurrence with {pretty_name}')
  ax.set_xlabel('Label')
  ax.set_ylabel('Proportion')
  ax.legend(title='Value', labels=['0', '1'])

plt.tight_layout()
plt.savefig("images/paper_b/paper_b_9_feature_cross_tabulation.png")


"""3. Feature Importance Analysis using Chi-Square Test"""

# Encode the 'label' column
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df_train_dataset['label'])

# Calculate chi-squared stats between each non-label feature and the labels
chi_scores, p_values = chi2(df_train_dataset.drop(['id', 'label'], axis=1), encoded_labels)

# Create a DataFrame to display feature importances
feature_importance = pd.DataFrame({
    'Feature': list(features.values()),
    'Chi-Squared': chi_scores,
    'P-Value': p_values
}).sort_values(by='Chi-Squared', ascending=False)

print(feature_importance)

fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Chi-squared values bar chart
ax[0].barh(feature_importance['Feature'], feature_importance['Chi-Squared'], color='skyblue')
ax[0].set_xlabel('Chi-Squared Values')
ax[0].set_title('Chi-Squared Values by Feature')
ax[0].invert_yaxis()  # Invert to match the order in the second plot

# P-value plot with a significance threshold line
ax[1].barh(feature_importance['Feature'], feature_importance['P-Value'], color='orange')
ax[1].set_xlabel('P-Value')
ax[1].set_title('P-Values by Feature')
ax[1].invert_yaxis()
ax[1].axvline(x=0.05, color='red', linestyle='--', label='Significance Threshold (p = 0.05)')
ax[1].legend()

plt.tight_layout()
plt.savefig("images/paper_b/paper_b_10_chi_squared_feature_importance.png")

