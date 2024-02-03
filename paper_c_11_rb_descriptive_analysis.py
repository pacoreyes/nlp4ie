"""
This script calculates the descriptive statistics for the features extracted from the measurements of linguistic
features extracted from the texts.
"""
import pandas as pd
# import numpy as np
# from scipy.stats import skew, kurtosis
# from typing import List
# from gspread_dataframe import set_with_dataframe
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import boxcox
from numpy import log1p
from pprint import pprint

from lib.utils import load_jsonl_file
from lib.semantic_frames import index_frame
# from db import spreadsheet_2
from pprint import pprint


def apply_transformation(feature):
  """
  Applies a transformation to a feature to attempt to normalize its distribution.
  The transformation to apply is chosen based on the nature of the data.
  """
  # Positive values are required for Box-Cox transformation
  if (feature <= 0).any():
    # Apply log1p (log(x+1)) for features that include non-positive values to ensure positivity
    return log1p(feature)
  else:
    # Apply Box-Cox transformation for strictly positive features
    transformed, _ = boxcox(feature)
    return transformed


# Load the data
ms_data = load_jsonl_file("shared_data/dataset_3_10_features.jsonl")

# Load the data of features into a DataFrame
df = pd.DataFrame(ms_data)

# The word_length feature is not present in the numerical_df, so we'll calculate it from the original df
df['average_word_length'] = df['word_length'].apply(lambda x: sum(x) / len(x) if x else 0)

# Convert 'average_word_length' to numeric type explicitly if not already
df['average_word_length'] = pd.to_numeric(df['average_word_length'], errors='coerce')


# Now select numerical columns, including 'average_word_length'
numerical_df = df.select_dtypes(include=['float64', 'int64', 'int']).copy()

# ------------------------------------------------------------
# Select numerical features for normality test excluding arrays
# numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

# Perform Shapiro-Wilk test for normality
normality_test_results = {feature: shapiro(df[feature]) for feature in numerical_df}

# Identify non-normally distributed features based on p-value (alpha=0.05)
non_normal_features = {feature: result for feature, result in normality_test_results.items() if result.pvalue < 0.05}

# Display Shapiro-Wilk test results
# normality_test_results, non_normal_features.keys()

# Apply transformations
transformed_features = {}
for feature in non_normal_features.keys():
    transformed_features[feature] = apply_transformation(df[feature])

# Convert transformed features back to DataFrame for easier handling
numerical_df = pd.DataFrame(transformed_features)

# Adding 'label' back to numerical_df for plotting purposes
numerical_df['label'] = df['label']

# General statistics for numerical features and class distribution
general_stats = numerical_df.describe()
class_distribution = df['label'].value_counts()

# Output general stats and class distribution
print("General Statistics for Numerical Features:")
# pprint(general_stats)
print("\nClass Distribution:")
pprint(class_distribution)

support_frame_counts = {}  # Counts of each frame in supportive sentences
oppose_frame_counts = {}  # Counts of each frame in opposing sentences

# pprint(frame_index)

for data_point in ms_data:
  # Directly use the 'semantic_frames' field from your datapoint
  semantic_frames = data_point['semantic_frames']

  # Determine the stance from the 'label' field
  stance = data_point['label']

  for index, presence in enumerate(semantic_frames):
    if presence:
      frame = index_frame[index]  # Assuming index_frame is the reverse mapping of frame_index
      if stance == 'support':
        support_frame_counts[frame] = support_frame_counts.get(frame, 0) + 1
      elif stance == 'oppose':
        oppose_frame_counts[frame] = oppose_frame_counts.get(frame, 0) + 1

# pprint(f"Supportive frame counts: {support_frame_counts}")
# pprint(f"Opposing frame counts: {oppose_frame_counts}")


# Select top N frames for visualization
N = 10
top_support_frames = sorted(support_frame_counts.items(), key=lambda x: x[1], reverse=True)[:N]
top_oppose_frames = sorted(oppose_frame_counts.items(), key=lambda x: x[1], reverse=True)[:N]

# Unzip the frame names and counts
support_frames, support_counts = zip(*top_support_frames)
oppose_frames, oppose_counts = zip(*top_oppose_frames)

# Create subplots for support and oppose frames
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Supportive Frames Bar Chart
axs[0].bar(support_frames, support_counts, color='green')
axs[0].set_title('Top Supportive Frames')
axs[0].tick_params(axis='x', rotation=45)

# Opposing Frames Bar Chart
axs[1].bar(oppose_frames, oppose_counts, color='red')
axs[1].set_title('Top Opposing Frames')
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Set the aesthetic style of the plots
sns.set_style("whitegrid")


# Function to plot histograms and density plots for each feature, separated by class
def plot_feature_distributions(_df, _feature):
  plt.figure(figsize=(14, 6))

  # Histogram
  plt.subplot(1, 2, 1)
  sns.histplot(data=_df, x=_feature, hue="label", multiple="stack", bins=15, kde=False)
  plt.title(f'Histogram of {_feature}')

  # Density Plot
  plt.subplot(1, 2, 2)
  sns.kdeplot(data=_df, x=_feature, hue="label", common_norm=False)
  plt.title(f'Density Plot of {_feature}')

  plt.tight_layout()
  plt.show()


# List of features to plot, ensuring only numeric features are considered
features_to_plot = numerical_df.columns.drop(['label'])

# Generating visualizations for each numeric feature
for feature in features_to_plot:
    plot_feature_distributions(numerical_df, feature)

# pprint(features)


"""# Initialize dictionaries to store the results
support_stats = {}
oppose_stats = {}

# Store the dataframe on a Google Sheet
sheet = spreadsheet_2.worksheet("features_extraction")
# Append DataFrame to worksheet
set_with_dataframe(sheet, df)

# Calculate the statistics for each feature
for feature in ms_features:
  support_stats[feature] = calculate_statistics([item for sublist in df_support[feature] for item in sublist])
  oppose_stats[feature] = calculate_statistics([item for sublist in df_oppose[feature] for item in sublist])

# Convert the dictionaries into dataframes and reset index
df_support_stats = pd.DataFrame(support_stats).transpose().reset_index()
df_oppose_stats = pd.DataFrame(oppose_stats).transpose().reset_index()

# Rename the new column containing the feature names to 'features'
df_support_stats = df_support_stats.rename(columns={"index": "features"})
df_oppose_stats = df_oppose_stats.rename(columns={"index": "features"})

print("Descriptive statistics for 'support'")
pprint(df_support_stats)
print()
print("Descriptive statistics for 'oppose'")
pprint(df_oppose_stats)"""

"""# Store the monologic dataframe on a Google Sheet
sheet = spreadsheet_2.worksheet("monologic_desc_stat")
# Append DataFrame to worksheet
set_with_dataframe(sheet, df_support_stats)

# Store the dialogic dataframe on a Google Sheet
sheet = spreadsheet_2.worksheet("dialogic_desc_stat")
# Append DataFrame to worksheet
set_with_dataframe(sheet, df_oppose_stats)"""
