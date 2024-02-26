"""
This script calculates the descriptive statistics for the features extracted from the measurements of linguistic
features extracted from the sentences.
"""
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import boxcox
from numpy import log1p
# from pprint import pprint

from lib.utils import load_jsonl_file
from lib.visualizations import plot_feature_distributions, plot_bar_chart, plot_2x2_feature_boxplots
from lib.semantic_frames import index_frame
from pprint import pprint


# Set the aesthetic style of the plots
# sns.set_style("whitegrid")


def apply_transformation(feature):
  """
  Applies a transformation to a feature to attempt to normalize its distribution.
  The transformation to apply is chosen based on the nature of the data.

  :param feature: The feature to transform
  :return: The transformed feature
  """
  # Positive values are required for Box-Cox transformation
  if (feature <= 0).any():
    # Apply log1p (log(x+1)) for features that include non-positive values to ensure positivity
    return log1p(feature)
  else:
    # Apply Box-Cox transformation for strictly positive features
    transformed, _ = boxcox(feature)
    return transformed


def calculate_class_stats(_df, label_column='label'):
  """
  Calculate and return statistics (Mean, Median, SD, IQR) for each class in the DataFrame.

  :param _df: pandas DataFrame containing numerical features and a class label column.
  :param label_column: str, name of the column containing class labels.
  :return: dict, containing statistics (mean, median, SD, IQR) for each class.
  """
  class_labels = _df[label_column].unique()
  stats_dict = {}
  for label in class_labels:
    class_df = _df[_df[label_column] == label].drop(label_column, axis=1)
    stats = {
      'mean': class_df.mean(),
      'median': class_df.median(),
      'sd': class_df.std(),
      'iqr': class_df.quantile(0.75) - class_df.quantile(0.25),
    }
    stats_dict[label] = {stat: value.round(3).to_dict() for stat, value in stats.items()}

    # Optionally, print the statistics for each class
    """print(f"\nGeneral Statistics for '{label}' Class:")
    for stat, value in stats_dict[label].items():
      print(f"{stat.upper()}:")
      print(value)"""
  return stats_dict


def count_frames(data, frame_mapping):
  """
  Count occurrences of frames based on the label.

  :param data: list, containing the data points
  :param frame_mapping: dict, mapping of frame indices to frame names
  :return: dict, containing counts of frames for each stance
  """
  _frame_counts = {'support': {}, 'oppose': {}}
  for _data_point in data:
    _semantic_frames = _data_point['semantic_frames']
    _stance = _data_point['label']
    for _index, _presence in enumerate(_semantic_frames):
      if _presence:
        _frame = frame_mapping[_index]
        _frame_counts[_stance][_frame] = _frame_counts[_stance].get(_frame, 0) + 1
  return _frame_counts


# Load the data
ms_data = load_jsonl_file("shared_data/dataset_2_2_1a_train_features.jsonl")

# Load the data of features into a DataFrame
df = pd.DataFrame(ms_data)

""" ##############################
# Feature Engineering: Handle values and convert to numeric data, applying normality tests and transformations
############################## """

# Convert the list of number "word_length" to the average word length
# df['word_length'] = df['word_length'].apply(lambda x: sum(x) / len(x) if x else 0)

"""# Convert 'word_length' to numeric type explicitly if not already
df['average_word_length'] = pd.to_numeric(df['average_word_length'], errors='coerce')"""

# Select numerical features, excluding other data types
numerical_df = df.select_dtypes(include=['float64', 'int64', 'int']).copy()

# Perform Shapiro-Wilk test for normality
normality_test_results = {feature: shapiro(df[feature]) for feature in numerical_df}

# Identify non-normally distributed features based on p-value (alpha=0.05)
# non_normal_features = {feature: result for feature, result in normality_test_results.items() if result[1] < 0.05}
non_normal_features = {feature: (stat, pval) for feature, (stat, pval) in normality_test_results.items() if pval < 0.05}

# Print normality test results
# print("\nNormality test results:")
# pprint(f"{normality_test_results}")
# print("\nTransforming non-normal features...")

# Apply transformations
transformed_features = {}
for feature in non_normal_features.keys():
    transformed_features[feature] = apply_transformation(df[feature])

# print("\nFeatures after transformation:")
# pprint(transformed_features)

# Convert transformed features back to DataFrame for easier handling
numerical_df = pd.DataFrame(transformed_features)

""" ##############################
# Visualize the transformed features
############################## """

# Adding 'label' back to numerical_df for plotting purposes
numerical_df['label'] = df['label']
# Calculate general statistics for each class
class_stats_dict = calculate_class_stats(numerical_df)

pprint(class_stats_dict)

# Call the refactored function with the loaded data, a title, and an image name
plot_2x2_feature_boxplots(ms_data,
                          "Boxplots of 4 main features by Stance",
                          "paper_b_rb_boxplot_features.png")


# Counting frames for each stance
frame_counts = count_frames(ms_data, index_frame)

"""# Splitting the DataFrame into two based on the labels "support" and "oppose"
support_df = numerical_df[numerical_df['label'] == 'support']
oppose_df = numerical_df[numerical_df['label'] == 'oppose']

# Calculating general statistics for each class
support_stats = support_df.describe()
oppose_stats = oppose_df.describe()

# Converting general statistics to dictionaries for easier handling
support_stats_dict = support_stats.to_dict()
oppose_stats_dict = oppose_stats.to_dict()

# Outputting the general statistics for each class
print("\nGeneral Statistics for 'Support' Class:")
print(support_stats_dict)

print("\nGeneral Statistics for 'Oppose' Class:")
print(oppose_stats_dict)"""

"""# General statistics for numerical features and class distribution
general_stats = numerical_df.describe()
# Convert general statistics to a dictionary for easier handling
general_stats = general_stats.to_dict()

class_distribution = df['label'].value_counts()

# Output general stats and class distribution
# print("\nGeneral Statistics for Numerical Features:")
pprint(general_stats)
print()
print("\nClass Distribution:")
pprint(class_distribution)
print()"""

""" ##############################
# Convert one-hot encoded semantic frames to frames information , and then to counts
############################## """

support_frame_counts = {}  # Counts of each frame in supportive sentences
oppose_frame_counts = {}  # Counts of each frame in opposing sentences

for data_point in ms_data:
  # Get 'semantic_frames' attribute from datapoint
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

print()
print("Supportive frame counts:")
# pprint(support_frame_counts)
print("Opposing frame counts:")
# pprint(oppose_frame_counts)
print()

""" ##############################
# Visualize the top N frames per stance class
############################## """

# Plot the top N frames for each stance class
# plot_bar_chart(support_frame_counts, oppose_frame_counts, "paper_c_rb_frames_bar_chart.png")

plot_bar_chart(frame_counts['support'], frame_counts['oppose'], "paper_b_rb_frames_bar_chart.png")

# List of features to plot, ensuring only numeric features are considered
"""features_to_plot = numerical_df.columns.drop(['label'])

# Generating visualizations for each numeric feature
for feature in features_to_plot:
    plot_feature_distributions(numerical_df, feature)"""

features_to_plot = numerical_df.columns.drop('label')
for feature in features_to_plot:
  print(f"Feature: {feature}")
  plot_feature_distributions(numerical_df, feature)
