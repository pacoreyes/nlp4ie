"""
#######################################################################
Functions to plot visualizations
#######################################################################
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from spacy import displacy
import seaborn as sns
import pandas as pd
import numpy as np


def plot_confusion_matrix(y_true, y_pred,
                          class_names,
                          file_name,
                          title,
                          values_fontsize=14,
                          caption=None):
  """
  Plot a confusion matrix using the provided true and predicted labels.

  Parameters:
      y_true (array-like): True labels
      y_pred (array-like): Predicted labels
      class_names (list): Names of classes
      file_name (str): Path to save the figure
      title (str): Title of the figure
      values_fontsize (int): Font size of the values inside the matrix
      caption (str): Caption of the figure
  """
  # Compute the confusion matrix
  cm = confusion_matrix(y_true, y_pred)

  # Create the ConfusionMatrixDisplay object
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

  # Plot the confusion matrix
  disp.plot(cmap='Greens')

  # Adding title and labels
  plt.title(title, fontsize=16, y=1.05)
  plt.xlabel("True labels", fontsize=16)
  plt.ylabel("Predicted labels", fontsize=16)

  # Adjusting font size of the values inside the matrix
  for text in disp.text_:
    for t in text:
      t.set_fontsize(values_fontsize)

  # Adding a caption
  if caption:
    plt.figtext(0.5, -0.1, caption, ha="center", fontsize=12, wrap=True)

  # Save the plot to a file
  plt.savefig(f"images/{file_name}", format='png', bbox_inches='tight', dpi=300)
  plt.close()


def plot_dependency_visualization(sentence, output_path, nlp):
  """
  Generate a dependency parse visualization for a given sentence and save it to a file.
  :param sentence: A sentence to visualize
  :param output_path: The path to save the visualization, including the file name and extension
  :param nlp: The spaCy model to use for processing the sentence
  :return:
  """

  # Process the sentence using the model
  doc = nlp(sentence)

  dep_options = {
    "bg": False,
    "collapse_phrases": True,
    "collapse_punct": True,
    "distance": 100,
    "add_lemma": True,
    "fine_grained": True,
    "offset_x": 100
  }
  # Generate dependency parse using displacy
  svg = displacy.render(doc, style="dep", options=dep_options, jupyter=False)

  # Save the SVG to a file
  with open(output_path, "w", encoding="utf-8") as file:
    file.write(svg)

  # Find and print the ROOT of the sentence
  for token in doc:
    if token.dep_ == "ROOT":
      print(f"ROOT: {token.text} ({token.lemma_}) - POS: {token.pos_}")


def plot_feature_distributions(_df, _feature):
  """
Plots the distribution of a given feature in the dataset.
  :param _df: pandas DataFrame
  :param _feature: str, name of the feature to plot
  :return:
  """
  # Define your custom color palette
  custom_palette = {"support": "green", "oppose": "orange"}

  plt.figure(figsize=(14, 6))

  # Histogram
  plt.subplot(1, 2, 1)
  sns.histplot(data=_df, x=_feature, hue="label", multiple="stack", bins=15, kde=False, palette=custom_palette)
  plt.title(f'Histogram of {_feature}')

  # Density Plot
  plt.subplot(1, 2, 2)
  sns.kdeplot(data=_df, x=_feature, hue="label", common_norm=False, palette=custom_palette)
  plt.title(f'Density Plot of {_feature}')

  plt.tight_layout()
  # plt.show()
  plt.savefig(f"images/paper_c_rb_{_feature}_distribution.png", format='png', bbox_inches='tight', dpi=300)
  plt.close()


def plot_bar_chart(_df_1, _df_2, image_name, n=10):
  """
  Plots the top N frames for each stance class.
  :param _df_1: dict, counts of each frame in supportive
  :param _df_2: dict, counts of each frame in opposing
  :param image_name: str, name of the image to save
  :param n: int, number of top frames to plot
  """
  # Select top N frames for visualization
  top_support_frames = sorted(_df_1.items(), key=lambda x: x[1], reverse=True)[:n]
  top_oppose_frames = sorted(_df_2.items(), key=lambda x: x[1], reverse=True)[:n]

  # Unzip the frame names and counts
  support_frames, support_counts = zip(*top_support_frames)
  oppose_frames, oppose_counts = zip(*top_oppose_frames)

  # Create subplots for support and oppose frames
  fig, axs = plt.subplots(2, 1, figsize=(10, 8))

  # Supportive Frames Bar Chart
  axs[0].bar(support_frames, support_counts, color='Green')
  axs[0].set_title('Top Supportive Frames')
  axs[0].tick_params(axis='x', rotation=45)

  # Opposing Frames Bar Chart
  axs[1].bar(oppose_frames, oppose_counts, color='Orange')
  axs[1].set_title('Top Opposing Frames')
  axs[1].tick_params(axis='x', rotation=45)

  plt.tight_layout()
  # plt.show()
  plt.savefig(f'images/{image_name}.png', format='png', bbox_inches='tight', dpi=300)
  plt.close()


def plot_2x2_feature_boxplots(features_data, title, image_name):
  """
  Plots separate boxplots for the given features for both 'support' and 'oppose' stances using actual distribution data,
  arranged in a 2x2 layout on a single canvas.
  :param features_data: list of dicts, each dict contains the features data for a single instance
  :param title: str, title of the plot
  :param image_name: str, name of the image to save
  :return:
  """
  # Define the features to plot
  features = ['lexical_density', 'sentence_length', 'neg_verb_polarity', 'nominalization_use']

  # Initialize a dictionary to hold the data for plotting
  plot_data = {feature: {'support': [], 'oppose': []} for feature in features}

  # Extract data for each feature based on stance
  for item in features_data:
    label = item['label']
    for feature in features:
      if feature in item:  # Check if the feature exists in the data
        plot_data[feature][label].append(item[feature])

  # Create a figure and axes for the subplots in a 2x2 layout
  fig, axs = plt.subplots(2, 2, figsize=(12, 12))
  axs = axs.flatten()  # Flatten the 2x2 array to iterate easily

  # Plot each feature in its subplot
  for i, feature in enumerate(features):
    data_to_plot = [plot_data[feature]['support'], plot_data[feature]['oppose']]
    axs[i].boxplot(data_to_plot, patch_artist=True)
    axs[i].set_title(feature)
    axs[i].set_xticks([1, 2])
    axs[i].set_xticklabels(['Support', 'Oppose'])
    if i % 2 == 0:  # Only plots on the left need ylabels
      axs[i].set_ylabel('Values')

  # Super title for all subplots
  plt.suptitle(title)
  # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
  plt.savefig(f'images/{image_name}', format='png', bbox_inches='tight', dpi=300)
  # plt.show()


def plot_correlation_heatmap_double(corr_1, corr_2, title_1, title_2, feature_names, image_name):
  """
  Plots two heatmaps side by side for the given correlation matrices.
  :param corr_1: 2D array, correlation matrix 1
  :param corr_2: 2D array, correlation matrix 2
  :param title_1: str, title for the first heatmap
  :param title_2: str, title for the second heatmap
  :param feature_names: list, names of the features
  :param image_name: str, name of the image to save
  """
  # Create a figure with two subplots
  fig, ax = plt.subplots(ncols=2, figsize=(20, 8))

  # Heatmap for corr_1 with two decimal places in annotation
  sns.heatmap(corr_1, cmap='Greens', annot=True, fmt=".2f", ax=ax[0], xticklabels=feature_names,
              yticklabels=feature_names)
  ax[0].set_title(title_1)

  # Heatmap for corr_2 with two decimal places in annotation
  sns.heatmap(corr_2, cmap='Greens', annot=True, fmt=".2f", ax=ax[1], xticklabels=feature_names,
              yticklabels=feature_names)
  ax[1].set_title(title_2)

  plt.tight_layout()
  plt.savefig(image_name)
  plt.close()

  print(f"Heatmap saved successfully in {image_name}!")
