"""
#######################################################################
Functions to plot visualizations
#######################################################################
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
  plt.savefig(f"shared_images/{file_name}", format='png', bbox_inches='tight', dpi=300)
