from pprint import pprint

from scipy.stats import chi2_contingency
import numpy as np

from lib.semantic_frames import api_get_frames, encode_sentence_frames, frame_index
from lib.utils import load_jsonl_file

# Load the dataset
dataset = load_jsonl_file("shared_data/dataset_3_11_frame_counts.jsonl")

supportive_counts = dataset[0]
opposing_counts = dataset[1]


# Prepare to store results of Chi-square tests
chi2_results = []

# Conduct Chi-square tests for each frame
for i, frame_index in enumerate(frame_index):
  # Construct the contingency table for the current frame
  contingency_table = np.array([
    [supportive_counts[i], opposing_counts[i]],  # Counts of frame presence in each stance
    [(supportive_counts.sum() - supportive_counts[i]), (opposing_counts.sum() - opposing_counts[i])]
    # Counts of frame absence in each stance
  ])

  # Perform the Chi-square test
  chi2, p, dof, expected = chi2_contingency(contingency_table)
  # Store the results along with the frame index
  chi2_results.append((frame_index, chi2, p))

# Sort results by p-value to see the most significant frames first
chi2_results_sorted = sorted(chi2_results, key=lambda x: x[2])

# Display the top 5 most significant frames and their test results
print(chi2_results_sorted[:5])
