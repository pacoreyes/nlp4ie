import pandas as pd
# from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import chi2_contingency
import numpy as np

from lib.semantic_frames import decode_frame_vector

# Load the train and test datasets
train_file_path = 'shared_data/dataset_2_3_1a_train_features_aggregated.jsonl'
test_file_path = 'shared_data/dataset_2_3_1a_test_features_aggregated.jsonl'

train_data = pd.read_json(train_file_path, lines=True)
test_data = pd.read_json(test_file_path, lines=True)

# Combine the datasets
combined_data = pd.concat([train_data, test_data], ignore_index=True)

semantic_frames = combined_data['semantic_frames'].apply(lambda x: np.array(x))

"""# Split the semantic frames into 2 classes
support_frames = semantic_frames[combined_data['label'] == 'support']
oppose_frames = semantic_frames[combined_data['label'] == 'oppose']"""

""" Chi2 for the Support Class """

# Get the number of semantic frames
num_frames = len(semantic_frames.iloc[0])

# Prepare the data for chi-square tests
chi2_results = []

for i in range(num_frames):
  contingency_table = pd.crosstab(combined_data['label'], semantic_frames.apply(lambda x: x[i]))
  chi2, p, _, _ = chi2_contingency(contingency_table)
  chi2_results.append((i, chi2, p))

# Convert results to DataFrame
chi2_df = pd.DataFrame(chi2_results, columns=['frame_index', 'chi2', 'p_value'])

# Select the top 50 most significant frames based on chi2 values
top_frames = chi2_df.nlargest(30, 'chi2')

# Decode the frame indices to frame names
top_frames['frame_name'] = top_frames['frame_index'].apply(
  lambda x: decode_frame_vector([1 if i == x else 0 for i in range(num_frames)]))

print(top_frames)

# Convert to dictionary
df_dict = top_frames.to_dict(orient='records')

# remove the first and last characters from each frame name
for row in df_dict:
  row['frame_name'] = row['frame_name'][0]

# Create HTML table
html_table = """
<table border="1">
  <tr>
    <th>Order</th>
    <th>Frame Index</th>
    <th>Semantic Frame</th>
    <th>chi2</th>
    <th>p_value</th>
  </tr>
"""
for idx, row in enumerate(df_dict, start=1):
  html_table += f"""
  <tr>
    <td>{idx}</td>
    <td>{row['frame_index']}</td>
    <td>{row['frame_name']}</td>
    <td>{row['chi2']:.2f}</td>
    <td>{row['p_value']:.3f}</td>
  </tr>
  """
html_table += "</table>"

# Print HTML table
print(html_table)
