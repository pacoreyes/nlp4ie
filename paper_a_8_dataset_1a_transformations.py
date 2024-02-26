import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import boxcox

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

features = ["sentence_length", "word_length", "sentence_complexity", "personal_pronoun_d",
            "passive_voice_d", "nominalization_d", "lexical_d", "interjection_d", "modal_verb_d",
            "discourse_marker_d"]

for idx, dataset in enumerate([dataset_train, dataset_test]):

  # Convert the data into a DataFrame
  df = pd.json_normalize(dataset, max_level=0)

  # copy the dataframe
  df_flatten = df.copy()

  # Step 1: Aggregate list-based features using mean
  for feature in features:
      df_flatten[feature + "_mean"] = df_flatten[feature].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)

  # Step 2: Check for skewness and decide on transformations
  skewed_features = df_flatten[[feature + "_mean" for feature in features]].apply(lambda x: stats.skew(x.dropna()))
  skew_limit = 0.75  # a common threshold to consider a distribution skewed
  skewed_cols = skewed_features[skewed_features.abs() > skew_limit].index.tolist()

  # Step 3: Apply Box-Cox transformation to reduce skewness, if necessary
  # Note: Box-Cox requires all data to be positive; if any columns contain negative data, use another method
  for feature in skewed_cols:
      # Add a small positive value to avoid zero values
      df_flatten[feature], _ = boxcox(df_flatten[feature].values + 1e-5)

  # Step 4: Normalize the feature values (using Z-score standardization as an example)
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  features_to_scale = [feature + "_mean" for feature in features]
  df_flatten[features_to_scale] = scaler.fit_transform(df_flatten[features_to_scale])

  # Drop original list-based feature columns
  df_flatten.drop(columns=features, inplace=True)

  # Replace label with 0/1 using LABEL_MAP
  df_flatten['label'] = df_flatten['label'].map(LABEL_MAP)

  # Save the transformed dataset
  save_jsonl_file(df_flatten.to_dict(orient='records'), output_paths[idx])
