import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from lib.visualizations import plot_correlation_heatmap_double

features = {
  "positive_affect": "Positive Affect",
  "negative_affect": "Negative Affect",
  "epistemic_certainty": "Epistemic Certainty",
  "epistemic_doubt": "Epistemic Doubt",
  "emphatic": "Emphatics",
  "hedge": "Hedge",
  "polarity_pro": "Polarity Pro",
  "polarity_con": "Polarity Con",
}

# ms_features = list(features.keys())
feature_names = list(features.values())

class_names = ["support", "oppose"]

# Load the train and test datasets
train_file_path = 'shared_data/dataset_2_3_1a_train_features_aggregated.jsonl'
test_file_path = 'shared_data/dataset_2_3_1a_test_features_aggregated.jsonl'

train_df = pd.read_json(train_file_path, lines=True)
test_df = pd.read_json(test_file_path, lines=True)

# Dropping irrelevant columns and 'semantic_frames'
train_df = train_df.drop(columns=['id', 'metadata', 'semantic_frames', 'text'])
test_df = test_df.drop(columns=['text'])

# Splitting the data into support and oppose classes
support_df = train_df[train_df['label'] == 'support'].drop(columns=['label'])
oppose_df = train_df[train_df['label'] == 'oppose'].drop(columns=['label'])

# Calculate correlation matrices
support_corr = support_df.corr(method='pearson')
oppose_corr = oppose_df.corr(method='pearson')

# Round the correlation matrices to three decimal places
correlation_matrix_support = support_corr.round(3)
correlation_matrix_oppose = oppose_corr.round(3)

# Plot the correlation matrices
plot_correlation_heatmap_double(
  correlation_matrix_support,
  correlation_matrix_oppose,
  'Correlation Matrix of Linguistic features for the Support Class',
  'Correlation Matrix of Linguistic features for the Oppose Class',
  feature_names,
  "images/paper_b_7_correlation_matrix_heatmap.png")

""" ############################## """


def prune_features(corr_matrix, threshold=0.8):
  # Create a set to hold the features to drop
  to_drop = set()

  # Iterate over the correlation matrix
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if abs(corr_matrix.iloc[i, j]) > threshold:
        colname = corr_matrix.columns[i]
        to_drop.add(colname)

  return list(to_drop)


# Prune features for both classes
support_features_to_drop = prune_features(support_corr)
oppose_features_to_drop = prune_features(oppose_corr)

# Combine the lists and get unique features to drop
features_to_drop = list(set(support_features_to_drop + oppose_features_to_drop))

# Drop these features from the filtered dataset
X_train = train_df.drop(columns=features_to_drop + ['label'])

# Convert labels to numeric
y_train = pd.Categorical(train_df['label']).codes

X_test = test_df.drop(columns=['id', 'metadata', 'semantic_frames'])

# Convert labels to numeric
y_test = pd.Categorical(test_df['label']).codes

# Add an intercept term to the X_train and X_test for statsmodels
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit the logistic regression model
model = sm.Logit(y_train, X_train_sm).fit()

odds_ratios = np.exp(model.params)

# Optionally, you can create a DataFrame to nicely format the results
odds_ratio_df = pd.DataFrame({
    "Feature": X_train_sm.columns,
    "Odds Ratio": odds_ratios
})

print(odds_ratio_df)

# Display the summary of the model
print(model.summary())

html_data = model.summary().as_html()

# Save the HTML data to a file
html_file_path = "shared_data/paper_b_8_logistic_regression_results.html"
with open(html_file_path, "w") as file:
    file.write(html_data)

print(f"Logistic Regression scores saved in HTML format to '{html_file_path}'")

features = {
  "positive_affect": "Positive Affect",
  "negative_affect": "Negative Affect",
  "epistemic_doubt": "Epistemic Doubt",
  "hedge": "Hedge",
  "polarity_pro": "Polarity Pro",
  "polarity_con": "Polarity Con",
}

ms_features = list(features.keys())
feature_names = list(features.values())

# List of features to drop based on correlation analysis
features_to_drop = ['epistemic_certainty', 'emphatic']

X_train = X_train.drop(columns=features_to_drop)
X_test = X_test.drop(columns=features_to_drop)
X_test = X_test.drop(columns=['label'])

# Add an intercept term to the X_train and X_test for statsmodels
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit the logistic regression model
model = sm.Logit(y_train, X_train_sm).fit()

odds_ratios = np.exp(model.params)

# Optionally, you can create a DataFrame to nicely format the results
odds_ratio_df = pd.DataFrame({
    "Feature": X_train_sm.columns,
    "Odds Ratio": odds_ratios
})

print(odds_ratio_df)

# Display the summary of the model
print(model.summary())

# Predicting the test dataset
y_pred_prob = model.predict(X_test_sm)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate various performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.3f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Print class-wise performance metrics
class_wise_metrics = pd.DataFrame({
    "Class": class_names,
    "Precision": precision_score(y_test, y_pred, average=None),
    "Recall": recall_score(y_test, y_pred, average=None),
    "F1 Score": f1_score(y_test, y_pred, average=None)
})

# Format output to three decimal places
pd.options.display.float_format = '{:,.3f}'.format

print("Class-wise Performance Metrics:")
print(class_wise_metrics)

# Reset display format if needed elsewhere
pd.reset_option('^display.', silent=True)

"""
                         Feature  Odds Ratio
const                      const    0.768993
positive_affect  positive_affect    5.857668
negative_affect  negative_affect    0.132598
epistemic_doubt  epistemic_doubt    1.416473
hedge                      hedge    0.722534
pro                          pro    6.320866
con                          con    0.067316
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                 1080
Model:                          Logit   Df Residuals:                     1073
Method:                           MLE   Df Model:                            6
Date:                Sat, 08 Jun 2024   Pseudo R-squ.:                  0.4386
Time:                        19:44:04   Log-Likelihood:                -420.25
converged:                       True   LL-Null:                       -748.60
Covariance Type:            nonrobust   LLR p-value:                1.366e-138
===================================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------
const              -0.2627      0.165     -1.587      0.112      -0.587       0.062
positive_affect     1.7678      0.182      9.687      0.000       1.410       2.125
negative_affect    -2.0204      0.236     -8.565      0.000      -2.483      -1.558
epistemic_doubt     0.3482      0.221      1.576      0.115      -0.085       0.781
hedge              -0.3250      0.192     -1.696      0.090      -0.700       0.050
pro                 1.8439      0.188      9.801      0.000       1.475       2.213
con                -2.6984      0.206    -13.116      0.000      -3.102      -2.295
===================================================================================
Accuracy: 0.820
Precision: 0.840
Recall: 0.790
F1 Score: 0.814
AUC-ROC: 0.820
Confusion Matrix:
[[85 15]
 [21 79]]
Class-wise Performance Metrics:
     Class  Precision  Recall  F1 Score
0  support      0.802   0.850     0.825
1   oppose      0.840   0.790     0.814
"""
