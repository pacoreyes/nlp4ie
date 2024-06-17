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
  "emphatic": "Emphatics",
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
         Iterations 7
                                 Feature  Odds Ratio
const                              const    1.164406
positive_affect          positive_affect    6.212204
negative_affect          negative_affect    0.126551
epistemic_certainty  epistemic_certainty    1.072766
epistemic_doubt          epistemic_doubt    1.125921
emphatic                        emphatic    1.011570
hedge                              hedge    0.724159
pro                                  pro    6.525052
con                                  con    0.036875
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                 1080
Model:                          Logit   Df Residuals:                     1071
Method:                           MLE   Df Model:                            8
Date:                Mon, 17 Jun 2024   Pseudo R-squ.:                  0.5158
Time:                        06:57:38   Log-Likelihood:                -362.46
converged:                       True   LL-Null:                       -748.60
Covariance Type:            nonrobust   LLR p-value:                1.938e-161
=======================================================================================
                          coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------------
const                   0.1522      0.303      0.503      0.615      -0.441       0.746
positive_affect         1.8265      0.201      9.094      0.000       1.433       2.220
negative_affect        -2.0671      0.255     -8.097      0.000      -2.567      -1.567
epistemic_certainty     0.0702      0.192      0.366      0.714      -0.306       0.446
epistemic_doubt         0.1186      0.238      0.498      0.619      -0.348       0.586
emphatic                0.0115      0.251      0.046      0.964      -0.481       0.504
hedge                  -0.3227      0.210     -1.537      0.124      -0.734       0.089
pro                     1.8756      0.207      9.057      0.000       1.470       2.282
con                    -3.3002      0.214    -15.396      0.000      -3.720      -2.880
=======================================================================================
Logistic Regression scores saved in HTML format to 'shared_data/paper_b_8_logistic_regression_results.html'
Optimization terminated successfully.
         Current function value: 0.335674
         Iterations 7
                         Feature  Odds Ratio
const                      const    1.220455
positive_affect  positive_affect    6.179413
negative_affect  negative_affect    0.125646
epistemic_doubt  epistemic_doubt    1.128956
hedge                      hedge    0.727361
pro                          pro    6.495653
con                          con    0.036775
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                 1080
Model:                          Logit   Df Residuals:                     1073
Method:                           MLE   Df Model:                            6
Date:                Mon, 17 Jun 2024   Pseudo R-squ.:                  0.5157
Time:                        06:57:38   Log-Likelihood:                -362.53
converged:                       True   LL-Null:                       -748.60
Covariance Type:            nonrobust   LLR p-value:                1.608e-163
===================================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------
const               0.1992      0.185      1.076      0.282      -0.164       0.562
positive_affect     1.8212      0.200      9.100      0.000       1.429       2.213
negative_affect    -2.0743      0.254     -8.158      0.000      -2.573      -1.576
epistemic_doubt     0.1213      0.238      0.509      0.611      -0.346       0.588
hedge              -0.3183      0.210     -1.519      0.129      -0.729       0.092
pro                 1.8711      0.206      9.099      0.000       1.468       2.274
con                -3.3030      0.214    -15.428      0.000      -3.723      -2.883
===================================================================================
Accuracy: 0.875
Precision: 0.879
Recall: 0.870
F1 Score: 0.874
AUC-ROC: 0.875
Confusion Matrix:
[[88 12]
 [13 87]]
Class-wise Performance Metrics:
     Class  Precision  Recall  F1 Score
0  support      0.871   0.880     0.876
1   oppose      0.879   0.870     0.874
"""
