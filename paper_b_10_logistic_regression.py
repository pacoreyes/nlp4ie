import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from lib.visualizations import plot_correlation_heatmap_double

features = {
  "positive_affect": "Positive affect",
  "negative_affect": "Negative affect",
  "epistemic_certainty": "Certainty",
  "epistemic_doubt": "Doubt",
  "emphatic": "Emphatics",
  "hedge": "Hedge",
  "polarity_pro": "Pro polarity",
  "polarity_con": "Con polarity",
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
                                 Feature  Odds Ratio
const                              const    0.730645
positive_affect          positive_affect    5.914431
negative_affect          negative_affect    0.138092
epistemic_certainty  epistemic_certainty    1.148120
epistemic_doubt          epistemic_doubt    1.362438
emphatic                        emphatic    1.032792
hedge                              hedge    0.725781
pro                                  pro    6.468472
con                                  con    0.065683
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                 1080
Model:                          Logit   Df Residuals:                     1071
Method:                           MLE   Df Model:                            8
Date:                Mon, 17 Jun 2024   Pseudo R-squ.:                  0.4465
Time:                        14:44:50   Log-Likelihood:                -414.34
converged:                       True   LL-Null:                       -748.60
Covariance Type:            nonrobust   LLR p-value:                4.288e-139
=======================================================================================
                          coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------------
const                  -0.3138      0.274     -1.144      0.252      -0.851       0.224
positive_affect         1.7774      0.184      9.641      0.000       1.416       2.139
negative_affect        -1.9798      0.239     -8.284      0.000      -2.448      -1.511
epistemic_certainty     0.1381      0.177      0.780      0.435      -0.209       0.485
epistemic_doubt         0.3093      0.222      1.393      0.164      -0.126       0.745
emphatic                0.0323      0.232      0.139      0.889      -0.422       0.487
hedge                  -0.3205      0.193     -1.658      0.097      -0.699       0.058
pro                     1.8669      0.191      9.772      0.000       1.493       2.241
con                    -2.7229      0.203    -13.417      0.000      -3.121      -2.325
=======================================================================================
Logistic Regression scores saved in HTML format to 'shared_data/paper_b_8_logistic_regression_results.html'
Optimization terminated successfully.
         Current function value: 0.384910
         Iterations 7
                         Feature  Odds Ratio
const                      const    0.873687
positive_affect  positive_affect    5.816746
negative_affect  negative_affect    0.138268
hedge                      hedge    0.749200
pro                          pro    6.126747
con                          con    0.064484
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                 1080
Model:                          Logit   Df Residuals:                     1074
Method:                           MLE   Df Model:                            5
Date:                Mon, 17 Jun 2024   Pseudo R-squ.:                  0.4447
Time:                        14:44:50   Log-Likelihood:                -415.70
converged:                       True   LL-Null:                       -748.60
Covariance Type:            nonrobust   LLR p-value:                1.221e-141
===================================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------
const              -0.1350      0.158     -0.853      0.394      -0.445       0.175
positive_affect     1.7607      0.183      9.600      0.000       1.401       2.120
negative_affect    -1.9786      0.237     -8.340      0.000      -2.444      -1.514
hedge              -0.2887      0.192     -1.504      0.133      -0.665       0.088
pro                 1.8127      0.186      9.735      0.000       1.448       2.178
con                -2.7413      0.202    -13.577      0.000      -3.137      -2.346
===================================================================================
Accuracy: 0.790
Precision: 0.822
Recall: 0.740
F1 Score: 0.779
AUC-ROC: 0.790
Confusion Matrix:
[[84 16]
 [26 74]]
Class-wise Performance Metrics:
     Class  Precision  Recall  F1 Score
0  support      0.764   0.840     0.800
1   oppose      0.822   0.740     0.779
"""
