from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

features = {
  # "sentence_length": "Sentence Length",
  # "word_length": "Word Length",
  # "sentence_complexity": "Sentence Complexity",
  # "passive_voice_d": "Passive Voice Freq",
  "lexical_d": "Lexical Word Freq",
  "nominalization_d": "Nominalization Freq",
  "personal_pronoun_d": "Personal Pronoun Freq",
  "interjection_d": "Interjection Freq",
  "modal_verb_d": "Modal Verb Freq",
  "discourse_marker_d": "Discourse Marker Freq"
}

ms_features = list(features.keys())
feature_names = list(features.values())

# Initialize constants
SEED = 42

class_names = ["monologic", "dialogic"]

# Load training dataset and extract features
df_train = pd.read_json("shared_data/dataset_1_5_1a_train_features_transformed.jsonl", lines=True)
X_train = df_train.drop(columns=["id", "label",
                                 "sentence_length", "sentence_complexity",
                                 "passive_voice_d", "lexical_d", "word_length"])

# Extract labels for the training data
# Convert labels to numeric
y_train = pd.Categorical(df_train['label']).codes

# Prepare the test dataset
df_test = pd.read_json("shared_data/dataset_1_5_1a_test_features_transformed.jsonl", lines=True)
print(len(df_test))
X_test = df_test.drop(columns=["id", "label",
                               "sentence_length", "sentence_complexity",
                               "passive_voice_d", "lexical_d", "word_length"])

# Convert labels to numeric
y_test = pd.Categorical(df_test['label']).codes

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

print(model.summary())
html_data = model.summary().as_html()

# Save the HTML data to a file
html_file_path = "shared_data/paper_a_8_logistic_regression_results.html"
with open(html_file_path, "w") as file:
    file.write(html_data)

print(f"Logistic Regression scores saved in HTML format to '{html_file_path}'")

# Make predictions on the test set
y_pred_prob = model.predict(X_test_sm)
# Convert probabilities to binary output using 0.5 threshold
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

# Extract coefficients
coefficients = model.params.sort_values(ascending=True)

pprint(coefficients)

# Calculate ROC curve from y_test and y_pred_prob
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the AUC (Area under the ROC Curve)
roc_auc = auc(fpr, tpr)

print(f"AUC: {roc_auc}")

print("Confusion Matrix:")
print(cm)

# Plot the ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('images/paper_a_18_logistic_regression_roc_curve.png')

coefficients = model.params[1:]  # Exclude the intercept
features = X_train_sm.columns[1:]  # Adjust to match the coefficients' indexing

# Create a DataFrame for easier handling
# coef_df = pd.DataFrame(coefficients, index=features).reset_index()

# Create a DataFrame for the coefficients with human-readable feature names
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

coef_df.columns = ['Feature', 'Coefficient']

# Sort the DataFrame by the coefficient values for better visualization
coef_df = coef_df.sort_values(by='Coefficient', ascending=True)

# Assign a color based on the coefficient's sign
coef_df['Color'] = ['mediumseagreen' if x < 0 else 'orange' for x in coef_df['Coefficient']]


# Plotting coefficient values
plt.figure(figsize=(10, 9))
# plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=coef_df['Color'])
bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=coef_df['Color'])
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
# plt.title('Coefficient Plot')

for bar in bars:
    width = bar.get_width()
    label_x_pos = width if width > 0 else width - 0.5
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
             va='center', ha='right' if width < 0 else 'left')

# Optionally, set the margins
plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)

plt.grid(True)
plt.savefig('images/paper_a_19_logistic_regression_coefficients.png')


"""# Print the coefficients
print("Coefficients:")
for feature, coef in coefficients.items():
  print(f"{feature}: {coef}")"""

"""# Associate each coefficient with its corresponding feature name
features_importance = pd.DataFrame(coefficients, index=feature_names, columns=["coef"])
print(features_importance)


# Order the features by their coefficient values, ascending
features_importance = features_importance.sort_values(by="coef", ascending=True)"""

"""
Optimization terminated successfully.
         Current function value: 0.355268
         Iterations 8
                               Feature    Odds Ratio
const                            const  1.883640e+01
personal_pronoun_d  personal_pronoun_d  2.942276e+00
nominalization_d      nominalization_d  1.305066e+01
interjection_d          interjection_d  2.003801e-10
modal_verb_d              modal_verb_d  1.614627e-02
discourse_marker_d  discourse_marker_d  1.678960e+00
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                  870
Model:                          Logit   Df Residuals:                      864
Method:                           MLE   Df Model:                            5
Date:                Mon, 01 Apr 2024   Pseudo R-squ.:                  0.4874
Time:                        14:08:42   Log-Likelihood:                -309.08
converged:                       True   LL-Null:                       -602.96
Covariance Type:            nonrobust   LLR p-value:                8.988e-125
======================================================================================
                         coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------------
const                  2.9358      0.457      6.426      0.000       2.040       3.831
personal_pronoun_d     1.0792      1.095      0.986      0.324      -1.067       3.225
nominalization_d       2.5688      1.159      2.217      0.027       0.298       4.840
interjection_d       -22.3308      1.575    -14.176      0.000     -25.418     -19.243
modal_verb_d          -4.1261      0.932     -4.426      0.000      -5.953      -2.299
discourse_marker_d     0.5182      1.247      0.415      0.678      -1.927       2.963
======================================================================================
Logistic Regression scores saved in HTML format to 'shared_data/paper_a_8_logistic_regression_results.html'
              precision    recall  f1-score   support

   monologic      0.922     0.856     0.888       111
    dialogic      0.862     0.926     0.893       108

    accuracy                          0.890       219
   macro avg      0.892     0.891     0.890       219
weighted avg      0.893     0.890     0.890       219

interjection_d       -22.330805
modal_verb_d          -4.126066
discourse_marker_d     0.518174
personal_pronoun_d     1.079183
nominalization_d       2.568839
const                  2.935791
dtype: float64
AUC: 0.952952952952953
Confusion Matrix:
[[ 95  16]
 [  8 100]]

"""
