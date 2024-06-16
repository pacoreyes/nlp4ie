import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Initialize constants
SEED = 42

class_names = ["support", "oppose"]

# Load training dataset and extract features
df_train = pd.read_json("shared_data/dataset_2_2_1a_train_features.jsonl", lines=True)
# X_train = df_train.drop(columns=["id", "label", "lexical_d", "sentence_length"])
X_train = df_train.drop(columns=["id", "label", "text", "semantic_frames"])

# Extract labels for the training data
y_train = df_train['label']

# Prepare the test dataset
df_test = pd.read_json("shared_data/dataset_2_2_1a_test_features.jsonl", lines=True)
#X_test = df_test.drop(columns=["id", "label", "lexical_d", "sentence_length"])
X_test = df_test.drop(columns=["id", "label", "text", "semantic_frames"])

y_test = df_test['label']

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=SEED)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred, target_names=class_names, digits=3)
print(report)

# Extract the coefficients from the model
coefficients = model.coef_[0]

# Associate each coefficient with its corresponding feature name
feature_names = X_train.columns
features_importance = pd.DataFrame(coefficients, index=feature_names, columns=["Coefficient"])

# Calculate the absolute value of coefficients to determine influence
features_importance["Absolute_Coefficient"] = features_importance["Coefficient"].abs()

# Sort the features by their absolute coefficient in descending order to see the most influential features
features_importance_sorted = features_importance.sort_values(by="Absolute_Coefficient", ascending=False)

print(features_importance_sorted)

"""
              precision    recall  f1-score   support

     support      0.778     0.840     0.808        25
      oppose      0.826     0.760     0.792        25

    accuracy                          0.800        50
   macro avg      0.802     0.800     0.800        50
weighted avg      0.802     0.800     0.800        50

                       Coefficient  Absolute_Coefficient
neg_noun_polarity        -2.684334              2.684334
pos_verb_polarity         1.062900              1.062900
pos_noun_polarity         0.885325              0.885325
pos_adj_polarity          0.797054              0.797054
nominalization_use        0.708892              0.708892
neg_verb_polarity        -0.569064              0.569064
negation_use             -0.533128              0.533128
sentence_complexity      -0.389651              0.389651
modal_verbs_use          -0.355166              0.355166
word_length               0.354889              0.354889
discourse_markers_use    -0.347835              0.347835
lexical_density           0.341382              0.341382
neg_adj_polarity         -0.340006              0.340006
personal_pronouns        -0.074736              0.074736
passive_voice_use        -0.023091              0.023091
sentence_length           0.018428              0.018428

"""
