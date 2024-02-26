import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Initialize constants
SEED = 42

class_names = ["monologic", "dialogic"]

# Load training dataset and extract features
df_train = pd.read_json("shared_data/dataset_1_5_1a_train_features_transformed.jsonl", lines=True)
X_train = df_train.drop(columns=["id", "label", "lexical_d", "sentence_length"])

# Extract labels for the training data
y_train = df_train['label']

# Prepare the test dataset
df_test = pd.read_json("shared_data/dataset_1_5_1a_test_features_transformed.jsonl", lines=True)
X_test = df_test.drop(columns=["id", "label", "lexical_d", "sentence_length"])
y_test = df_test['label']

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=SEED)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred, target_names=class_names)
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

   monologic       0.78      0.91      0.84       111
    dialogic       0.89      0.73      0.80       108

    accuracy                           0.82       219
   macro avg       0.83      0.82      0.82       219
weighted avg       0.83      0.82      0.82       219

                     Coefficient  Absolute_Coefficient
interjection_d         -7.239860              7.239860
nominalization_d        4.703147              4.703147
sentence_complexity    -2.106434              2.106434
personal_pronoun_d      1.003056              1.003056
modal_verb_d           -0.266802              0.266802
passive_voice_d        -0.178065              0.178065
discourse_marker_d     -0.086783              0.086783
word_length             0.002430              0.002430

"""
