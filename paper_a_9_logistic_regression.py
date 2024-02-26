import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Initialize constants
SEED = 42

class_names = ["monologic", "dialogic"]

# Load training dataset and extract features
df_train = pd.read_json("shared_data/dataset_1_5_1a_train_features_transformed_.jsonl", lines=True)
# X_train = df_train.drop(columns=["id", "label", "lexical_d", "sentence_length"])
X_train = df_train.drop(columns=["id", "label"])

# Extract labels for the training data
y_train = df_train['label']

# Prepare the test dataset
df_test = pd.read_json("shared_data/dataset_1_5_1a_test_features_transformed_.jsonl", lines=True)
#X_test = df_test.drop(columns=["id", "label", "lexical_d", "sentence_length"])
X_test = df_test.drop(columns=["id", "label"])

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

   monologic      0.905     0.880     0.892       108
    dialogic      0.886     0.910     0.898       111

    accuracy                          0.895       219
   macro avg      0.895     0.895     0.895       219
weighted avg      0.895     0.895     0.895       219

                          Coefficient  Absolute_Coefficient
interjection_d_mean          2.660952              2.660952
lexical_d_mean              -2.272831              2.272831
sentence_complexity_mean     1.822530              1.822530
nominalization_d_mean        1.111391              1.111391
sentence_length_mean        -0.944209              0.944209
personal_pronoun_d_mean     -0.744464              0.744464
modal_verb_d_mean            0.647865              0.647865
discourse_marker_d_mean      0.495911              0.495911
passive_voice_d_mean         0.331185              0.331185
word_length_mean            -0.296988              0.296988
"""
