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

"""
