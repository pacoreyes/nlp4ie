import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

from lib.utils import load_jsonl_file
from lib.visualizations import plot_confusion_matrix

features = {
  "sentence_length": "Sentence Length",
  "word_length": "Word Length",
  "sentence_complexity": "Sentence Complexity",
  "passive_voice_d": "Passive Voice Freq",
  "lexical_d": "Lexical Word Freq",
  "nominalization_d": "Nominalization Freq",
  "personal_pronoun_d": "Personal Pronoun Freq",
  "interjection_d": "Interjection Freq",
  "modal_verb_d": "Modal Verb Freq",
  "discourse_marker_d": "Discourse Marker Freq"
}

# Make list of keys and list of values
ms_features = list(features.keys())
feature_names = list(features.values())

dataset_train = load_jsonl_file("shared_data/dataset_1_5_1a_train_features_transformed.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_1_5_1a_test_features_transformed.jsonl")

class_names = ["speech", "interview"]

# Convert to DataFrame
df_train = pd.DataFrame(dataset_train)
df_test = pd.DataFrame(dataset_test)

# Define features and target
X_train = df_train.drop(["id", "label"], axis=1)
y_train = df_train["label"]
X_test = df_test.drop(["id", "label"], axis=1)
y_test = df_test["label"]

# Perform grid search for hyperparameter tuning
print("Performing grid search for hyperparameter tuning...")
param_grid = {'C': [0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(SVC(
  kernel='linear',
  random_state=42,
  probability=True,
  class_weight='balanced'),
  param_grid,
  cv=StratifiedKFold(n_splits=5),
  scoring='accuracy'
)
grid_search.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...\n")
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='monologic')
recall = recall_score(y_test, y_pred, pos_label='monologic')
f1 = f1_score(y_test, y_pred, pos_label='monologic')
roc_auc = roc_auc_score(y_test, grid_search.decision_function(X_test))  # For ROC AUC, use decision scores
mcc = matthews_corrcoef(y_test, y_pred)


# Get feature weights
print("Calculating feature weights...")
feature_weights = grid_search.best_estimator_.coef_[0]
weights_df = pd.DataFrame(
  {
    'feature': feature_names,
    'weight': feature_weights
  }
).sort_values('weight', ascending=False)

# Order the features by their absolute weight
weights_df = weights_df.sort_values('weight', ascending=False)


# Plot feature weights
print("Plotting feature weights...")
plt.figure(figsize=(12, 8))
sns.barplot(x='weight', y='feature', data=weights_df, palette='Greens', hue='feature')
plt.title('Feature Weights', fontsize=16)
plt.xlabel('Weight', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig('images/paper_a_8_ml_model_feature_weights_.png')


weights_df['weight'] = weights_df['weight'].abs()
# Order the features by their absolute weight
weights_df = weights_df.sort_values('weight', ascending=False)
# Print feature weights
print(weights_df)

# Create confusion matrix
print("Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred, labels=["monologic", "dialogic"])
cm_df = pd.DataFrame(cm, index=["monologic", "dialogic"], columns=["monologic", "dialogic"])

plot_confusion_matrix(y_test, y_pred,
                      class_names,
                      "paper_a_7_ml_confusion_matrix_.png",
                      "Confusion Matrix for SVM Model",
                      values_fontsize=22)

# Print metrics
print("\nSVM Model Evaluation\n")
print(f"- Accuracy: {accuracy:.4f}")
print(f"- Precision: {precision:.4f}")
print(f"- Recall: {recall:.4f}")
print(f"- F1 Score: {f1:.4f}")
print(f"- AUC-ROC: {roc_auc:.4f}")
print(f"- Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"- Confusion Matrix:\n{cm_df}")

# Cross-validation (ensure you're using the correct variables)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=cv, scoring='accuracy')
print(f"\nCV Accuracy: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")

# Save the trained model
joblib.dump(grid_search.best_estimator_, 'models/1/svm_model.pkl')
