import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

from lib.utils import load_jsonl_file
from lib.visualizations import plot_confusion_matrix

"""SVM Support Vector Machine"""

ms_data = load_jsonl_file("shared_data/paper_a_2_feature_extraction.jsonl")

class_names = ["monologic", "dialogic"]

# Convert to DataFrame
df = pd.DataFrame(ms_data)

df_monologic = df[df['discourse_type'] == 0]
df_dialogic = df[df['discourse_type'] == 1]

print(f"Loaded {len(df_monologic)} monologic texts")
print(f"Loaded {len(df_dialogic)} dialogic texts")

# Convert string representations of lists into actual lists of numbers
print("Initial conversion of data types...")
for column in df.columns:
  if isinstance(df[column][0], str) and ',' in df[column][0]:
    df[column] = df[column].apply(lambda x: [float(num) for num in x.split(",")])

# Calculate summary statistics
print("Performing Exploratory Data Analysis...")
for column in df.columns:
  if isinstance(df[column][0], list):
    df[column + "_mean"] = df[column].apply(np.mean)
    df[column + "_max"] = df[column].apply(np.max)
    df[column + "_min"] = df[column].apply(np.min)
    df[column + "_std"] = df[column].apply(np.std)
    df.drop(column, axis=1, inplace=True)

# Define features and target
print("Defining features and target...")
X = df.drop(["id", "discourse_type"], axis=1)
y = df["discourse_type"]

# Split data into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Apply feature scaling
print("Applying feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM with a linear kernel
print("Training SVM with a linear kernel...")
svm = SVC(kernel='linear', random_state=42, C=0.1, probability=True, class_weight='balanced')
svm.fit(X_train_scaled, y_train)

# Perform grid search for hyperparameter tuning
print("Performing grid search for hyperparameter tuning...")
param_grid = {'C': [0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(SVC(kernel='linear',
                               random_state=42,
                               probability=True),
                           param_grid, cv=StratifiedKFold(n_splits=5),
                           scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Evaluate the model
print("Evaluating the model...\n")
y_pred = grid_search.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# Get feature weights
print("Calculating feature weights...")
feature_weights = grid_search.best_estimator_.coef_[0]
weights_df = pd.DataFrame({'feature': X.columns, 'weight': feature_weights})
# Sort by absolute weight
weights_df['abs_weight'] = weights_df['weight'].abs()
weights_df = weights_df.sort_values('abs_weight', ascending=False)

# Print feature weights
print(weights_df)

"""print("Plotting accuracy scores for different hyperparameters...")
# Get mean test scores
mean_test_scores = grid_search.cv_results_['mean_test_score']
# Create a DataFrame to store the scores and corresponding hyperparameters
scores_df = pd.DataFrame({'C': param_grid['C'], 'Mean Test Score': mean_test_scores})
# Plot the scores
plt.figure(figsize=(10,6))
plt.plot(scores_df['C'], scores_df['Mean Test Score'], marker='o')
plt.title('Model Accuracy for Different Hyperparameters')
plt.xlabel('C')
plt.ylabel('Mean Test Score')
plt.xscale('log')
plt.grid(True)
plt.savefig('shared_images/paper_a_6_ml_model_accuracy.png')"""


# Plot feature weights
print("Plotting feature weights...")

# Increase the figure size
plt.figure(figsize=(12, 8))

# Plot the data
sns.barplot(x='weight', y='feature', data=weights_df,
            order=weights_df.sort_values('weight',ascending=False).feature,
            palette='Greens')

# Add titles and labels with potentially larger font size
plt.title('Feature Weights', fontsize=16)
plt.xlabel('Weight', fontsize=14)
plt.ylabel('Feature', fontsize=14)

# Adjust label size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# If x-axis labels need to be rotated
#plt.xticks(rotation=45)

# Ensure that the labels are fully visible
plt.tight_layout()
# Save Image
plt.savefig('shared_images/paper_a_8_ml_model_feature_weights.png')

# Create confusion matrix
print("Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["monologic", "dialogic"], columns=["monologic", "dialogic"])

plot_confusion_matrix(y_test, y_pred,
                      class_names,
                      "paper_a_7_ml_confusion_matrix.png",
                      "Confusion Matrix for SVM Model",
                      values_fontsize=22)

# Print metrics
print("\nModel Support Vector Machine (SVM) Model\n")
print(f"- Accuracy: {accuracy}")
print(f"- Precision: {precision}")
print(f"- Recall: {recall}")
print(f"- F1 Score: {f1}")
print(f"- AUC-ROC: {roc_auc}")
print(f"- Matthews Correlation Coefficient (MCC): {mcc})")
print(f"- Confusion Matrix:\n{cm_df}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(svm, X, y, cv=cv, scoring='accuracy')

print(f"\nCV Accuracy: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")

# Save the trained model and scaler
joblib.dump(grid_search.best_estimator_, 'models/paper_a/ml/svm_model.pkl')
joblib.dump(scaler, 'models/paper_a/ml/scaler.pkl')
