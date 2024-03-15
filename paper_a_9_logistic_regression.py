import shap
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

features = {
  # "sentence_length": "Sentence Length",
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

feature_names = list(features.values())

# Initialize constants
SEED = 42

class_names = ["monologic", "dialogic"]

# Load training dataset and extract features
df_train = pd.read_json("shared_data/dataset_1_5_1a_train_features_transformed.jsonl", lines=True)
X_train = df_train.drop(columns=["id", "label", "sentence_length"])

# Extract labels for the training data
y_train = df_train['label']

# Prepare the test dataset
df_test = pd.read_json("shared_data/dataset_1_5_1a_test_features_transformed.jsonl", lines=True)
X_test = df_test.drop(columns=["id", "label", "sentence_length"])

y_test = df_test['label']

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=SEED)

# Train the model
model.fit(X_train, y_train)

# Class "0" is dialogic (positive), and class "1" is monologic (negative)
print(model.classes_)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred, target_names=class_names, digits=3)
print(report)

# Initialize a SHAP Explainer object with your model and the training data
explainer = shap.Explainer(model, X_train)

# Calculate SHAP values for the test data
shap_values = explainer(X_test)

# Summary Plot
shap.summary_plot(
  shap_values,
  X_test,
  feature_names=feature_names,
  show=False
)
plt.savefig('images/paper_a_15_logistic_shap_summary_plot.png')
plt.close()

# Extract the coefficients from the model
coefficients = model.coef_[0]

# Associate each coefficient with its corresponding feature name
features_importance = pd.DataFrame(coefficients, index=feature_names, columns=["Coefficient"])

# Order the features by their coefficient values, ascending
features_importance = features_importance.sort_values(by="Coefficient", ascending=True)

print(features_importance)
# Order the features by their coefficient, descending
features_importance = features_importance.sort_values(by="Coefficient", ascending=False)

# Recreate the colors array based on the sorted DataFrame
colors_sorted = ['green' if x < 0 else 'darkorange' for x in features_importance['Coefficient']]

# Prepare the legend elements
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Interview', markersize=10, markerfacecolor='green'),
    Line2D([0], [0], marker='o', color='w', label='Speech', markersize=10, markerfacecolor='darkorange')
]

# Plotting
plt.figure(figsize=(14, 8))

# Plot each feature with its coefficient
for index, (feature, row) in enumerate(features_importance.iterrows()):
    plt.plot(row['Coefficient'], index, 'o', color=colors_sorted[index], markersize=12)
    text_offset = 0.2
    coeff_text = f"{row['Coefficient']:.3f}"
    plt.text(
        row['Coefficient'],
        index + text_offset,
        coeff_text,
        color=colors_sorted[index],
        fontsize=11,
        ha='center'
    )

plt.legend(handles=legend_elements, title="Classes", loc='upper right')


# Enhancements
plt.axvline(x=0, color='grey', linestyle='--')
plt.title('Feature Coefficients and Influence on Class Predictions - Logistic Regression')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.yticks(ticks=range(len(features_importance)), labels=features_importance.index)
plt.legend(handles=legend_elements, title="Classes", loc='best')

# Adjust the plot limits and layout
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim(min(features_importance['Coefficient']) - 1, max(features_importance['Coefficient']) + 1)
plt.tight_layout()

# Show the plot
plt.savefig('images/paper_a_16_logistic_feature_coefficients.png')

"""
3. Permutation Feature Importance
This technique involves randomly shuffling individual features in the validation set and measuring the change in the 
model's performance. A significant decrease in model performance after shuffling a feature indicates that the model 
relied heavily on that feature for predictions. This method is model-agnostic and can be a more reliable measure of 
feature importance, especially when features are correlated.
"""

"""# Assess feature importance using Permutation Feature Importance
result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=SEED)

# Store the importances in a DataFrame for easier visualization
perm_importances_df = pd.DataFrame(
  result.importances_mean,
  index=X_train.columns,
  columns=["Importance"]).sort_values(by="Importance", ascending=False)

print(perm_importances_df)"""

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

              precision    recall  f1-score   support

   monologic      0.875     0.883     0.879       111
    dialogic      0.879     0.870     0.874       108

    accuracy                          0.877       219
   macro avg      0.877     0.877     0.877       219
weighted avg      0.877     0.877     0.877       219

                     Coefficient  Absolute_Coefficient
interjection_d         -8.576805              8.576805
sentence_complexity    -3.454525              3.454525
lexical_d               3.043941              3.043941
word_length             2.727276              2.727276
sentence_length         1.770396              1.770396
modal_verb_d           -1.523589              1.523589
personal_pronoun_d      1.279159              1.279159
passive_voice_d        -0.930723              0.930723
discourse_marker_d     -0.674760              0.674760
nominalization_d        0.089162              0.089162


   monologic      0.875     0.883     0.879       111
    dialogic      0.879     0.870     0.874       108

    accuracy                          0.877       219
   macro avg      0.877     0.877     0.877       219
weighted avg      0.877     0.877     0.877       219

                     Importance (permutation)
interjection_d         0.260426
sentence_complexity    0.037747
lexical_d              0.023896
modal_verb_d           0.018874
word_length            0.017504
personal_pronoun_d     0.009132
sentence_length        0.005936
discourse_marker_d     0.004414
passive_voice_d        0.002131
nominalization_d       0.000152

"""
