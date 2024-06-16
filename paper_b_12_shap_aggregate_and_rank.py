from pprint import pprint

import spacy
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Import stance markers for adjectives
from lib.stance_markers_adj import (
  positive_adj,
  negative_adj,
  certainty_adj,
  doubt_adj,
  emphatic_adj,
  hedge_adj,
  pro_adj,
  con_adj,
)
# Import stance markers for adverbs
from lib.stance_markers_adv import (
  positive_adv,
  negative_adv,
  certainty_adv,
  doubt_adv,
  emphatic_adv,
  hedge_adv,
  pro_adv,
  con_adv,
)
# Import stance markers for verbs
from lib.stance_markers_verb import (
  positive_verb,
  negative_verb,
  certainty_verb,
  doubt_verb,
  emphatic_verb,
  hedge_verb,
  pro_verb,
  con_verb,
)
# Import stance markers for modality
from lib.stance_markers_modals import (
  predictive_modal,
  possibility_modal,
  necessity_modal,
)
from db import spreadsheet_4
from lib.utils import read_from_google_sheet, save_json_file

# Set spaCy to use GPU if available
if spacy.prefer_gpu():
    print("spaCy is using GPU!")
else:
    print("GPU not available, spaCy is using CPU instead.")

nlp = spacy.load("en_core_web_trf")

# Lemmatize the stance markers
positive_adj = [nlp(item)[0].lemma_ for item in positive_adj]
positive_adv = [nlp(item)[0].lemma_ for item in positive_adv]
positive_verb = [nlp(item)[0].lemma_ for item in positive_verb]
negative_adj = [nlp(item)[0].lemma_ for item in negative_adj]
negative_adv = [nlp(item)[0].lemma_ for item in negative_adv]
negative_verb = [nlp(item)[0].lemma_ for item in negative_verb]
certainty_adj = [nlp(item)[0].lemma_ for item in certainty_adj]
certainty_adv = [nlp(item)[0].lemma_ for item in certainty_adv]
certainty_verb = [nlp(item)[0].lemma_ for item in certainty_verb]
doubt_adj = [nlp(item)[0].lemma_ for item in doubt_adj]
doubt_adv = [nlp(item)[0].lemma_ for item in doubt_adv]
doubt_verb = [nlp(item)[0].lemma_ for item in doubt_verb]
emphatic_adj = [nlp(item)[0].lemma_ for item in emphatic_adj]
emphatic_adv = [nlp(item)[0].lemma_ for item in emphatic_adv]
emphatic_verb = [nlp(item)[0].lemma_ for item in emphatic_verb]
hedge_adj = [nlp(item)[0].lemma_ for item in hedge_adj]
hedge_adv = [nlp(item)[0].lemma_ for item in hedge_adv]
hedge_verb = [nlp(item)[0].lemma_ for item in hedge_verb]
pro_adj = [nlp(item)[0].lemma_ for item in pro_adj]
pro_adv = [nlp(item)[0].lemma_ for item in pro_adv]
pro_verb = [nlp(item)[0].lemma_ for item in pro_verb]
con_adj = [nlp(item)[0].lemma_ for item in con_adj]
con_adv = [nlp(item)[0].lemma_ for item in con_adv]
con_verb = [nlp(item)[0].lemma_ for item in con_verb]
predictive_modal = [nlp(item)[0].lemma_ for item in predictive_modal]
possibility_modal = [nlp(item)[0].lemma_ for item in possibility_modal]
necessity_modal = [nlp(item)[0].lemma_ for item in necessity_modal]

positive_affect = [
  {"name": "Positive Adj", "pos": "ADJ", "data": positive_adj},
  {"name": "Positive Adv", "pos": "ADV", "data": positive_adv},
  {"name": "Positive Verb", "pos": "VERB", "data": positive_verb},
]
negative_affect = [
  {"name": "Negative Adj", "pos": "ADJ", "data": negative_adj},
  {"name": "Negative Adv", "pos": "ADV", "data": negative_adv},
  {"name": "Negative Verb", "pos": "VERB", "data": negative_verb},
]
certainty = [
  {"name": "Certainty Adj", "pos": "ADJ", "data": certainty_adj},
  {"name": "Certainty Adv", "pos": "ADV", "data": certainty_adv},
  {"name": "Certainty Verb", "pos": "VERB", "data": certainty_verb},
]
doubt = [
  {"name": "Doubt Adj", "pos": "ADJ", "data": doubt_adj},
  {"name": "Doubt Adv", "pos": "ADV", "data": doubt_adv},
  {"name": "Doubt Verb", "pos": "VERB", "data": doubt_verb},
]
emphatic = [
  {"name": "Emphatic Adj", "pos": "ADJ", "data": emphatic_adj},
  {"name": "Emphatic Adv", "pos": "ADV", "data": emphatic_adv},
  {"name": "Emphatic Verb", "pos": "VERB", "data": emphatic_verb},
  {"name": "Predictive Modal", "pos": "AUX", "data": predictive_modal},
]
hedge = [
  {"name": "Hedge Adj", "pos": "ADJ", "data": hedge_adj},
  {"name": "Hedge Adv", "pos": "ADV", "data": hedge_adv},
  {"name": "Hedge Verb", "pos": "VERB", "data": hedge_verb},
  {"name": "Possibility Modal", "pos": "AUX", "data": possibility_modal},
  {"name": "Necessity Modal", "pos": "AUX", "data": necessity_modal}
]
pro_indicator = [
  {"name": "Pro Adj", "pos": "ADJ", "data": pro_adj},
  {"name": "Pro Adv", "pos": "ADV", "data": pro_adv},
  {"name": "Pro Verb", "pos": "VERB", "data": pro_verb},
  {"name": "Pro Noun", "pos": "NOUN", "data": [
    "commitment", "support", "help", "endorsement", "favor", "agreement", "effort", "importance"]},
]
con_indicator = [
  {"name": "Con Adj", "pos": "ADJ", "data": con_adj},
  {"name": "Con Adv", "pos": "ADV", "data": con_adv},
  {"name": "Con Verb", "pos": "VERB", "data": con_verb},
  {"name": "Con Noun", "pos": "NOUN", "data": [
    "concern", "opposition", "disagreement", "fight", "struggle", "battle", "attack"]},
  # {"name": "Con Preposition", "pos": "ADP", "data": ["against"]},
]

aggregated_features = [
  positive_affect,
  negative_affect,
  certainty,
  doubt,
  emphatic,
  hedge,
  pro_indicator,
  con_indicator
]

accumulated_positive = []
accumulated_negative = []
accumulated_certainty = []
accumulated_doubt = []
accumulated_emphatic = []
accumulated_hedge = []
accumulated_pro = []
accumulated_con = []

dataset = read_from_google_sheet(spreadsheet_4, "SHAP")

# dataset = dataset[:10]

for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):

  shap_values = eval(datapoint["shap_values"])

  for item in shap_values:
    token = nlp(item["feature"])
    if token:
      token = token[0]
      # check if token is punctuation
      if token.is_punct:
        continue
    else:
      continue

    # feature example = "positive_affect"
    for feature in aggregated_features:
      # sub_feature example = {"name": "Positive Adj", "pos": "ADJ", "data": positive_adj}
      for sub_feature in feature:
        if sub_feature["pos"] == token.pos_:
          if token.lemma_ in sub_feature["data"]:
            if feature == positive_affect:
              accumulated_positive.append(item)
            elif feature == negative_affect:
              accumulated_negative.append(item)
            elif feature == certainty:
              accumulated_certainty.append(item)
            elif feature == doubt:
              accumulated_doubt.append(item)
            elif feature == emphatic:
              accumulated_emphatic.append(item)
            elif feature == hedge:
              accumulated_hedge.append(item)
            elif feature == pro_indicator:
              accumulated_pro.append(item)
            elif feature == con_indicator:
              accumulated_con.append(item)

data = {
  "pro_indicator": accumulated_pro,
  "positive_affect": accumulated_positive,
  "certainty": accumulated_certainty,
  "doubt": accumulated_doubt,
  "emphatic": accumulated_emphatic,
  "hedge": accumulated_hedge,
  "negative_affect": accumulated_negative,
  "con_indicator": accumulated_con
}
save_json_file(data, "shared_data/dataset_2_4_shap_analysis.json")

_shap_values = []

for key, all_values in data.items():
  total_sum = sum(item['value'] for item in all_values)
  total_support = sum(item['value'] for item in all_values if item['value'] > 0)
  total_oppose = sum(item['value'] for item in all_values if item['value'] < 0)
  row = {
    "feature": key,
    "total": total_sum,
    "total_support": total_support,
    "total_oppose": total_oppose,
  }
  _shap_values.append(row)

pprint(_shap_values)

for d in _shap_values:
  print(f"{d['feature']}: {d['total']}")
  print(f"- Support: {d['total_support']}")
  print(f"- Oppose: {d['total_oppose']}")
  print()

lr_coeff = [
  {"feature": "polarity_pro", "total": 1.799},
  {"feature": "positive_affect", "total": 1.798},
  {"feature": "certainty", "total": 0.035},
  {"feature": "doubt", "total": 0.085},
  {"feature": "emphatic", "total": 0.038},
  {"feature": "hedge", "total": 0.321},
  {"feature": "negative_affect", "total": -2.097},
  {"feature": "polarity_con", "total": -3.303}
]

# Processing SHAP values
shap_processed = [abs(abs(item['total_support']) - abs(item['total_oppose'])) for item in _shap_values]
lr_processed = [abs(item['total']) for item in lr_coeff]

print("SHAP Values (differences)")
pprint(shap_processed)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Scale and then round the shap_processed array
shap_normalized = scaler.fit_transform(np.array(shap_processed).reshape(-1, 1)).flatten()
shap_normalized = np.round(shap_normalized, 3)

# Scale and then round the lr_processed array
lr_normalized = scaler.fit_transform(np.array(lr_processed).reshape(-1, 1)).flatten()
lr_normalized = np.round(lr_normalized, 3)

pprint(lr_normalized)
pprint(shap_normalized)

_features = [
  "Polarity Pro",
  "Positive Affect",
  "Epistemic Certainty",
  "Epistemic Doubt",
  "Emphatic",
  "Hedge",
  "Negative Affect",
  "Polarity Con"
]

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(_features, shap_normalized, label='SHAP Values', color='orange', marker='o')
plt.plot(_features, lr_normalized, label='LR Coefficients', color='green', marker='o')
plt.title('Comparison of SHAP Values and Logistic Regression Coefficients')
plt.ylabel('Values')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("images/paper_b_compare_shap_lr.png")

# Bland-Altman Plot Analysis and Visualization

# Calculate differences and averages for Bland-Altman plot
differences = np.array(shap_normalized) - np.array(lr_normalized)
averages = (np.array(shap_normalized) + np.array(lr_normalized)) / 2

# Mean and standard deviation of the differences
mean_diff = np.mean(differences)
std_diff = np.std(differences)

# Plotting the Bland-Altman plot
fig, ax = plt.subplots(figsize=(9, 6))

# Set the figure background color to a light green (10% green)
ax.patch.set_facecolor((0.95, 1, 0.95))

# Scatter plot of differences against averages
for i, feature in enumerate(_features):
    ax.scatter(averages[i], differences[i], color='green')
    # Adding a vertical offset to the label position
    label_y_pos = differences[i] + 0.01  # Adjust 0.02 or any suitable value that corresponds to about 5px
    ax.text(averages[i], label_y_pos, feature, fontsize=9, ha='center', va='bottom')

# Calculate padding for x-axis
x_range = max(averages) - min(averages)
padding = 0.1 * x_range  # 10% padding on each side

# Set new x-axis limits with padding
ax.set_xlim(min(averages) - padding, max(averages) + padding)

# Scatter plot of differences against averages
# ax.scatter(averages, differences, color='blue', label='Differences vs. Averages')

# Add lines for mean difference and limits of agreement
ax.axhline(mean_diff, color='blue', linestyle='--', label='Mean Difference')
ax.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label='Upper Limit of Agreement (Mean+1.96*SD)')
ax.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', label='Lower Limit of Agreement (Mean-1.96*SD)')

# Fill between the limits of agreement
# ax.fill_between(ax.get_xlim(), mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff, color='green', alpha=0.1)

# Labels, title and legend
ax.set_xlabel('Average of SHAP Values and LR Coefficients')
ax.set_ylabel('Difference of SHAP Values and LR Coefficients')
# ax.set_title('Bland-Altman Plot of LR and SetFit Models coefficients')
# ax.legend()

# Adding a dotted grid
ax.grid(True, linestyle=':', color='gray', linewidth=0.5)  # Set the grid style to dotted

plt.tight_layout()
plt.savefig("images/paper_b_bland_altman.png")
