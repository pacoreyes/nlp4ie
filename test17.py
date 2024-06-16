import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Data
shap_values = [
  {'feature': 'polarity_pro', 'total': 13.14469369912314, 'total_oppose': -0.22475094151929279, 'total_support': 13.369444640642435},
  {'feature': 'positive_affect', 'total': 8.484477900450578, 'total_oppose': -0.24856920378092845, 'total_support': 8.733047104231506},
  {'feature': 'certainty', 'total': 2.8993374641477665, 'total_oppose': -1.2220057574152392, 'total_support': 4.121343221563005},
  {'feature': 'doubt', 'total': -1.2499886105555928, 'total_oppose': -1.5370165060537697, 'total_support': 0.287027895498177},
  {'feature': 'emphatic', 'total': 3.9923184506144085, 'total_oppose': -4.673472380332836, 'total_support': 8.66579083094724},
  {'feature': 'hedge', 'total': -0.44298058228901044, 'total_oppose': -2.4134265138002453, 'total_support': 1.9704459315112341},
  {'feature': 'negative_affect', 'total': -6.073322276066617, 'total_oppose': -6.169214687924749, 'total_support': 0.09589241185813296},
  {'feature': 'polarity_con', 'total': -17.979790725999916, 'total_oppose': -18.155276419445, 'total_support': 0.17548569344508697}
]

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
shap_processed = [abs(abs(item['total_support']) - abs(item['total_oppose'])) for item in shap_values]
lr_processed = [abs(item['total']) for item in lr_coeff]

# Normalizing
scaler = MinMaxScaler()
shap_normalized = scaler.fit_transform(np.array(shap_processed).reshape(-1, 1)).flatten()
lr_normalized = scaler.fit_transform(np.array(lr_processed).reshape(-1, 1)).flatten()

# Labels
features = ["Polarity Pro", "Positive Affect", "Epistemic Certainty", "Epistemic Doubt",
            "Emphatic", "Hedge", "Negative Affect", "Polarity Con"]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(features, shap_normalized, label='SHAP Values', color='orange', marker='o')
plt.plot(features, lr_normalized, label='LR Coefficients', color='green', marker='o')
plt.title('Comparison of SHAP Values and Logistic Regression Coefficients')
plt.ylabel('Values')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bland-Altman Plot Analysis and Visualization

# Calculate differences and averages for Bland-Altman plot
differences = np.array(shap_normalized) - np.array(shap_processed)
averages = (np.array(lr_normalized) + np.array(lr_processed)) / 2

# Mean and standard deviation of the differences
mean_diff = np.mean(differences)
std_diff = np.std(differences)

# Plotting the Bland-Altman plot
fig, ax = plt.subplots(figsize=(14, 6))

# Scatter plot of differences against averages
ax.scatter(averages, differences, color='blue', label='Differences vs. Averages')

# Add lines for mean difference and limits of agreement
ax.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference')
ax.axhline(mean_diff + 1.96 * std_diff, color='green', linestyle='--', label='Upper Limit of Agreement (Mean + 1.96*SD)')
ax.axhline(mean_diff - 1.96 * std_diff, color='green', linestyle='--', label='Lower Limit of Agreement (Mean - 1.96*SD)')

# Labels, title and legend
ax.set_xlabel('Average of SHAP Values and LR Coefficients')
ax.set_ylabel('Difference between SHAP Values and LR Coefficients')
ax.set_title('Bland-Altman Plot')
ax.legend()

plt.tight_layout()
plt.show()

