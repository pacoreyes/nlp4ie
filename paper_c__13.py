import pandas as pd

# Load the data from the provided Excel file
data_path = '/mnt/data/linguistic_features.xlsx'
data = pd.read_excel(data_path, sheet_name=None)

# Check the names of the sheets and load the data from both sheets
data.keys(), data['continue'].head(), data['not_continue'].head()


from scipy.stats import skew, kurtosis

# Load the actual data from both sheets
continue_data = data.parse('continue')
not_continue_data = data.parse('not_continue')

# Function to calculate descriptive statistics
def calculate_statistics(df):
    # Group by feature and calculate statistics
    grouped = df.groupby('feature')['value']
    stats = grouped.agg(['mean', 'min', 'max', 'std', 'var'])
    stats['range'] = stats['max'] - stats['min']
    stats['skewness'] = grouped.apply(skew)
    stats['kurtosis'] = grouped.apply(kurtosis)
    return stats.drop(columns=['min', 'max'])

# Calculate statistics for both classes
stats_continue = calculate_statistics(continue_data)
stats_not_continue = calculate_statistics(not_continue_data)


# stats_continue, stats_not_continue

import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot histograms for each feature
def plot_histograms(data1, data2, feature):
    plt.figure(figsize=(10, 6))
    sns.histplot(data1[data1['feature'] == feature]['value'], color="green", label='Continue', kde=True, bins=30)
    sns.histplot(data2[data2['feature'] == feature]['value'], color="orange", label='Not Continue', kde=True, bins=30)
    plt.title(f'Distribution of {feature} Feature Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate histograms for each feature
features = continue_data['feature'].unique()
for feature in features:
    plot_histograms(continue_data, not_continue_data, feature)


