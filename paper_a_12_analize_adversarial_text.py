# from pprint import pprint

from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
# import numpy as np
import pandas as pd

# from lib.utils import save_row_to_jsonl_file

speech_frequencies = [
  ('people', 5173),
  ('year', 3404),
  ('go', 3237),
  ('want', 3227),
  ('know', 3210),
  ('country', 3041),
  ('work', 2760),
  ('america', 2558),
  ('great', 2310),
  ('time', 2286),
  ('world', 2261),
  ('good', 2207),
  ('say', 2161),
  ('come', 2114),
  ('american', 2056),
  ('get', 1958),
  ('like', 1897),
  ('think', 1874),
  ('new', 1864),
  ('job', 1772),
  ('nation', 1677),
  ('president', 1672),
  ('need', 1627),
  ('way', 1624),
  ('help', 1623),
  ('government', 1579),
  ('right', 1559),
  ('thing', 1531),
  ('united', 1497),
  ('today', 1419),
  ('state', 1418),
  ('child', 1324),
  ('states', 1317),
  ('day', 1298),
  ('tell', 1294),
  ('life', 1284),
  ('man', 1223),
  ('thank', 1142),
  ('believe', 1128),
  ('family', 1074),
  ('congress', 1065),
  ('americans', 1054),
  ('million', 1054),
  ('look', 1045),
  ('lot', 1045),
  ('let', 1045),
  ('take', 987),
  ('school', 987),
  ('well', 953),
  ('mean', 930)
]

interview_frequencies = [
  ('think', 6736),
  ('people', 5721),
  ('go', 4835),
  ('know', 3855),
  ('thing', 2772),
  ('say', 2752),
  ('get', 2714),
  ('want', 2549),
  ('country', 2501),
  ('time', 2377),
  ('work', 2306),
  ('year', 2276),
  ('way', 2104),
  ('come', 2065),
  ('president', 2060),
  ('like', 2010),
  ('lot', 1964),
  ('entity', 1949),
  ('good', 1830),
  ('look', 1735),
  ('talk', 1730),
  ('try', 1671),
  ('right', 1626),
  ('united', 1385),
  ('government', 1320),
  ('issue', 1298),
  ('world', 1293),
  ('believe', 1275),
  ('american', 1221),
  ('need', 1216),
  ('states', 1211),
  ('mean', 1185),
  ('question', 1154),
  ('problem', 1110),
  ('ask', 1088),
  ('let', 1070),
  ('take', 1068),
  ('help', 1060),
  ('job', 1053),
  ('tell', 1032),
  ('kind', 1026),
  ('important', 997),
  ('deal', 996),
  ('state', 993),
  ('happen', 970),
  ('great', 951),
  ('sure', 949),
  ('new', 922),
  ('congress', 915),
  ('fact', 906)
]

# Convert the top 10 tuples to dictionaries for easier access
speech_frequencies_dict = dict(speech_frequencies[:50])
interview_frequencies_dict = dict(interview_frequencies[:50])

# Generate a combined list of unique terms from the top 10 of each class for comparison
combined_terms = list(set(speech_frequencies_dict.keys()) | set(interview_frequencies_dict.keys()))

# chi_square_results = {}

# Assuming total counts for speech and interview represent the sum of frequencies for their top 10 terms
total_speech = sum(speech_frequencies_dict.values())
total_interview = sum(interview_frequencies_dict.values())

chi_square_calcs = {}
chi_square_results = []

for term in combined_terms:
  # Frequencies of the term in each class
  freq_speech = speech_frequencies_dict.get(term, 0)
  freq_interview = interview_frequencies_dict.get(term, 0)

  # Contingency table for the term across the two classes
  contingency_table = [
    [freq_speech, total_speech - freq_speech],  # Frequency in speech and remaining frequency in speech
    [freq_interview, total_interview - freq_interview]  # Frequency in interview and remaining frequency in interview
  ]

  # Performing the Chi-square test
  chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

  if p_value < 0.001:
    # Determine which class may be biased with this term
    biased_class = "speech" if freq_speech > expected[0][0] else "interview"
    # Determine the frequency from the appropriate list
    if biased_class == "speech":
      frequency = speech_frequencies_dict[term]
    else:
      frequency = interview_frequencies_dict[term]

    chi_square_calcs[term] = {
      "statistic": chi2_stat,
      "p-value": p_value,
      "biased class": biased_class,
      "frequency": frequency
    }

    # Create JSONL row
    row = {
      "term": term,
      "statistic": chi2_stat,
      "p-value": p_value,
      "biased_class": biased_class,
      "frequency": frequency
    }
    chi_square_results.append(row)
    # save_row_to_jsonl_file(row, "data.jsonl")

df = pd.DataFrame(chi_square_results)

# Normalize the Chi2 Statistic for better color mapping
df['statistic_norm'] = (df['statistic'] - df['statistic'].min()) / (df['statistic'].max() - df['statistic'].min())

# Create a pivot table for the heatmap
pivot_table = df.pivot(columns=["term", "biased_class", "p-value", "statistic_norm", "statistic", "frequency"])

# Sort the pivot table by biased class and normalized statistic
sorted_words = df.sort_values(by=['biased_class', 'p-value'], ascending=[True, False])['term']
pivot_table = pivot_table.reindex(sorted_words)

# Define a custom colormap: red for 'speech', blue for 'interview'
colors = ["red" if cls == "speech" else "blue" for cls in df.sort_values(
  by=['biased_class', 'statistic_norm'], ascending=[True, False])['biased_class']]

cmap = sns.blend_palette(colors, as_cmap=True)

speech_df = df[df['biased_class'] == 'speech'].sort_values(by='statistic_norm', ascending=False).set_index('term')
# Trim the dataframe to the top 25 terms
speech_df = speech_df.head(25)

interview_df = df[df['biased_class'] == 'interview'].sort_values(by='statistic_norm', ascending=False).set_index('term')
interview_df = interview_df.head(25)

# Define the color map using lighter colors
light_red_cmap = sns.light_palette("red", as_cmap=True)
light_blue_cmap = sns.light_palette("blue", as_cmap=True)

sns.set(font_scale=1.4)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 12), gridspec_kw={'width_ratios': [1, 1]})

interview_df['formatted_frequency'] = interview_df['frequency'].apply(lambda x: f"{x:,}")
interview_df['formatted_frequency'] = interview_df['formatted_frequency'].astype(str)

speech_df['formatted_frequency'] = speech_df['frequency'].apply(lambda x: f"{x:,}")
speech_df['formatted_frequency'] = speech_df['formatted_frequency'].astype(str)

# Plotting the Speech heatmap on the first axes
# sns.heatmap(speech_df[['statistic_norm']], annot=True, fmt=".2f", cmap=light_red_cmap, cbar=False, ax=ax1)
sns.heatmap(
  speech_df[['statistic_norm']],
  # annot=speech_df[['frequency']].values,
  annot=speech_df[['formatted_frequency']].values.astype(str),
  fmt="s",
  cmap=light_red_cmap,
  cbar=False,
  ax=ax1,
  annot_kws={'size': 14},
  xticklabels=[]
)

ax1.set_title('Speech', pad=16, fontsize=16)
# ax1.set_xlabel('statistic_norm')
ax1.set_ylabel('')
ax1.set_xlabel('')
ax1.tick_params(axis='y', rotation=0)

# Plotting the Interview heatmap on the second axes
# sns.heatmap(interview_df[['statistic_norm']], annot=True, fmt=".2f", cmap=light_blue_cmap, cbar=False, ax=ax2)
sns.heatmap(
  interview_df[['statistic_norm']],
  # annot=interview_df[['frequency']].values,
  annot=interview_df[['formatted_frequency']].values.astype(str),
  fmt="s",
  cmap=light_blue_cmap,
  cbar=False,
  ax=ax2,
  annot_kws={'size': 14},
  xticklabels=[]
)
ax2.set_title('Interview', pad=16, fontsize=16)
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.tick_params(axis='y', rotation=0)  # Set the y-tick labels to horizontal

# Set the x-axis label for both heatmaps
ax1.set_xlabel('Word Frequency', fontsize=16)
ax2.set_xlabel('Word Frequency', fontsize=16)

# Adjust the layout
# plt.title('Test')
# plt.suptitle('Chi-Square Analysis of Word Frequency Bias in Political Speeches and Interviews')
plt.tight_layout(rect=(0, 0.03, 1, 0.99))  # Adjust the padding to provide space for the main title

plt.savefig('images/paper_a_17_chi2_word_bias_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Sort the results by p-value
chi_square_calcs = {
  term: results for term, results in sorted(chi_square_calcs.items(), key=lambda item: item[1]['p-value'])
}

for term, results in chi_square_calcs.items():
  chi2_stat = results['statistic']
  p_value = results['p-value']
  biased_class = results['biased class']
  frequency = results['frequency']
  print(f"'{term} ({frequency})' (p-value: {p_value} / stats: {chi2_stat:.3f}) signals bias to '{biased_class}' class.")
