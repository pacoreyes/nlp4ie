"""import csv


def calculate_mean(numbers):
  return sum(numbers) / len(numbers)


speech = []
with open('shared_data/dataset_1_9_speech_shap_features_plot_bar.csv', newline='') as csvfile:
  reader = csv.reader(csvfile)
  # Skip the header
  next(reader)
  # convert cvs to list
  reader = list(reader)
  reader = reader[:5000]
  for row in reader:
    # print(row[0])
    speech.append(len(row[1]))

print(f"Speech: {calculate_mean(speech)}")
print(len(speech))

interview = []
with open('shared_data/dataset_1_9_interview_shap_features_plot_bar.csv', newline='') as csvfile:
  reader = csv.reader(csvfile)
  # Skip the header
  next(reader)
  # convert cvs to list
  reader = list(reader)
  reader = reader[:5000]
  for row in reader:
    # print(row[0])
    interview.append(len(row[1]))

print(f"Interview: {calculate_mean(interview)}")
print(len(interview))"""

import pandas as pd


