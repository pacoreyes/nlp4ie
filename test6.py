import json

from lib.stance_markers_adv import (positive_adv,
                                     negative_adv,
                                     certainty_adv, certainty_adv_phrase,
                                     doubt_adv,
                                     emphatic_adv, emphatic_adv_phrase,
                                     hedge_adv, hedge_adv_phrase,
                                     pro_adv, pro_adv_phrase,
                                     con_adv, con_adv_phrase,
                                     )

from collections import Counter


def find_duplicates(*lists):
  # Flatten the list of lists into a single list of strings
  all_strings = [item for sublist in lists for item in sublist]

  # Create a counter to count occurrences of each string
  counter = Counter(all_strings)

  # Filter the counter to only include items that occur more than once
  duplicates = [string for string, count in counter.items() if count > 1]

  return duplicates


duplicated_strings = find_duplicates(positive_adv,
                                     negative_adv,
                                     certainty_adv, certainty_adv_phrase,
                                     doubt_adv,
                                     emphatic_adv, emphatic_adv_phrase,
                                     hedge_adv, hedge_adv_phrase,
                                     pro_adv, pro_adv_phrase,
                                     con_adv, con_adv_phrase,
                                     )
print(duplicated_strings)
