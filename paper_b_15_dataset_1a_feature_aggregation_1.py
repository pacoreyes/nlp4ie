from tqdm import tqdm

from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file


"""
This script aggregates some features and groups them into parent features.
"""

# Load datasets
dataset_train = load_jsonl_file("shared_data/dataset_2_2_1a_train_features.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_2_2_1a_test_features.jsonl")

# Join datasets
# dataset_test = dataset_test + dataset_validation

# dataset = [dataset[15]]

# pprint(dataset)

"""dataset1 = dataset[:100]
dataset = dataset1 + dataset[-100:]"""

output_file_train = "shared_data/dataset_2_3_1a_train_features_aggregated_1.jsonl"
output_file_test = "shared_data/dataset_2_3_1a_test_features_aggregated_1.jsonl"

# Empty JSONL files
empty_json_file(output_file_train)
empty_json_file(output_file_test)

for _idx, dataset in enumerate([dataset_train, dataset_test]):

  for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):

    positive_affect = [
      datapoint["positive_adj"],
      datapoint["positive_adv"],
      datapoint["positive_verb"],
      ]
    negative_affect = [
      datapoint["negative_adj"],
      datapoint["negative_adv"],
      datapoint["negative_verb"],
      ]
    epistemic_certainty = [
      datapoint["certainty_adj"],
      datapoint["certainty_verb"],
      datapoint["certainty_adv"],
      datapoint["emphatic_adj"],
      datapoint["emphatic_adv"],
      datapoint["emphatic_verb"],
      datapoint["predictive_modal"],
    ]
    epistemic_doubt = [
      datapoint["doubt_adj"],
      datapoint["doubt_verb"],
      datapoint["doubt_adv"],
      datapoint["hedge_adj"],
      datapoint["hedge_adv"],
      datapoint["hedge_verb"],
      datapoint["possibility_modal"],
      datapoint["necessity_modal"],
    ]
    polarity_pro = [
      datapoint["pro_adj"],
      datapoint["pro_adv"],
      datapoint["pro_verb"],
    ]
    polarity_con = [
      datapoint["con_adj"],
      datapoint["con_adv"],
      datapoint["con_verb"],
    ]

    # Aggregate Features
    """aggr_positive_affect = [int(any(values)) for values in zip(*positive_affect)]
    aggr_negative_affect = [int(any(values)) for values in zip(*negative_affect)]
    aggr_epistemic_certainty = [int(any(values)) for values in zip(*epistemic_certainty)]
    aggr_epistemic_doubt = [int(any(values)) for values in zip(*epistemic_doubt)]
    aggr_pro_indicator = [int(any(values)) for values in zip(*polarity_pro)]
    aggr_con_indicator = [int(any(values)) for values in zip(*polarity_con)]"""

    aggr_positive_affect = 1 if any(1 in lst for lst in positive_affect) else 0
    aggr_negative_affect = 1 if any(1 in lst for lst in negative_affect) else 0
    aggr_epistemic_certainty = 1 if any(1 in lst for lst in epistemic_certainty) else 0
    aggr_epistemic_doubt = 1 if any(1 in lst for lst in epistemic_doubt) else 0
    # aggr_emphatics = 1 if any(1 in lst for lst in emphatics) else 0
    # aggr_hedge = 1 if any(1 in lst for lst in hedge) else 0
    aggr_pro = 1 if any(1 in lst for lst in polarity_pro) else 0
    aggr_con = 1 if any(1 in lst for lst in polarity_con) else 0

    row = {
      "id": datapoint["id"],
      "label": datapoint["label"],
      "text": datapoint["text"],

      "positive_affect": aggr_positive_affect,
      "negative_affect": aggr_negative_affect,
      "epistemic_certainty": aggr_epistemic_certainty,
      "epistemic_doubt": aggr_epistemic_doubt,
      "pro": aggr_pro,
      "con": aggr_con,

      "semantic_frames": datapoint["semantic_frames"],
    }

    # convert rec["metadata"] to dict
    if "metadata" in datapoint:
      row["metadata"] = eval(datapoint["metadata"])

    if _idx == 0:
      save_row_to_jsonl_file(row, output_file_train)
    else:
      save_row_to_jsonl_file(row, output_file_test)
    print(f"Saved row to file: {row['id']}")
