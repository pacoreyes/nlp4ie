from pprint import pprint
from collections import Counter
import json

from tqdm import tqdm
import spacy

from db import spreadsheet_7
from lib.utils import read_from_google_sheet, write_to_google_sheet, save_jsonl_file
# from lib.ner_processing import custom_anonymize_text
from lib.continuity_checks import (check_lexical_continuity, check_syntactic_continuity, check_semantic_continuity,
                                   check_transition_markers_continuity, check_coreference)
from lib.text_utils import preprocess_text
from paper_c__4_inference_bert import inference_pair


dataset = read_from_google_sheet(spreadsheet_7, "_test_dataset")
# dataset = dataset[62:63]

nlp_trf = spacy.load("en_core_web_trf")

# Initialize path and name of output JSON-L file
output_sheet = "test_dataset"

output_misclassifications = "EA"

# Initialize a JSONL file for the dataset
# empty_json_file(output_file)


"""#############################################
Step 1: Extract continuity features, reclassify, and save to Google Sheets
#############################################"""

misclassification_data = []

for idx, datapoint in tqdm(enumerate(dataset, start=1), desc=f"Reclassifying {len(dataset)} datapoints", total=len(dataset)):
  # Split the text into two sentences
  sentences = datapoint["text"].split(" [SEP] ")
  sent1 = sentences[0].strip()
  sent2 = sentences[1].strip()

  sent1 = preprocess_text(sent1, nlp_trf,
                          with_remove_known_unuseful_strings=False,
                          with_remove_parentheses_and_brackets=False,
                          with_remove_text_inside_parentheses=False,
                          with_remove_leading_patterns=False,
                          with_remove_timestamps=False,
                          with_replace_unicode_characters=False,
                          with_expand_contractions=True,
                          with_remove_links_from_text=False,
                          with_put_placeholders=False,
                          with_final_cleanup=False)

  sent2 = preprocess_text(sent2, nlp_trf,
                          with_remove_known_unuseful_strings=False,
                          with_remove_parentheses_and_brackets=False,
                          with_remove_text_inside_parentheses=False,
                          with_remove_leading_patterns=False,
                          with_remove_timestamps=False,
                          with_replace_unicode_characters=False,
                          with_expand_contractions=True,
                          with_remove_links_from_text=False,
                          with_put_placeholders=False,
                          with_final_cleanup=False)

  # Combine the two sentences into one with contractions expanded
  datapoint["text"] = f"{sent1} [SEP] {sent2}"

  continuity = []
  not_continuity = []

  lexical_continuity = check_lexical_continuity(sent1, sent2)
  if lexical_continuity['lexical_continuity']:
    continuity.append(lexical_continuity)

  syntactic_continuity = check_syntactic_continuity(sent1, sent2)
  if syntactic_continuity['syntactic_continuity']:
    syntactic_continuity['syntactic_continuity'] = list(syntactic_continuity['syntactic_continuity'])
    continuity.append(syntactic_continuity)

  semantic_continuity = check_semantic_continuity(sent1, sent2)
  if semantic_continuity['semantic_continuity']:
    continuity.append(semantic_continuity)

  transition_markers = check_transition_markers_continuity(sent2)

  coreference = check_coreference(sent1, sent2)
  if coreference['coreference']:
    continuity.append(coreference)

  if transition_markers['transition_markers']:
    #continuity.append(transition_markers)
    if transition_markers['transition_markers'][0].get('continue'):
      continuity.append(transition_markers)
    elif transition_markers['transition_markers'][0].get('shift'):
      not_continuity.append(transition_markers)

  if continuity:
    datapoint['reclass'] = "continue"
    datapoint['continuity'] = continuity
    if not_continuity:
      datapoint['continuity'].append(not_continuity[0])
  else:
    datapoint['reclass'] = "not_continue"
    datapoint['continuity'] = not_continuity

  # Inference BERT model
  datapoint["label_bert"] = inference_pair([datapoint["text"]])
  if datapoint["label"] != datapoint["label_bert"]:
    row = [
      idx,
      datapoint["label"],
      datapoint["label_bert"]
    ]
    misclassification_data.append(row)

print(f"Reclassified dataset: {len(dataset)} datapoints\n")

# BEAUTIFY_JSON = True

new_dataset = []
continue_class = []
not_continue_class = []

id_counter = 0

for idx, _datapoint in tqdm(enumerate(dataset), desc=f"Uploading {len(dataset)} datapoints", total=len(dataset)):
  # Remove empty continuity
  continuity = str(_datapoint["continuity"])
  if continuity == "[{'transition_markers': []}]" or continuity == "[]":
    _datapoint["continuity"] = ""
    _datapoint["continuity_ugly"] = ""
  else:
    # if BEAUTIFY_JSON:
      # Format JSON continuity into a compact representation with smaller indentation
      # print(_datapoint["continuity"])
    _datapoint["continuity"] = json.dumps(_datapoint["continuity"], indent=2)
    # else:
    _datapoint["continuity_ugly"] = continuity

  id_counter += 1

  # Create new row
  _row = [
    id_counter,
    _datapoint["id"],
    _datapoint["passage_id"],
    _datapoint["text"],
    _datapoint["label"],
    _datapoint["reclass"],
    _datapoint["label_bert"],
    _datapoint["continuity"]
  ]
  new_dataset.append(_row)

  row = {
    "id": id_counter,
    "source+id": _datapoint["id"],
    "passage_id": _datapoint["passage_id"],
    "text": _datapoint["text"],
    "label_human": _datapoint["label"],
    "label_rb": _datapoint["reclass"],
    "label_bert": _datapoint["label_bert"],
    "metadata": _datapoint["continuity_ugly"]
    }

  if _datapoint["label"] == "continue":
    continue_class.append(row)
  else:
    not_continue_class.append(row)

# pprint(new_dataset)

print(f"Misclassifications: {len(misclassification_data)}")

write_to_google_sheet(spreadsheet_7, output_sheet, new_dataset)
write_to_google_sheet(spreadsheet_7, output_misclassifications, misclassification_data)
save_jsonl_file(continue_class, "shared_data/topic_boundary_continue_class.jsonl")
save_jsonl_file(not_continue_class, "shared_data/topic_boundary_not_continue_class.jsonl")


print(f"Uploaded dataset: {len(new_dataset)} datapoints")

counter = Counter([datapoint["reclass"] for datapoint in dataset])
continue_percentage = counter["continue"] / (counter["continue"] + counter["not_continue"]) * 100
not_continue_percentage = counter["not_continue"] / (counter["continue"] + counter["not_continue"]) * 100

print()
print("Class distribution after reclassification")
print(f"• Continue: {counter['continue']} ({continue_percentage:.2f})")
print(f"• Not continue: {counter['not_continue']} ({not_continue_percentage:.2f})")
