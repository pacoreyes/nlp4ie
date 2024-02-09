import spacy
import tqdm

from lib.linguistic_utils import check_minimal_meaning
from lib.text_utils import preprocess_text, remove_speaker_labels
from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file
from lib.ner_processing import anonymize_text

PREPROCESS_TEXT = True
REMOVE_SPEAKER_LABELS = True

# load spaCy's Transformer model
nlp = spacy.load("en_core_web_trf")

# load dataset
output_file = "shared_data/dataset_1_3_preprocessed_b.jsonl"
output_file_anonym = "shared_data/dataset_1_3_preprocessed_b_anonym.jsonl"

# Load the JSONL file with all the datapoints
dataset = load_jsonl_file('shared_data/dataset_1_1_raw.jsonl')

# Empty the output JSONL
empty_json_file(output_file)
empty_json_file(output_file_anonym)

# Initialize a list of entities not anonymized by spaCy
custom_entities = [
  "COVID-19",
  "COVID",
  "Army",
  "WeCanDoThis.HHS.gov",
  "HIV",
  "AIDS"
]

# def anonymize_text(_text, _nlp):
"""  
  Anonymize text by replacing all entities with their labels.

  :param _text:
  :param _nlp:
  :return:

  List of entity labels that spaCy recognizes (https://spacy.io/api/annotation#named-entities):

  - PERSON: People, including fictional.
  - NORP: Nationalities or religious or political groups.
  - FAC: Buildings, airports, highways, bridges, etc.
  - ORG: Companies, agencies, institutions, etc.
  - GPE: Countries, cities, states.
  - LOC: Non-GPE locations, mountain ranges, bodies of water.
  - PRODUCT: Objects, vehicles, foods, etc. (Not services.)
  - EVENT: Named hurricanes, battles, wars, sports events, etc.
  - WORK_OF_ART: Titles of books, songs, etc.
  - LAW: Named documents made into laws.
  - LANGUAGE: Any named language.
  - DATE: Absolute or relative dates or periods.
  - TIME: Times smaller than a day.
  - PERCENT: Percentage, including "%".
  - MONEY: Monetary values, including unit.
  - QUANTITY: Measurements, as of weight or distance.
  - ORDINAL: “first”, “second”, etc.
  - CARDINAL: Numerals that do not fall under another type.


  doc = _nlp(_text)
  entities = [ent for ent in doc.ents if ent.label_ in [
    "PERSON",
    "ORG",
    "NORP",
    "TIME",
    "DATE",
    "CARDINAL",
    "MONEY",
    "FAC",
    "QUANTITY",
    "PERCENT",
    "GPE"]
              ]
  sorted_entities = sorted(entities, key=lambda e: e.start_char, reverse=True)
  for ent in sorted_entities:
    _text = _text[:ent.start_char] + "[" + ent.label_ + "]" + _text[ent.end_char:]
  return _text"""


""" #######################################################################
Preprocess text
########################################################################"""

# Process all datapoints in Dataset 1

for idx, datapoint in tqdm.tqdm(enumerate(dataset),
                                desc=f"Processing {len(dataset)} datapoints", total=len(dataset)):
  text = datapoint["text"]
  # convert single string to list of sentences using spaCy
  text = [sent.text for sent in nlp(text).sents]

  # Remove sentences without minimal meaning
  text = [sent for sent in text if check_minimal_meaning(nlp(sent))]

  new_text = []
  for sent in text:
    sent = preprocess_text(sent, nlp,
                           with_remove_known_unuseful_strings=False,
                           with_remove_parentheses_and_brackets=False,
                           with_remove_text_inside_parentheses=False,
                           with_remove_leading_patterns=True,
                           with_remove_timestamps=False,
                           with_replace_unicode_characters=True,
                           with_expand_contractions=False,
                           with_remove_links_from_text=False,
                           with_put_placeholders=False,
                           with_final_cleanup=False)

    if REMOVE_SPEAKER_LABELS:
      sent = remove_speaker_labels(sent)
    new_text.append(sent)

  # Text non-anonymized
  text = " ".join(new_text)

  # Replace speaker labels with placeholders
  """for label in SPEAKER_LABELS:
    text = text.replace(label, "SPEAKER")"""

  # Text anonymized
  text_anonym = anonymize_text(text, nlp)

  for entity in custom_entities:
    text = text.replace(entity, "ENTITY")

  row = {
    "id": datapoint["id"],
    "text": text,
    "label": datapoint["label"],
    "metadata": datapoint["metadata"]
  }
  save_row_to_jsonl_file(row, output_file)
  row["text"] = text_anonym
  save_row_to_jsonl_file(row, output_file_anonym)

print()
print("Process finished.")
