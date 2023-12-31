import spacy
import tqdm

from lib.text_utils import preprocess_text, remove_leading_placeholders
from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file

# load spaCy's Transformer model
# install the model with: python -m spacy download en_core_web_trf
nlp = spacy.load("en_core_web_trf")

PREPROCESS_TEXT = True
REMOVE_PLACEHOLDERS = True
ANONYMIZE = True

output_file = "shared_data/dataset_1_3.jsonl"

# Load the JSONL file with all the datapoints
dataset_raw = load_jsonl_file('shared_data/dataset_1_1_raw.jsonl')

# Empty the output JSONL
empty_json_file(output_file)

# Initialize a list of entities not anonymized by spaCy
custom_entities = [
  "COVID-19",
  "COVID",
  "Army",
  "WeCanDoThis.HHS.gov",
  "HIV",
  "AIDS"
]


def anonymize_text(_text, _nlp):
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
  """

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
  return _text


""" #######################################################################
Preprocess text
########################################################################"""

# Process all datapoints in Dataset 1

for idx, datapoint in tqdm.tqdm(enumerate(dataset_raw),
                                desc=f"Processing {len(dataset_raw)} datapoints", total=len(dataset_raw)):
  text = datapoint["text"]
  # convert list of strings to a single string
  text = " ".join(text)
  if PREPROCESS_TEXT:
    text = preprocess_text(text, nlp,
                           with_remove_known_unuseful_strings=False,
                           with_remove_parentheses_and_brackets=False,
                           with_remove_text_inside_parentheses=True,
                           with_remove_leading_patterns=False,
                           with_remove_timestamps=False,
                           with_replace_unicode_characters=True,
                           with_expand_contractions=True,
                           with_remove_links_from_text=True,
                           with_put_placeholders=False,
                           with_final_cleanup=True)
  if REMOVE_PLACEHOLDERS:
    text = remove_leading_placeholders(text)
  if ANONYMIZE:
    text = anonymize_text(text, nlp)
    # Anonymize additional entities
    for entity in custom_entities:
      pair = text.replace(entity, "[ENTITY]")
  slots = {
    "id": datapoint["id"],
    "text": text,
    "discourse_type": datapoint["discourse_type"],
    "metadata": datapoint["metadata"]
  }

  save_row_to_jsonl_file(slots, output_file)

print()
print("Process finished.")
