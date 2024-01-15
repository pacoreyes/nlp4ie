import functools
import time

from google.cloud.exceptions import GoogleCloudError
import requests
import spacy
from afinn import Afinn

from db import firestore_db, spreadsheet_4
from lib.issues_matcher import match_issues
from lib.linguistic_utils import check_minimal_meaning
from lib.text_utils import preprocess_text
from lib.utils import read_from_google_sheet, load_txt_file

# from pprint import pprint

"""
This script extracts sentences with support/oppose stances from texts to create a corpus of sentences.
The script uses a fine-tuned BERT model to predict the the support/oppose stance of a sentence.
"""

# Load spacy models
nlp = spacy.load("en_core_web_sm")
# Load transformer models
nlp_trf = spacy.load("en_core_web_trf")

text_col_ref = firestore_db.collection("texts2")
dataset3_col_ref = firestore_db.collection("_dataset3")

speeches_ids = load_txt_file("shared_data/text_ids_speeches.txt")
interviews_ids = load_txt_file("shared_data/text_ids_interviews.txt")

# all_ids = speeches_ids + interviews_ids
all_ids = speeches_ids

# Get column name from tab "column_name" in the spreadsheet
rule_frames = read_from_google_sheet(spreadsheet_4, "stance_frames_rules")

# Initialize IP address and port
IP = "141.43.202.175"
# IP = "localhost"
PORT = "5000"

# Create a list of dicts with the first 2 columns of the dataframe, "name" and "object"
rule_frames = [{k: v.split(",") if k == "object" else v for k, v in frame.items()} for frame in rule_frames]
rule_frame_names = set(frame["name"] for frame in rule_frames)

# Initialize Afinn
afinn = Afinn()


@functools.lru_cache(maxsize=None)
def preprocess_sentence_cached(sentence):
  return preprocess_text(sentence, nlp_trf,
                         with_remove_known_unuseful_strings=True,
                         with_remove_parentheses_and_brackets=True,
                         with_remove_text_inside_parentheses=True,
                         with_remove_leading_patterns=True,
                         with_remove_timestamps=True,
                         with_replace_unicode_characters=True,
                         with_expand_contractions=False,
                         with_remove_links_from_text=True,
                         with_put_placeholders=False,
                         with_final_cleanup=True)


def process_sentence(sent):
  _doc = nlp_trf(sent)
  minimal_meaning = check_minimal_meaning(_doc)
  _matches = match_issues(sent)
  _matches = list({_match[4] for _match in _matches})
  # print(sent)
  # pprint(_matches)
  leading_patterns = [
    "Q."
  ]
  # check if the sentence start with any leading pattern
  has_leading_pattern = any(sent.startswith(pattern) for pattern in leading_patterns)

  if sent[0].isupper() and minimal_meaning and _matches and not has_leading_pattern:
    return _doc, _matches
  return None, None


def check_if_issue_is_main_topic(_issue, _text):
  _doc = nlp_trf(_text)

  # Initialize variables for analysis
  issue_dependencies = []
  issue_positions = []

  # Prepare the lemma of the issue
  issue_lemma = nlp_trf(_issue)[0].lemma_

  # Iterate through tokens to find all occurrences of the issue based on lemma
  for token in _doc:
    if token.lemma_ == issue_lemma:
      issue_dependencies.append(token.dep_)
      issue_positions.append(token.i)

  main_verb = None
  for token in doc:
    if token.dep_ == "ROOT":
      main_verb = token
      break

  # Determine if the issue is a main topic
  is_main_topic = False
  if issue_dependencies and main_verb:
    for dep, pos in zip(issue_dependencies, issue_positions):
      if dep in {"nsubj", "nsubjpass", "dobj", "pobj", "attr", "advcl"} and abs(pos - main_verb.i) <= 5:
        is_main_topic = True
        break

  return is_main_topic


def extract_frames(_sentence):
  try:
    response = requests.get(f"http://{IP}:{PORT}/api/extract_frames/{_sentence}").json()
    _extracted_frames = response["frames"]["textae"]
    return _extracted_frames
  except requests.RequestException:
    print("Network error...")
    return None


# Fill ID number with zeros
def id_with_zeros(number):
  return str(number).zfill(10)


def check_if_issue_is_in_string(issue, string):
  # compare the lemma of issue with every token in the string
  issue_lemma = nlp_trf(issue)[0].lemma_
  for token in nlp_trf(string):
    if token.lemma_ == issue_lemma:
      return True


def assess_stance(sentence, target_word):
  _doc = nlp(sentence)
  sentiment_score = 0

  for token in _doc:
    # Check for words related to the target word
    if token.head.lemma_ == target_word or token.lemma_ == target_word:
      # Add sentiment score of related words
      sentiment_score += afinn.score(token.text)

  # Determine stance
  if sentiment_score > 0:
    return "support"
  elif sentiment_score < 0:
    return "oppose"
  else:
    return "neutral"


def store_record_in_firestore(record, collection_ref, max_retries=5, wait_time=5):
  """
  Tries to store a record in Firestore, with retries on failure.

  Args:
  - record: The record to store.
  - collection_ref: Reference to the Firestore collection.
  - max_retries: Maximum number of retries.
  - wait_time: Wait time between retries in seconds.
  """
  retries = 0
  while retries < max_retries:
    try:
      # Attempt to store the record
      document_id = record["id"]
      collection_ref.document(document_id).set(record)
      print(f"Record stored successfully: {document_id}")
      return  # Exit the function on success
    except GoogleCloudError as e:
      print(f"Error storing record: {e}. Retrying... ({retries + 1}/{max_retries})")
      time.sleep(wait_time)  # Wait before retrying
      retries += 1

  print("Failed to store record after multiple retries.")


# Initialize counter
sentences_counter = 0
print()

for _id in all_ids:
  text_doc = text_col_ref.document(_id).get()
  rec = text_doc.to_dict()

  if "text_split" not in rec or "https://transcripts.cnn.com" in rec["url"]:
    print("#####################################################")
    print(f"Skipping {_id}...")
    print("#####################################################")
    continue

  text = rec["text_split"]

  # Preprocess each paragraph
  sentences = [preprocess_sentence_cached(sent) for sent in text]
  # Convert paragraphs into sentences
  # sentences = [sent.text for para in paragraphs for sent in nlp_trf(para).sents]
  # Filter empty sentences
  sentences = [(sent, process_sentence(sent)) for sent in sentences
               if any(token.is_alpha for token in nlp_trf(sent))]

  print("#####################################################")
  print(rec['id'])
  print("#####################################################\n")

  for sent, (doc, matches) in sentences:

    """_doc = nlp(sent)
    # check if sentence length is longer than 30 tokens
    if len(_doc) <= 40:
      continue"""

    if doc is None:
      continue

    extracted_frames = extract_frames(sent)
    if extracted_frames is None:
      continue

    # Check if any matched issue is the main topic of the sentence
    found_issue = None
    for match in matches:
      valid_sentence = check_if_issue_is_main_topic(match, sent)
      if valid_sentence:
        found_issue = match
        break

    if not found_issue:
      continue

    # Extract frame names from extracted frames
    extracted_frame_names = {frame["frame"] for frame in extracted_frames}

    # Find common frame names
    matched_frame_names = rule_frame_names.intersection(extracted_frame_names)

    # Extract denotations
    denotations = [frame["denotations"] for frame in extracted_frames]

    if matched_frame_names:

      sentence_is_about_issue = False

      for frame in extracted_frames:
        frame_name = frame["frame"]
        frame_objects = next((f["object"] for f in rule_frames if f["name"] == frame_name), None)

        for denotation in denotations:
          for d in denotation:
            # Check if any matched political issue is the argument of the frame
            if check_if_issue_is_in_string(found_issue, d["text"]) and d["role"] == "ARGUMENT":

              sentences_counter += 1
              sentence_id = id_with_zeros(sentences_counter)

              stance = assess_stance(sent, found_issue)

              print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
              print(f"> Sentence:", sent)
              print(f"> Issue:", found_issue)
              print(f"> Stance:", stance)
              print()
              # print(d)
              sentence_record = {
                "id": sentence_id,
                "text": sent,
                "main_frame": frame_name,
                # "matched_frames": matched_frame_names,
                "semantic_frames": extracted_frame_names,
                "frameNet_data": extracted_frames,
                "issue": found_issue,
                "source_url": rec["url"],
                "source": rec["id"],
                "stance": stance,
              }
              store_record_in_firestore(sentence_record, dataset3_col_ref)
              sentence_is_about_issue = True
              break

          if sentence_is_about_issue:
            break

        if sentence_is_about_issue:
          break

      sentence_is_about_issue = False
