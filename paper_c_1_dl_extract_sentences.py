"""
This script extracts sentences with support/oppose stances from texts to create a corpus of sentences.
The script uses a fine-tuned BERT model to predict the the support/oppose stance of a sentence.
"""

import functools
import time

import requests
import spacy
from google.cloud.exceptions import GoogleCloudError

from db import firestore_db, spreadsheet_4
from lib.issues_matcher import match_issues
from lib.text_utils import preprocess_text
from lib.utils import load_txt_file, read_from_google_sheet

# from pprint import pprint

# Set spaCy to use GPU if available
if spacy.prefer_gpu():
  print("spaCy is using GPU")
else:
  print("GPU not available, spaCy is not using GPU")

# Load spacy models
nlp = spacy.load("en_core_web_sm")
# Load transformer models
nlp_trf = spacy.load("en_core_web_trf")

text_col_ref = firestore_db.collection("texts2")
sentences_col_ref = firestore_db.collection("sentences3")

speeches_ids = load_txt_file("shared_data/text_ids_speeches.txt")
interviews_ids = load_txt_file("shared_data/text_ids_interviews.txt")
text2_ids = load_txt_file("shared_data/text_ids_texts2.txt")

# all_ids = speeches_ids + interviews_ids
# all_ids = interviews_ids
all_ids = text2_ids
print(f"Number of texts from Firestore (texts2): {len(all_ids)}")
# Join speeches and interviews ids
ids_to_remove = speeches_ids + interviews_ids
print(f"Number of texts to remove (speeches+interviews): {len(ids_to_remove)}")
# Remove ids from list
all_ids = [item for item in all_ids if item not in ids_to_remove]
print(f"Number of texts to process: {len(all_ids)}")

# Prune ids to the last index added
all_ids = all_ids[30427:]

# Get column name from tab "column_name" in the spreadsheet
rule_frames = read_from_google_sheet(spreadsheet_4, "stance_frames_rules")
# Create a list of dicts with the first 2 columns of the dataframe, "name" and "object"
rule_frames = [{k: v.split(",") if k == "object" else v for k, v in frame.items()} for frame in rule_frames]
rule_frame_names = set(frame["name"] for frame in rule_frames)

# define IP address and port
IP = "141.43.202.175"
# IP = "localhost"
PORT = "5000"


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


def check_undesirable_leading_patterns(sent):
  leading_patterns = [
    "Q."
  ]
  # check if the sentence start with any leading pattern
  return any(sent.startswith(pattern) for pattern in leading_patterns)


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
      if dep in {"nsubj", "nsubjpass", "dobj", "pobj", "attr", "advcl"} and abs(pos - main_verb.i) <= 7:
        is_main_topic = True
        break

  return is_main_topic


# Fill ID number with zeros
def id_with_zeros(number):
  return str(number).zfill(10)


def check_if_issue_is_in_string(issue, string):
  # compare the lemma of issue with every token in the string
  issue_lemma = nlp_trf(issue)[0].lemma_
  for token in nlp_trf(string):
    if token.lemma_ == issue_lemma:
      return True


def get_num_of_clauses(_doc):
  # Count the number of clauses (roughly estimated by the number of verbs)
  return sum(1 for token in _doc if token.pos_ == "VERB")


def extract_frames(_sentence):
  try:
    response = requests.get(f"http://{IP}:{PORT}/api/extract_frames/{_sentence}").json()
    _extracted_frames = response["frames"]["textae"]
    return _extracted_frames
  except requests.RequestException:
    print("Network error...")
    return None


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
      print(f"Record stored successfully: {document_id}\n")
      return  # Exit the function on success
    except GoogleCloudError as e:
      print(f"Error storing record: {e}. Retrying... ({retries + 1}/{max_retries})\n")
      time.sleep(wait_time)  # Wait before retrying
      retries += 1

  print("Failed to store record after multiple retries.")


# Initialize counter
sentences_counter = 53998  # the las sentence id in the database
print()

for idx, _id in enumerate(all_ids):
  text_doc = text_col_ref.document(_id).get()
  rec = text_doc.to_dict()

  if "text_split" not in rec:
    text = rec["text"]
    # Split text into sentences
    text = [preprocess_sentence_cached(sent.text) for sent in nlp_trf(text).sents]
    # Remove empty sentences
    text = [sentence for sentence in text if any(token.is_alpha for token in nlp_trf(sentence))]
    # Join sentences into a single string
    text = ' '.join(text)
    # Initialize the spaCy pipeline with new text
    doc_text = nlp_trf(text)
    # Split text into sentences
    text_split = list(doc_text.sents)
    # Convert sentences into a list of strings
    text_split = [sentence.text for sentence in text_split]
    rec["text_split"] = text_split

  """if "text_split" not in rec or "https://transcripts.cnn.com" in rec["url"]:
    print("#####################################################")
    print(f"Skipping {_id}...")
    print("#####################################################")
    continue"""

  text = rec["text_split"]

  # Preprocess each paragraph
  sentences = [preprocess_sentence_cached(sent) for sent in text]

  print("#####################################################")
  print(f"({idx + 1}) {rec['id']}")
  print("#####################################################\n")

  for sent in sentences:

    doc = nlp(sent)

    # check if sentence so short or so complex
    num_clauses = get_num_of_clauses(doc)
    if num_clauses > 4 or num_clauses == 0:
      continue

    # check if sentence length is longer than 30 tokens
    if len(doc) >= 50:
      continue

    # Check if sentence begins with a capital letter
    if not sent[0].isupper():
      continue

    # Check if sentence is a question
    if sent.endswith("?"):
      continue

    # check if sentence has any undesirable leading pattern
    if check_undesirable_leading_patterns(sent):
      continue

    # check if sentence has any political issue
    matches = match_issues(sent)
    matches = list({_match[4] for _match in matches})
    if not matches:
      continue

    # Check if any matched issue is the main topic of the sentence
    issue_is_main_topic = None
    for match in matches:
      found_issue_as_main_topic = check_if_issue_is_main_topic(match, sent)
      if found_issue_as_main_topic:
        issue_is_main_topic = match
        break
    if not issue_is_main_topic:
      continue

    # Extract semantic frames from sentence
    extracted_frames = extract_frames(sent)
    if extracted_frames:

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
              """found_issue_as_argument = check_if_issue_is_in_string(issue_is_main_topic, d["text"])
              found_issue_is_target = found_issue_as_argument and d["role"] == "TARGET"
              found_issue_is_argument = found_issue_as_argument and d["role"] == "ARGUMENT"

              if found_issue_is_target or found_issue_is_argument:"""
              sentences_counter += 1
              sentence_id = id_with_zeros(sentences_counter)

              print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
              print(f"> Sentence:", sent)
              print(f"> Issue:", issue_is_main_topic)
              # print()
              # print(d)
              sentence_record = {
                "id": sentence_id,
                "text": sent,
                "main_frame": frame_name,
                "semantic_frames": extracted_frame_names,
                "frameNet_data": extracted_frames,
                "issue": issue_is_main_topic,
                "source_url": rec["url"],
                "source": rec["id"],
              }
              store_record_in_firestore(sentence_record, sentences_col_ref)
              sentence_is_about_issue = True
              break

            if sentence_is_about_issue:
              break

          if sentence_is_about_issue:
            break

        sentence_is_about_issue = False
