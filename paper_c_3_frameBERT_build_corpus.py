import functools

import requests
import spacy

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
dataset3_col_ref = firestore_db.collection("dataset3")

speeches_ids = load_txt_file("shared_data/text_ids_speeches.txt")
interviews_ids = load_txt_file("shared_data/text_ids_interviews.txt")

all_ids = speeches_ids + interviews_ids

# Get column name from tab "column_name" in the spreadsheet
rule_frames = read_from_google_sheet(spreadsheet_4, "stance_frames_rules")

# Initialize IP address and port
# IP = "141.43.202.175"
IP = "localhost"
PORT = "5001"

# Create a list of dicts with the first 2 columns of the dataframe, "name" and "object"
rule_frames = [{k: v.split(",") if k == "object" else v for k, v in frame.items()} for frame in rule_frames]
rule_frame_names = set(frame["name"] for frame in rule_frames)


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
    if doc is None:
      continue

    # print(f"> Sentence", sent)

    extracted_frames = extract_frames(sent)
    if extracted_frames is None:
      continue

    # Extract frame names from extracted frames
    extracted_frame_names = {frame["frame"] for frame in extracted_frames}

    # Find common frame names
    matched_frame_names = rule_frame_names.intersection(extracted_frame_names)
    # Extract denotations
    denotations = [frame["denotations"] for frame in extracted_frames]

    if matched_frame_names:
      for frame in extracted_frames:
        frame_name = frame["frame"]
        if frame_name in matched_frame_names:
          frame_objects = next((f["object"] for f in rule_frames if f["name"] == frame_name), None)

          for denotation in denotations:
            for d in denotation:

              if d["obj"] in frame_objects:
                # check if any match is in the denotation span text (denotation["text"])

                sentences_counter += 1
                sentence_id = id_with_zeros(sentences_counter)

                print(f"> Sentence {sentence_id}:", sent)
                # print("  Matches:", matches)
                print(f"  frame: {frame_name} / match: {matches}")
                print()

                dataset3_col_ref.document(sentence_id).set({
                  "id": sentence_id,
                  "text": sent,
                  "main_frame": frame_name,
                  "matched_frames": matched_frame_names,
                  "semantic_frames": extracted_frame_names,
                  "frameNet_data": extracted_frames,
                  "issues": matches,
                  "source_url": rec["url"],
                  "source": rec["id"],
                })

                break
          break

"""

                text_doc = nlp(d["text"])
                for token in text_doc:

                    if token.text == match:
                      sentences_counter += 1
                      sentence_id = id_with_zeros(sentences_counter)

                      print(f"> Sentence {sentence_id}:", sent)
                      # print("  Matches:", matches)
                      print(f"  frame: {frame_name} / match: {match}")
                      print()

                      dataset3_col_ref.document(sentence_id).set({
                        "id": sentence_id,
                        "text": sent,
                        "frame": frame_name,
                        "frameNetData": extracted_frames,
                        "issues": matches,
                        "url": rec["url"],
                        "source": rec["id"],
                      })
                      break
"""
