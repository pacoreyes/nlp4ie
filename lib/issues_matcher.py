from pprint import pprint

import spacy
from spacy.language import Language
from spacy.tokens import Span
from spacy.matcher import Matcher, PhraseMatcher
import pandas as pd

from db import spreadsheet_1
from lib.utils import read_from_google_sheet


@Language.component("custom_ner")
def custom_ner(_doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
  ner_label = "ISSUE"

  # Clone the doc by generating a new Doc object using the NLP for tagger
  doc_tagger = nlp_tagger(_doc.text)

  # 1. Match hyphenated terms
  matches = matcher(doc_tagger)
  spans = []
  for _match_id, start, end in matches:
    span = _doc[start:end]
    # Store the lemma of the matched term in the Span object
    span._.lemmatized = nlp.vocab.strings[_match_id]
    spans.append(span)

  # 2. Match lemmatized terms
  phrase_matches = phrase_matcher_lemma(_doc)
  phrase_spans = []
  for _match_id, start, end in phrase_matches:
    phrase_span = _doc[start:end]
    # Store the lemma of the matched term in the Span object
    phrase_span._.lemmatized = nlp.vocab.strings[_match_id]
    phrase_spans.append(phrase_span)

  # 3. Match exact terms
  phrase_matches = phrase_matcher_orth(_doc)
  phrase_spans += [_doc[start:end] for _, start, end in phrase_matches]

  # Combine matches (spans) from all matchers
  combined_spans = spans + phrase_spans
  # Remove overlapping matches
  filtered_spans = spacy.util.filter_spans(combined_spans)

  # Get Wikidata info for each entity
  for spans in filtered_spans:
    matched_entities = []
    for i in pol_issues:
      if spans._.lemmatized:
        if spans._.lemmatized in [i["pp_name"]] + str(i.get("pp_aliases", "")).split(", "):
          matched_entities.append(i)
      else:
        if spans.text in [i["pp_name"]] + str(i.get("pp_aliases", "")).split(", "):
          matched_entities.append(i)

    # Ensure unique entities using a set
    matched_qids = list(set(e["qid"] for e in matched_entities))
    matched_names = list(set(e["name"] for e in matched_entities))

    if matched_entities:
      entity = Span(_doc, spans.start, spans.end, label=ner_label)
      entity._.qid = matched_qids
      entity._.wd_name = matched_names
      _doc.set_ents([entity], default="unmodified")
  return _doc


""" 1. Initialize spaCy models and custom attributes """
# Add custom attribute to the Span object
Span.set_extension("qid", default=None, force=True)
Span.set_extension("wd_name", default=None, force=True)
Span.set_extension("lemmatized", default=None, force=True)

# Load spaCy model with tagger disabled
nlp = spacy.load("en_core_web_sm", exclude=["ner"], disable=["tagger"])
# Load spaCy model with tagger enabled
nlp_tagger = spacy.load("en_core_web_sm")

""" 2. Load gazetteers and split them into three lists """
# Load gazetteers from Google Sheet
gazetteers = read_from_google_sheet(spreadsheet_1, "pol_issues_split")

# Load gazetteers into a dataframe
df = pd.DataFrame(gazetteers)

# Split gazetteers into three lists
exact_terms = df["exact"].tolist()
hyphenated_terms = df["hyphenated"].tolist()
lemmatized_terms = df["lemmatized"].tolist()

# Remove blank entries (pads) from the lists
exact_terms = [term for term in exact_terms if term]
hyphenated_terms = [term for term in hyphenated_terms if term]
lemmatized_terms = [term for term in lemmatized_terms if term]

""" 3. Load political issues and create and load matchers """
# Read the political issues data from the Google Sheet
pol_issues = read_from_google_sheet(spreadsheet_1, "pol_issues2")

# Create matchers
matcher = Matcher(nlp.vocab)
phrase_matcher_lemma = PhraseMatcher(nlp.vocab, attr="LEMMA")
phrase_matcher_orth = PhraseMatcher(nlp.vocab, attr="ORTH")

# 1. Load exact terms into phrase matcher
patterns = [nlp(text) for text in exact_terms]
phrase_matcher_orth.add("EXACT_MATCH", patterns)

# 2. Load hyphenated terms into matcher
for term in hyphenated_terms:
  doc = nlp(term)
  # Create a pattern based on lemma
  pattern = [{'LEMMA': token.lemma_} for token in doc]
  # Convert the lemmatized term to a hash
  match_id = nlp.vocab.strings.add(term)
  # Add the pattern to the matcher with the lemmatized term as the match_id
  matcher.add(match_id, [pattern])

# 3. Load lemmatized terms into phrase matcher
for term in lemmatized_terms:
  pattern = nlp(term)
  # print("Pattern:", [token.text for token in pattern])  # This should print "war" for the term "war"
  phrase_matcher_lemma.add(term, [pattern])

""" 4. Add custom component to the pipeline """
nlp.add_pipe("custom_ner", last=True)


def match_issues(text):
  _doc = nlp(text)
  return [(ent.start, ent.end, ent._.qid, ent._.wd_name, ent.text, ent.label_) for ent in _doc.ents]
