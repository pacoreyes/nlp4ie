import spacy

from lib.stance_markers_adj import (
  positive_adj, positive_adj_phrase,
  negative_adj, negative_adj_phrase,
  certainty_adj, certainty_adj_phrase,
  doubt_adj,
  emphatic_adj, emphatic_adj_phrase,
  hedge_adj,
  pro_adj,
  con_adj, con_adj_phrase
)

# Import stance markers for adverbs
from lib.stance_markers_adv import (
  positive_adv,
  negative_adv,
  certainty_adv, certainty_adv_phrase,
  doubt_adv,
  emphatic_adv, emphatic_adv_phrase,
  hedge_adv, hedge_adv_phrase,
  pro_adv, pro_adv_phrase,
  con_adv, con_adv_phrase
)

# Import stance markers for verbs
from lib.stance_markers_verb import (
  positive_verb,
  negative_verb,
  certainty_verb, certainty_verb_phrase,
  doubt_verb,
  emphatic_verb, emphatic_verb_phrase,
  hedge_verb, hedge_verb_phrase,
  pro_verb, pro_verb_phrase,
  con_verb, con_verb_phrase
)

# Import stance markers for modality
from lib.stance_markers_modals import (
  predictive_modal,
  possibility_modal,
  necessity_modal,
  predictive_modal_phrase,
  necessity_modal_phrase
)

from db import spreadsheet_4
from lib.utils import read_from_google_sheet

all_data = [
  {"name": "Positive Adj", "pos": "ADJ", "data": positive_adj},
  {"name": "Negative Adj", "pos": "ADJ", "data": negative_adj},
  {"name": "Certainty Adj", "pos": "ADJ", "data": certainty_adj},
  {"name": "Doubt Adj", "pos": "ADJ", "data": doubt_adj},
  {"name": "Emphatic Adj", "pos": "ADJ", "data": emphatic_adj},
  {"name": "Hedge Adj", "pos": "ADJ", "data": hedge_adj},
  {"name": "Pro Adj", "pos": "ADJ", "data": pro_adj},
  {"name": "Con Adj", "pos": "ADJ", "data": con_adj},
  {"name": "Positive Adv", "pos": "ADV", "data": positive_adv},
  {"name": "Negative Adv", "pos": "ADV", "data": negative_adv},
  {"name": "Certainty Adv", "pos": "ADV", "data": certainty_adv},
  {"name": "Doubt Asv", "pos": "ADV", "data": doubt_adv},
  {"name": "Emphatic Adv", "pos": "ADV", "data": emphatic_adv},
  {"name": "Hedge Adv", "pos": "ADV", "data": hedge_adv},
  {"name": "Pro Adv", "pos": "ADV", "data": pro_adv},
  {"name": "Con Adv", "pos": "ADV", "data": con_adv},
  {"name": "Positive Verb", "pos": "VERB", "data": positive_verb},
  {"name": "Negative Verb", "pos": "VERB", "data": negative_verb},
  {"name": "Certainty Verb", "pos": "VERB", "data": certainty_verb},
  {"name": "Doubt Verb", "pos": "VERB", "data": doubt_verb},
  {"name": "Emphatic Verb", "pos": "VERB", "data": emphatic_verb},
  {"name": "Hedge Verb", "pos": "VERB", "data": hedge_verb},
  {"name": "Pro Verb", "pos": "VERB", "data": pro_verb},
  {"name": "Con Verb", "pos": "VERB", "data": con_verb},
  {"name": "Predictive Modal", "pos": "MD", "data": predictive_modal},
  {"name": "Possibility Modal", "pos": "MD", "data": possibility_modal},
  {"name": "Necessity Modal", "pos": "MD", "data": necessity_modal}
]

# Set spaCy to use GPU if available
if spacy.prefer_gpu():
    print("spaCy is using GPU!")
else:
    print("GPU not available, spaCy is using CPU instead.")

dataset = read_from_google_sheet(spreadsheet_4, "SHAP")


