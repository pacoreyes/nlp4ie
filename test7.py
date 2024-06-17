from lib.semantic_frames import frame_names

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

"""all_data = [
  positive_adj, positive_adj_phrase,
  negative_adj, negative_adj_phrase,
  certainty_adj, certainty_adj_phrase,
  doubt_adj,
  emphatic_adj, emphatic_adj_phrase,
  hedge_adj,
  pro_adj,
  con_adj, con_adj_phrase,
  positive_adv,
  negative_adv,
  certainty_adv, certainty_adv_phrase,
  doubt_adv,
  emphatic_adv, emphatic_adv_phrase,
  hedge_adv, hedge_adv_phrase,
  pro_adv, pro_adv_phrase,
  con_adv, con_adv_phrase,
  positive_verb,
  negative_verb,
  certainty_verb, certainty_verb_phrase,
  doubt_verb,
  emphatic_verb, emphatic_verb_phrase,
  hedge_verb, hedge_verb_phrase,
  pro_verb, pro_verb_phrase,
  con_verb, con_verb_phrase,
  predictive_modal,
  possibility_modal,
  necessity_modal,
  predictive_modal_phrase,
  necessity_modal_phrase
]"""

all_data = [frame_names]

for items in all_data:
  html = "<p>"

  # create a new list ordered alphabetically
  items = list(set(items))

  items.sort()

  for item in items:
    html += "<em>" + item + "</em>, "

  html = html + "</p>"

  print(html)
  print()
