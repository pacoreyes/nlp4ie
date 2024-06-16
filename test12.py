import spacy

from lib.stance_markers_adj import (positive_adj, positive_adj_phrase,
                                    negative_adj, negative_adj_phrase,
                                    certainty_adj, certainty_adj_phrase,
                                    doubt_adj,
                                    emphatic_adj, emphatic_adj_phrase,
                                    hedge_adj,
                                    pro_adj,
                                    con_adj, con_adj_phrase,
                                    )

from lib.stance_markers_adv import (positive_adv,
                                    negative_adv,
                                    certainty_adv, certainty_adv_phrase,
                                    doubt_adv,
                                    emphatic_adv, emphatic_adv_phrase,
                                    hedge_adv, hedge_adv_phrase,
                                    pro_adv, pro_adv_phrase,
                                    con_adv, con_adv_phrase,
                                    )

from lib.stance_markers_verb import (positive_verb,
                                     negative_verb,
                                     certainty_verb, certainty_verb_phrase,
                                     doubt_verb,
                                     emphatic_verb, emphatic_verb_phrase,
                                     hedge_verb, hedge_verb_phrase,
                                     pro_verb, pro_verb_phrase,
                                     con_verb, con_verb_phrase,
                                     )

dictionaries = [
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
]

nlp = spacy.load("en_core_web_trf")

for idx, dictionary in enumerate(dictionaries):
  print(f"\nDictionary: {idx}")

  for word in dictionary:
    doc = nlp(word)

    # evaluate number of tokens in the word
    if len(doc) > 1:
      print(f"{doc.text}")
