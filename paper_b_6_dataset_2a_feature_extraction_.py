from pprint import pprint
import spacy
from lib.lexicons import certainty_adj, certainty_adv, certainty_verbs, doubt_adj, doubt_adv, doubt_verbs

positive_adjs = [
  'allegedly', 'apparently', 'arguably', 'conceivably', 'ostensibly', 'perchance', 'perhaps', 'possibly',
  'presumably', 'purportedly', 'reportedly', 'reputedly', 'seemingly', 'supposedly', 'formally', 'hypothetically',
  'ideally', 'likely', 'officially', 'outwardly', 'superficially', 'technically', 'theoretically'
]

nlp = spacy.load("en_core_web_trf")

lemmas = [token.lemma_ for adj in positive_adjs for token in nlp(adj)]

# print(lemmas)

# Remove duplicates by converting to a set, then convert back to a list and sort it
lemmas = sorted(set(lemmas))

# Join the lemmas into a string separated by a comma
lemmas_str = ", ".join(lemmas)
print(lemmas_str)
print()
print(lemmas)
