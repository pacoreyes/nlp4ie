from pprint import pprint

from lib.lexicons import certainty_adj, certainty_adv, certainty_verbs, doubt_adj, doubt_adv, doubt_verbs

# order ascending the list of strings
list(set(doubt_adv)).sort()

pprint(doubt_adv)
