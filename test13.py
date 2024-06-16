import spacy
from spacy.matcher import PhraseMatcher, Matcher

# load transformer model
nlp = spacy.load("en_core_web_lg")

sentence = "All public housing should be equipped with up-to-date sprinkler systems, which is why I support my colleagues in introducing this critical legislation to prevent similar tragedies."

doc = nlp(sentence)

terms = [
  "up-to-date"
]

# Lexicon matchers
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA", validate=True)
patterns = [nlp(phrase) for phrase in terms]
phrase_matcher.add("PATTERN_TEST", patterns)

# Find lexicon matches
matches = phrase_matcher(doc)

# Extract indexes of tokens in the sentence, separating single and multi-token matches
for _, start, end in matches:
  print(doc[start:end].text, start, end)
