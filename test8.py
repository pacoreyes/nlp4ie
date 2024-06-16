import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_md')

sentences = [
  "I am opposed to abortion and the death penalty and I have lived my life that way.",
  "I said I am opposed to discrimination.",
  "I have been opposed to a poll tax all my life."
]

matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
terms = ["opposed"]
patterns = [nlp(text) for text in terms]
matcher.add("TerminologyList", patterns)
# doc = nlp("I said I am opposed to discrimination.")

for sentence in sentences:
    print()
    print(f"- {sentence}")
    doc = nlp(sentence)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        print(span.text)
