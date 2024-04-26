import spacy
from spacy.matcher import PhraseMatcher

# Load the spaCy model
nlp = spacy.load('en_core_web_lg')

# Create the PhraseMatcher object with the vocabulary of the model
# Set attr to 'LEMMA' to match on the lemmatized forms of the words
matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")

# List of lemmatized forms you want to match
lemmas = ["run", "buy", "love", "alarm"]

# Create pattern docs for each lemma
patterns = [nlp(lemma) for lemma in lemmas]

# Add the patterns to the matcher with the rule name 'LEMMA_MATCH'
matcher.add("LEMMA_MATCH", patterns)

# Example text to match against
text = "She loves running and buying things . He ran to the store and bought some milk."

# Process the text
doc = nlp(text)

# Apply the matcher to the processed doc
matches = matcher(doc)

# Print the matches found
print("Matches found:")
for match_id, start, end in matches:
    span = doc[start:end]  # The matched span
    print(span.text, span.start, span.end)