import spacy

nlp = spacy.load("en_core_web_lg")


def check_adverb_start(doc):
    adverbs = {"clearly", "plainly", "surely"}

    # Check if the sentence starts with an adverb
    if doc[0].text.lower() in adverbs:
        return True

    # Check for subordinate clauses that start with an adverb
    for token in doc:
        # Find the beginning of a clause
        if token.dep_ == "mark":
            # Check if the clause starts with an adverb
            if token.head.text.lower() in adverbs:
                return True

    return False


# Example usage:
text = "He said that, surely, he would come."
doc = nlp(text)

print(check_adverb_start(doc))
