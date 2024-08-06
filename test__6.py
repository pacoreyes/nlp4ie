import spacy
from transformers import BertTokenizer

s1 = "And then on Saturday, the President will deliver the commencement address to the 2016 graduating class of Howard University here in Washington, D.C."
s2 = "As one of the nation's top historically black colleges and universities, Howard University is recognized for its rigorous education and legacy of building lasting bridges of opportunity for young people."

nlp_trf = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenize s1 and s2 with spacy
doc1 = nlp_trf(s1)
doc2 = nlp_trf(s2)

# tokenize s1 and s2 with BERT tokenizer and print using the patterm (token, token index)
tokens1 = tokenizer.tokenize(s1)
tokens2 = tokenizer.tokenize(s2)

# Print the tokenized sentences
s1_tokens = [(token.text.lower(), token.i) for token in doc1]
print(s1_tokens)

s1_tokens = [(token, idx) for idx, token in enumerate(tokens1)]
print(s1_tokens)

print("--------------------------")

s2_tokens = [(token.text, token.i) for token in doc2]
print(s2_tokens)

s2_tokens = [(token, tokenizer.convert_tokens_to_ids(token)) for token in tokens2]
print(s2_tokens)

