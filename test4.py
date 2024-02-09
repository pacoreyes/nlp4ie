import spacy
from lib.utils import load_jsonl_file, empty_json_file
from lib.linguistic_utils import check_if_has_one_word_or_more

nlp = spacy.load("en_core_web_trf")

"""dataset = load_jsonl_file("shared_data/dataset_1_3_preprocessed_a.jsonl")

for datapoint in dataset[1:2]:
  text = datapoint["text"]
  sentences = [sent.text for sent in nlp(text).sents]
  sentences = [sent for sent in sentences if check_if_has_one_word_or_more(nlp(sent))]
  # datapoint["text"] = " ".join(sentences)
  for sentence in sentences:
    print(sentence)
"""

sentence = "Thanks for joining us live on BBC and Yahoo.com for this exclusive interview with William."

doc = nlp(sentence)

for token in doc:
  print(token.text, token.pos_)
