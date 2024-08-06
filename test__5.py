import spacy

from paper_c__4_inference_bert import inference_pair
from db import spreadsheet_7
from lib.text_utils import preprocess_text
from lib.utils import read_from_google_sheet, write_to_google_sheet, save_jsonl_file

nlp_trf = spacy.load("en_core_web_trf")

dataset = read_from_google_sheet(spreadsheet_7, "test_dataset")
dataset = dataset[177:178]

example = dataset[0]

"""sentences = example["text"].split(" [SEP] ")
sent1 = sentences[0].strip()
sent2 = sentences[1].strip()

sent1 = preprocess_text(sent1, nlp_trf,
                        with_remove_known_unuseful_strings=False,
                        with_remove_parentheses_and_brackets=False,
                        with_remove_text_inside_parentheses=False,
                        with_remove_leading_patterns=False,
                        with_remove_timestamps=False,
                        with_replace_unicode_characters=False,
                        with_expand_contractions=True,
                        with_remove_links_from_text=False,
                        with_put_placeholders=False,
                        with_final_cleanup=False)

sent2 = preprocess_text(sent2, nlp_trf,
                        with_remove_known_unuseful_strings=False,
                        with_remove_parentheses_and_brackets=False,
                        with_remove_text_inside_parentheses=False,
                        with_remove_leading_patterns=False,
                        with_remove_timestamps=False,
                        with_replace_unicode_characters=False,
                        with_expand_contractions=True,
                        with_remove_links_from_text=False,
                        with_put_placeholders=False,
                        with_final_cleanup=False)

# Combine the two sentences into one with contractions expanded
processed_sentence_pair = f"{sent1} [SEP] {sent2}"""

example["label_bert"] = inference_pair([example["text"]])

print(example["text"])
print(example["label_bert"])
