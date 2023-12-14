from collections import Counter

import spacy
import tqdm
from transformers import BertTokenizer

from lib.utils import load_jsonl_file, empty_json_file, save_row_to_jsonl_file

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_trf")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Load the JSON file
dataset = load_jsonl_file("shared_data/dataset_1_3.jsonl")

output_file = "shared_data/dataset_1_4_sliced.jsonl"

# Empty the output JSONL file
empty_json_file(output_file)

datapoint_id = 0  # Initialize a counter for the unique sequential id
monologic_counter = 0
dialogic_counter = 0

# Process each text
for idx, text_doc in tqdm.tqdm(enumerate(dataset), desc=f"Processing {len(dataset)} datapoints", total=len(dataset)):
  idx = int(idx)
  # print(f"Processing text {idx + 1}/{dataset}")
  doc = nlp(text_doc['text'])
  sentences = [sent.text for sent in doc.sents]

  # Create windows with overlap
  window = []
  for i in range(len(sentences)):
    # Add the sentence to the current window
    window.append(sentences[i])
    # Check if the total number of tokens in the current window exceeds the BERT token limit
    if len(tokenizer.tokenize(" ".join(window))) > 510:  # Saving 2 tokens for padding
      # If it does, remove the last sentence from the window
      window.pop()
      # Check the number of tokens, skip if less than 300
      if len(tokenizer.tokenize(" ".join(window))) < 300:
        window = [sentences[i]]
        continue
      # Join the sentences in the current window to form a single text, and add it to the new dataset
      new_text = " ".join(window)
      datapoint_id += 1  # Increment the unique id for each new datapoint
      datapoint = {'id': datapoint_id, 'text_id': text_doc["id"], 'text': new_text,
                   'label': text_doc['discourse_type'], 'metadata': text_doc['metadata']}
      save_row_to_jsonl_file(datapoint, output_file)
      if datapoint['label'] == 0:
        monologic_counter += 1
      else:
        dialogic_counter += 1

      # Start a new window with the last sentence of the previous window
      window = [sentences[i]]
  # Add the remaining sentences in the last window to the new dataset
  if window and len(tokenizer.tokenize(" ".join(window))) >= 450:
    new_text = " ".join(window)
    datapoint_id += 1  # Increment the unique id for each new datapoint
    datapoint = {'id': datapoint_id, 'text_id': text_doc["id"], 'text': new_text,
                 'label': text_doc['discourse_type'], 'metadata': text_doc['metadata']}
    save_row_to_jsonl_file(datapoint, output_file)
    if datapoint['label'] == 0:
      monologic_counter += 1
    else:
      dialogic_counter += 1

# Count the number of monologic and dialogic datapoints
counter = Counter([item['label'] for item in load_jsonl_file(output_file)])
monologic_percentage = counter["monologic"] / (counter["monologic"] + counter["dialogic"]) * 100
dialogic_percentage = counter["dialogic"] / (counter["monologic"] + counter["dialogic"]) * 100

print()
print("The original dataset has been split:")
print(f"\n• Monologic: {counter['monologic']} ({monologic_percentage:.2f})")
print(f"• Dialogic: {counter['dialogic']} ({dialogic_percentage:.2f})")
