import csv
import json
import spacy

nlp = spacy.load("en_core_web_sm")


def csv_to_jsonl(_csv_file, _jsonl_file):
  """
  Converts a CSV file to a JSON Lines file. The "sentence_id" and "sentence"
  fields in the CSV are renamed to "id" and "text", respectively, in the JSONL output.

  Parameters:
  - csv_filepath: str. The file path for the input CSV file.
  - jsonl_filepath: str. The file path for the output JSONL file.
  """
  with open(csv_filepath, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    with open(jsonl_filepath, mode='w', encoding='utf-8') as jsonl_file:
      for row in csv_reader:
        # use spacy to filter sentences longer than 50 tokens
        if len(nlp(row['sentence'])) > 50:
          continue
        # Create a new dictionary with only the required keys, renamed appropriately
        filtered_row = {'id': row['sentence_id'], 'text': row['sentence'] + '.'}
        # Write the filtered and modified row to the JSONL file
        jsonl_file.write(json.dumps(filtered_row) + '\n')


# Example usage
csv_filepath = 'shared_data/_123/argumentative_sentences_in_spoken_language_with split.csv'
jsonl_filepath = 'shared_data/_123/argumentative_sentences.jsonl'
csv_to_jsonl(csv_filepath, jsonl_filepath)
