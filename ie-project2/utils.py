import json
import typing

from google.api_core.datetime_helpers import DatetimeWithNanoseconds


# Save data as JSON file
def save_json_file(data, filename):
  """
  Save data as a JSON file.

  Args:
      data: The data to be saved.
      filename (str): The filename of the JSON file.
  """
  with open(filename, 'w') as json_file:
    json.dump(data, json_file, default=str)


# Open the JSON file
def load_json_file(filename):
  """
  Load data from a JSON file.

  Args:
      filename (str): The filename of the JSON file.

  Returns:
      The loaded data.
  """
  with open(filename) as json_file:
    return json.load(json_file)


def save_jsonl_file(data: typing.List[typing.Any], filename: str) -> None:
  """
  Saves data into a JSON Lines file.

  Args:
      data (List[Any]): A list of data items to be written to a JSONL file.
      filename (str): The name of the file to which data will be written.
  """
  with open(filename, 'w') as jsonl_file:
    for item in data:
      jsonl_file.write(json.dumps(item) + "\n")


def load_jsonl_file(filename: str) -> typing.List[typing.Any]:
  """
  Loads data from a JSON Lines file.

  Args:
      filename (str): The name of the file from which data will be loaded.

  Returns:
      List[Any]: A list of data items loaded from the file.
  """
  print(f"Loading data from {filename}...")
  data = []
  with open(filename) as jsonl_file:
    for line in jsonl_file:
      data.append(json.loads(line))
  print(f"Loaded {len(data)} items.")
  return data


def empty_json_file(filename):
  """
  Empty a JSON file.

  Args:
      filename (str): The filename of the JSON file.
  """
  with open(filename, 'w') as json_file:
    json_file.write('')


def save_row_to_jsonl_file(datapoint, file_path):
  """
  Saves a datapoint to a JSONL file.

  Args:
      datapoint (dict): The datapoint to save.
      file_path (str): The path of the file to save to.
  """
  with open(file_path, "a") as f:
    f.write(json.dumps(datapoint) + "\n")


def save_txt_file(data, filename):
  with open(filename, 'w') as f:
    for row in data:
      f.write(f"{row}\n")


def load_txt_file(filename):
  with open(filename, 'r') as f:
    ids = f.read().splitlines()
  return ids


def firestore_timestamp_to_string(timestamp: DatetimeWithNanoseconds) -> str:
  """
  Converts a Firestore timestamp to a string in the format "DD-MM-YYYY".

  Args:
      timestamp (DatetimeWithNanoseconds): The Firestore timestamp to convert.

  Returns:
      str: The timestamp as a string in the format "DD-MM-YYYY".
  """
  return timestamp.strftime('%d-%m-%Y')
