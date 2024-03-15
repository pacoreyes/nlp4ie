from pprint import pprint
from collections import Counter

from tqdm import tqdm
from db import firestore_db
from lib.utils import load_jsonl_file


def extract_core_domain(_url):
  """
  Extracts the core domain from a given URL.

  Parameters:
  - url (str): The URL from which to extract the core domain.

  Returns:
  - str: The core domain extracted from the URL.
  """
  from urllib.parse import urlparse

  # Parse the URL to extract the netloc part which includes the domain and possibly the port
  netloc = urlparse(_url).netloc

  # Split the netloc into parts (subdomain, domain, and TLD)
  parts = netloc.split('.')

  # However, if there is a subdomain like 'co.uk', 'com.au', etc., we need to include that as well
  if len(parts) > 2 and (len(parts[-2]) == 2 or parts[-2] in ['com', 'gov', 'org', 'net', 'edu']):
    _core_domain = '.'.join(parts[-3:])
  else:
    _core_domain = '.'.join(parts[-2:])

  return _core_domain


# Load datasets
train_set = load_jsonl_file("shared_data/dataset_1_6_1b_train.jsonl")
val_set = load_jsonl_file("shared_data/dataset_1_6_1b_validation.jsonl")
test_set = load_jsonl_file("shared_data/dataset_1_6_1b_test.jsonl")

ref_coll_texts = firestore_db.collection("texts2")

dataset = train_set + val_set + test_set
# dataset = dataset[:10]
print(len(dataset))

# Collect IDs of the data points
ids = [datapoint["id"] for datapoint in dataset]

# Initialize counters
source_counter = Counter()
unique_urls = set()

for datapoint in tqdm(dataset, desc=f"Processing {len(dataset)} datapoints"):
  url = datapoint["metadata"]["source"]
  # Check if URL is in the set of unique URLs
  if url not in unique_urls:
    # If not, add it to the set and increment the counter
    unique_urls.add(url)
    core_domain = extract_core_domain(url)
    source_counter[core_domain] += 1

# Totalize all sources and print the total number of sources
total_sources = sum(source_counter.values())

pprint(source_counter)
print(f"\nTotal number of sources: {total_sources}")
