import urllib.parse
from collections import Counter

from lib.utils import load_jsonl_file

dataset_train = load_jsonl_file("shared_data/dataset_2_train.jsonl")
dataset_validation = load_jsonl_file("shared_data/dataset_2_validation.jsonl")
dataset_test = load_jsonl_file("shared_data/dataset_2_test.jsonl")

dataset = dataset_train + dataset_validation + dataset_test


def get_domain(url):
  # Use urllib to parse the URL and extract the network location
  parsed_url = urllib.parse.urlparse(url)
  domain = parsed_url.netloc

  # Optionally, you could further clean this to remove 'www.', if commonly present
  if domain.startswith('www.'):
    domain = domain[4:]
  return domain


# Counter to hold domain counts
domain_counts = Counter()

# Iterate over each dictionary in the list
for idx, item in enumerate(dataset):
  if 'source' in item["metadata"]:
    url = item['metadata']['source']
    domain = get_domain(url)
    print(f"Item {idx}/{domain} processed.")
    domain_counts[domain] += 1

# Print the domain counts
for domain, count in domain_counts.items():
  print(f"{domain}: {count}")
