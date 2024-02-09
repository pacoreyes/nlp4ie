from lib.utils import load_jsonl_file, save_jsonl_file, read_from_google_sheet
from lib.utils2 import remove_duplicated_datapoints, remove_examples_in_dataset


dataset1 = load_jsonl_file("shared_data/dataset_3_9_unseen_unlabeled_sentences.jsonl")
dataset2 = load_jsonl_file("shared_data/dataset_3_9_unseen_unlabeled_sentences_2.jsonl")


# Remove duplicated datapoints
dataset1 = remove_duplicated_datapoints(dataset1)
dataset2 = remove_duplicated_datapoints(dataset2)

# Remove examples in dataset3
dataset = remove_examples_in_dataset(dataset2, dataset1)

# Save dataset
print(f"Saving dataset with {len(dataset)} datapoints...")
save_jsonl_file(dataset, "shared_data/dataset_3_9_unseen_unlabeled_sentences_3.jsonl")
