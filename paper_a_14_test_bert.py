import random

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score, \
  matthews_corrcoef
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

from lib.utils import load_jsonl_file, empty_json_file, save_row_to_jsonl_file
from lib.visualizations import plot_confusion_matrix


# Initialize constants
SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 16


# Initialize label map and class names
LABEL_MAP = {"monologic": 0, "dialogic": 1}
class_names = list(LABEL_MAP.keys())
REVERSED_LABEL_MAP = {0: "monologic", 1: "dialogic"}


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")
  # return torch.device("cpu")


def set_seed(seed_value):
  """Set seed for reproducibility."""
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


# Set seed for reproducibility
set_seed(SEED)

# Initialize BERT model
BERT_MODEL = 'bert-base-uncased'

# Assuming you've saved your best model to a path during training
MODEL_PATH = "models/1/paper_a_bert_solo_anonym.pth"
# MODEL_PATH = "models/1/paper_a_bert_solo_anonym.pth"

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# Load Pre-trained Model
model = BertForSequenceClassification.from_pretrained(BERT_MODEL)

# Move Model to Device
device = get_device()
model.to(device)

# Load Saved Weights
print("• Loading Saved Weights...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Load test set
test_set = load_jsonl_file("shared_data/dataset_1_6_1b_test_anonym.jsonl")

df_test = pd.DataFrame({
    "id": [entry["id"] for entry in test_set],
    "text": [entry["text"] for entry in test_set],
    "label": [LABEL_MAP[entry["label"]] for entry in test_set],
    "metadata": [entry["metadata"] for entry in test_set]
})


def create_dataset(_df):
  _texts = _df['text'].tolist()
  _labels = _df['label'].tolist()
  _ids = _df['id'].tolist()  # keep ids as a list of strings
  # Tokenize and preprocess texts
  _input_ids, _attention_masks = preprocess(_texts, tokenizer, device, max_length=MAX_LENGTH)
  # Create TensorDataset without ids, since they are strings
  return TensorDataset(_input_ids, _attention_masks, torch.tensor(_labels)), _ids


def preprocess(_texts, _tokenizer, _device, max_length=MAX_LENGTH):
  """Tokenize and preprocess texts."""
  inputs = _tokenizer(_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
  return inputs["input_ids"].to(_device), inputs["attention_mask"].to(_device)


# Create test dataset
test_dataset, test_ids = create_dataset(df_test)

# DataLoader for the test dataset
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model.eval()
total_test_loss = 0
test_predictions, test_true_labels = [], []
all_probabilities = []

# Create a softmax layer to convert logits to probabilities
softmax = torch.nn.Softmax(dim=1)

# Initialize JSONL file for misclassified examples
misclassified_output_file = "shared_data/dataset_1_8_2b_misclassified_examples_anonym.jsonl"
empty_json_file(misclassified_output_file)

for i, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
  with torch.no_grad():
    # Unpack batch data
    b_input_ids, b_attention_mask, b_labels = batch
    b_input_ids, b_attention_mask, b_labels = b_input_ids.to(device), b_attention_mask.to(device), b_labels.to(device)

    # Forward pass
    outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
    loss = outputs.loss
    total_test_loss += loss.item()

    # Get probabilities and predictions
    logits = outputs.logits
    probabilities = softmax(logits)
    all_probabilities.append(probabilities.cpu().numpy())  # Move to CPU before conversion
    label_ids = b_labels.cpu().numpy()  # Move to CPU before conversion

    # Store predictions and true labels
    batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()  # Move to CPU before conversion
    test_predictions.extend(batch_predictions)
    test_true_labels.extend(label_ids)

    for j, (pred, true) in enumerate(zip(batch_predictions, label_ids)):
      if pred != true:
        example_id = test_ids[i * BATCH_SIZE + j]
        save_row_to_jsonl_file({
          "id": example_id,  # corrected to use the separate ids list
          "true_label": REVERSED_LABEL_MAP[true],
          "predicted_label": REVERSED_LABEL_MAP[pred],
          # "text": df_test.to_dict()[i * batch_size + j]["text"],
          # "metadata": df_test.to_dict()[i * batch_size + j]["metadata"]
        }, misclassified_output_file)

# Concatenate all probabilities
all_probabilities = np.concatenate(all_probabilities, axis=0)

# Calculate and print metrics
test_accuracy, test_precision, test_recall, test_f1 = None, None, None, None
if test_predictions and test_true_labels:
  test_accuracy = accuracy_score(test_true_labels, test_predictions)
  test_precision, test_recall, test_f1, _ = (
    precision_recall_fscore_support(test_true_labels, test_predictions, average=None, zero_division=0))

  print("\nMetrics for testing:")
  print(f"Accuracy: {test_accuracy}")

# Get AUC ROC score
roc_auc = roc_auc_score(test_true_labels, all_probabilities[:, 1])
# Get MCC score
mcc = matthews_corrcoef(test_true_labels, test_predictions)
# Calculate confusion matrix
cm = confusion_matrix(test_true_labels, test_predictions)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

# Plot confusion matrix and save it as PNG
plot_confusion_matrix(test_true_labels,
                      test_predictions,
                      class_names,
                      "paper_a_1_bert_confusion_matrix_anonym.png",
                      "Confusion Matrix for BERT Model",
                      values_fontsize=22
                      )

print("\nModel: BERT\n")
print(f"- Accuracy: {test_accuracy:.3f}")
print(f"- Precision: {np.mean(test_precision):.3f}")
print(f"- Recall: {np.mean(test_recall):.3f}")
print(f"- F1 Score: {np.mean(test_f1):.3f}")
print(f"- AUC-ROC: {roc_auc:.3f}")
print(f"- Matthews Correlation Coefficient (MCC): {mcc:.3f}")
print(f"- Confusion Matrix:")
print(df_cm)
print()

print("Class-wise metrics:")
for i, class_name in enumerate(class_names):
  print(f"{class_name}: Precision = {test_precision[i]:.3f}, Recall = {test_recall[i]:.3f}, F1 = {test_f1[i]:.3f}")
