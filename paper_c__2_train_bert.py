import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import (confusion_matrix, roc_auc_score, matthews_corrcoef, accuracy_score,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

from lib.utils import load_jsonl_file
# from lib.visualizations import plot_confusion_matrix


# Initialize label map and class names
LABEL_MAP = {"continue": 0, "not_continue": 1}
class_names = list(LABEL_MAP.keys())

# Initialize constants
MAX_LENGTH = 512  # the maximum sequence length that can be processed by the BERT model
SEED = 42

# Hyperparameters
LEARNING_RATE = 1.9862845049132457e-05
BATCH_SIZE = 16
WARMUP_STEPS = 635
NUM_EPOCHS = 3


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


# Set device to CUDA, MPS, or CPU
device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")


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

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(LABEL_MAP),)
#                                                      hidden_dropout_prob=0.2)
# Move model to device
model.to(device)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load datasets
train_set = load_jsonl_file("shared_data/topic_continuity_train.jsonl")
val_set = load_jsonl_file("shared_data/topic_continuity_valid.jsonl")
test_set = load_jsonl_file("shared_data/topic_continuity_test.jsonl")

# Convert to pandas DataFrame for stratified splitting
df_train = pd.DataFrame({
    "id": [entry["id"] for entry in train_set],
    "text": [entry["text"] for entry in train_set],
    "label": [LABEL_MAP[entry["label"]] for entry in train_set],
    # "metadata": [entry["metadata"] for entry in train_set]
})

df_val = pd.DataFrame({
    "id": [entry["id"] for entry in val_set],
    "text": [entry["text"] for entry in val_set],
    "label": [LABEL_MAP[entry["label"]] for entry in val_set],
    # "metadata": [entry["metadata"] for entry in val_set]
})

df_test = pd.DataFrame({
    "id": [entry["id"] for entry in test_set],
    "text": [entry["text"] for entry in test_set],
    "label": [LABEL_MAP[entry["label"]] for entry in test_set],
    # "metadata": [entry["metadata"] for entry in test_set]
})


# Function for handling sentence pairs
def create_dataset(_df):
  _texts = _df['text'].tolist()  # Assuming this contains sentence pairs
  _labels = _df['label'].tolist()
  _ids = _df['id'].tolist()  # Keep ids as a list of strings

  # Tokenize and preprocess texts for sentence pairs
  _input_ids, _attention_masks = preprocess_pairs(_texts, tokenizer, device, max_length=MAX_LENGTH)

  # Create TensorDataset without ids, since they are strings
  return TensorDataset(_input_ids, _attention_masks, torch.tensor(_labels)), _ids


# Function for handling sentence pairs
def preprocess_pairs(_texts, _tokenizer, _device, max_length=MAX_LENGTH):
  """Tokenize and preprocess text pairs."""
  input_ids = []
  attention_masks = []

  for text in _texts:
    # Split the text into two sentences using a delimiter
    sentence1, sentence2 = text.split('[SEP]')
    encoded_input = _tokenizer.encode(
      sentence1.strip(),
      sentence2.strip(),
      add_special_tokens=True,
      max_length=max_length,
      truncation=True,
      return_tensors='pt'  # Ensure output is in tensor format
    )

    # Pad the encoded_input to max_length
    padded_input = torch.full((1, max_length), _tokenizer.pad_token_id)
    padded_input[:, :encoded_input.size(1)] = encoded_input

    # Create an attention mask for the non-padded elements
    attention_mask = (padded_input != _tokenizer.pad_token_id).int()

    input_ids.append(padded_input)
    attention_masks.append(attention_mask)

  # Concatenate all input_ids and attention_masks
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)

  return input_ids.to(_device), attention_masks.to(_device)


# Create TensorDatasets
train_dataset, train_ids = create_dataset(df_train)
val_dataset, val_ids = create_dataset(df_val)
test_dataset, test_ids = create_dataset(df_test)

# Calculate class weights
labels = df_train["label"].tolist()
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

# Optimizer and Scheduler
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                              num_training_steps=len(train_dataloader) * NUM_EPOCHS)

# Initialize the gradient scaler only if the device is a GPU
use_cuda = device.type == "cuda"
grad_scaler = None
if use_cuda:
  grad_scaler = GradScaler()

# Initialize lists to store losses
train_losses = []
val_losses = []
predictions, true_labels = [], []

print("\nHyperparameters:")
print(f"- Learning Rate: {LEARNING_RATE}")
print(f"- Batch Size: {BATCH_SIZE}")
print(f"- Warmup Steps: {WARMUP_STEPS}")
print(f"- Number of Epochs: {NUM_EPOCHS}")
print("---")
print(f"- Seed: {SEED}")

# epoch, step = 0, 0

# Training Loop
for epoch in range(NUM_EPOCHS):
  model.train()
  total_train_loss = 0
  loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
  print()

  for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{NUM_EPOCHS}")):
    b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

    # Clear gradients
    model.zero_grad()

    # Forward pass: get model outputs
    outputs = model(b_input_ids, attention_mask=b_attention_mask)

    # Calculate loss manually using the logits and labels
    loss = loss_fct(outputs.logits.view(-1, len(LABEL_MAP)), b_labels.view(-1))

    # Perform a backward pass to calculate gradients
    if use_cuda:
      with autocast():
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
      loss.backward()
      optimizer.step()

    scheduler.step()
    total_train_loss += loss.item()

  # Calculate average training loss for this epoch
  avg_train_loss = total_train_loss / len(train_dataloader)
  train_losses.append(avg_train_loss)

  # Validation Loop
  model.eval()
  total_val_loss = 0

  print()
  for batch in tqdm(val_dataloader, desc="Validating..."):
    with torch.no_grad():
      b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

      # Forward pass
      outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
      loss = loss_fct(outputs.logits.view(-1, len(LABEL_MAP)), b_labels.view(-1))
      total_val_loss += loss.item()

      logits = outputs.logits.detach().cpu().numpy()
      label_ids = b_labels.to("cpu").numpy()

      # Store predictions and true labels
      predictions.extend(np.argmax(logits, axis=1).flatten())
      true_labels.extend(label_ids.flatten())

  # Calculate average validation loss
  avg_val_loss = total_val_loss / len(val_dataloader)
  val_losses.append(avg_val_loss)

  # Print average losses for this epoch
  print(f"* Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

""" END of training /validation loop ------------------- """

# Evaluate and log metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions)

print("\nAccuracy:", accuracy)
print("Training Class-wise metrics:")
for i, class_name in enumerate(class_names):
  print(f"{class_name}: Precision = {precision[i]:.2f}, Recall = {recall[i]:.2f}, F1 = {f1[i]:.2f}")

# Test Loop
model.eval()
total_test_loss = 0
test_predictions, test_true_labels = [], []
all_probabilities = []

softmax = torch.nn.Softmax(dim=1)

for i, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
  with torch.no_grad():
    b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

    # Forward pass
    outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
    loss = outputs.loss
    total_test_loss += loss.item()

    logits = outputs.logits
    probabilities = softmax(logits)
    all_probabilities.append(probabilities.cpu().numpy())  # Move to CPU before conversion

    label_ids = b_labels.cpu().numpy()  # Move to CPU before conversion

    # Store predictions and true labels
    test_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())  # Move to CPU before conversion
    test_true_labels.extend(label_ids)

"""plt.figure()
plot_confusion_matrix(test_true_labels,
                      test_predictions,
                      class_names,
                      "paper_c_bert_confusion_matrix.png",
                      "Confusion Matrix for BERT Model",
                      values_fontsize=22
                      )"""

# Concatenate all probabilities
all_probabilities = np.concatenate(all_probabilities, axis=0)

# Calculate average test loss
avg_test_loss = total_test_loss / len(test_dataloader)
print(f"Average test loss: {avg_test_loss}")

# Convert logits to probabilities
probabilities = softmax(torch.Tensor(logits))

# Evaluate metrics for the test set
test_accuracy = accuracy_score(test_true_labels, test_predictions)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
  test_true_labels, test_predictions, average=None, zero_division=0)

# Get AUC ROC score
roc_auc = roc_auc_score(test_true_labels, all_probabilities[:, 1])

# Get MCC score
mcc = matthews_corrcoef(test_true_labels, test_predictions)
# Make confusion matrix
cm = confusion_matrix(test_true_labels, test_predictions)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

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

# print("Test Class-wise metrics:")
for i, class_name in enumerate(class_names):
  print(f"{class_name}: Precision = {precision[i]:.3f}, Recall = {recall[i]:.3f}, F1 = {f1[i]:.3f}")

print()
print("Hyperparameters:")
print(f"- Learning Rate: {LEARNING_RATE}")
print(f"- Batch Size: {BATCH_SIZE}")
print(f"- Warmup Steps: {WARMUP_STEPS}")
print(f"- Number of Epochs: {NUM_EPOCHS}")
# print(f"- Weight Decay: {WEIGHT_DECAY}")
# print(f"- Dropout Rate: {DROP_OUT_RATE}")
print("---")
print(f"- Seed: {SEED}")
print()

# Make visualization for training and validation losses
plt.figure()
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training Loss", color="green")
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", color="black")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("images/paper_c_bert_losses_final_28_06.png")
plt.close()

# Save the model in the 'models' directory
torch.save(model.state_dict(), 'models/11/paper_c_boundary_detection_final_28_06.pth')


"""
THIS MODEL WENT TO THE PAPER

Model: BERT

- Accuracy: 0.855
- Precision: 0.855
- Recall: 0.855
- F1 Score: 0.855
- AUC-ROC: 0.919
- Matthews Correlation Coefficient (MCC): 0.711
- Confusion Matrix:
              continue  not_continue
continue           122            23
not_continue        19           126

continue: Precision = 0.684, Recall = 0.782, F1 = 0.730
not_continue: Precision = 0.746, Recall = 0.639, F1 = 0.688

Hyperparameters:
- Learning Rate: 1.9862845049132457e-05
- Batch Size: 16
- Warmup Steps: 635
- Number of Epochs: 3
---
- Seed: 42
"""
