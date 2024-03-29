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
from lib.visualizations import plot_confusion_matrix


# Initialize label map and class names
LABEL_MAP = {"monologic": 0, "dialogic": 1}
class_names = list(LABEL_MAP.keys())

# Initialize constants
MAX_LENGTH = 512  # the maximum sequence length that can be processed by the BERT model
SEED = 1234  # 42, 1234, 2021

# Hyperparameters
LEARNING_RATE = 1.6e-5  # 1.5e-5, 2e-5, 3e-5, 5e-5
BATCH_SIZE = 16  # 16, 32
WARMUP_STEPS = 700  # 0, 100, 1000, 10000
NUM_EPOCHS = 4  # 2, 3, 4, 5
WEIGHT_DECAY = 1e-3  # 1e-2 or 1e-3
DROP_OUT_RATE = 0.2  # 0.1 or 0.2


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
                                                      num_labels=len(LABEL_MAP),
                                                      hidden_dropout_prob=DROP_OUT_RATE)
# Move model to device
model.to(device)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load datasets
# train_set = load_jsonl_file("shared_data/dataset_1_6_1b_train.jsonl")
# val_set = load_jsonl_file("shared_data/dataset_1_6_1b_validation.jsonl")
# test_set = load_jsonl_file("shared_data/dataset_1_6_1b_test.jsonl")

train_set = load_jsonl_file("shared_data/dataset_1_6_1b_train_anonym.jsonl")
val_set = load_jsonl_file("shared_data/dataset_1_6_1b_validation_anonym.jsonl")
test_set = load_jsonl_file("shared_data/dataset_1_6_1b_test_anonym.jsonl")

# Convert to pandas DataFrame for stratified splitting
df_train = pd.DataFrame({
    "id": [entry["id"] for entry in train_set],
    "text": [entry["text"] for entry in train_set],
    "label": [LABEL_MAP[entry["label"]] for entry in train_set],
    "metadata": [entry["metadata"] for entry in train_set]
})

df_val = pd.DataFrame({
    "id": [entry["id"] for entry in val_set],
    "text": [entry["text"] for entry in val_set],
    "label": [LABEL_MAP[entry["label"]] for entry in val_set],
    "metadata": [entry["metadata"] for entry in val_set]
})

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
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=WARMUP_STEPS,
                                            num_training_steps=total_steps)

# Initialize the gradient scaler only if the device is a GPU
use_cuda = device.type == "cuda"
grad_scaler = None
if use_cuda:
  grad_scaler = GradScaler()

train_losses = []
val_losses = []

predictions, true_labels = [], []

# Training Loop
for epoch in range(NUM_EPOCHS):
  model.train()
  total_train_loss = 0
  loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

  print()
  for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{NUM_EPOCHS}"):
    b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]
    # Clear gradients
    model.zero_grad()

    # Forward and backward passes
    if use_cuda:
      with autocast():
        outputs = model(b_input_ids, attention_mask=b_attention_mask)
        loss = loss_fct(outputs.logits.view(-1, len(LABEL_MAP)), b_labels.view(-1))
      grad_scaler.scale(loss).backward()
      grad_scaler.step(optimizer)
      grad_scaler.update()
    else:
      outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
      loss = outputs.loss
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

  for batch in tqdm(val_dataloader, desc="Validating"):
    with torch.no_grad():
      b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

      # Forward pass
      outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
      loss = outputs.loss
      total_val_loss += loss.item()

      logits = outputs.logits
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to("cpu").numpy()

      # Store predictions and true labels
      predictions.extend(np.argmax(logits, axis=1).flatten())
      true_labels.extend(label_ids.flatten())

  # Calculate average validation loss
  avg_val_loss = total_val_loss / len(val_dataloader)
  val_losses.append(avg_val_loss)

  # Update the learning rate
  # scheduler.step()

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

for batch in tqdm(test_dataloader, desc="Testing"):
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

plt.figure()
plot_confusion_matrix(test_true_labels,
                      test_predictions,
                      class_names,
                      "paper_a_1_bert_confusion_matrix_anonym.png",
                      "Confusion Matrix for BERT Model",
                      values_fontsize=22
                      )

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
  test_true_labels, test_predictions, zero_division=0)

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
print(f"- Weight Decay: {WEIGHT_DECAY}")
print(f"- Dropout Rate: {DROP_OUT_RATE}")
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
plt.savefig("images/paper_a_1_bert_losses_anonym.png")
plt.close()

# Save the model in the 'models' directory
# torch.save(model.state_dict(), 'models/1/paper_a_bert_solo.pth')
torch.save(model.state_dict(), 'models/1/paper_a_bert_solo_anonym.pth')


"""
Model: BERT

- Accuracy: 0.981
- Precision: 0.981
- Recall: 0.981
- F1 Score: 0.981
- AUC-ROC: 0.993
- Matthews Correlation Coefficient (MCC): 0.962
- Confusion Matrix:
           monologic  dialogic
monologic        231         3
dialogic           6       228

monologic: Precision = 0.95, Recall = 0.93, F1 = 0.94
dialogic: Precision = 0.93, Recall = 0.96, F1 = 0.94

Hyperparameters:
- Learning Rate: 1.6e-05
- Batch Size: 16
- Warmup Steps: 700
- Number of Epochs: 4
- Weight Decay: 0.001
- Dropout Rate: 0.2
---
- Seed: 1234

---------------------------------------------------------------

Model: BERT

- Accuracy: 0.974
- Precision: 0.974
- Recall: 0.974
- F1 Score: 0.974
- AUC-ROC: 0.995
- Matthews Correlation Coefficient (MCC): 0.949
- Confusion Matrix:
           monologic  dialogic
monologic        227         7
dialogic           5       229

monologic: Precision = 0.97, Recall = 0.91, F1 = 0.94
dialogic: Precision = 0.91, Recall = 0.97, F1 = 0.94

Hyperparameters:
- Learning Rate: 1.6e-05
- Batch Size: 16
- Warmup Steps: 700
- Number of Epochs: 4
- Weight Decay: 0.001
- Dropout Rate: 0.2
---
- Seed: 1234
"""
