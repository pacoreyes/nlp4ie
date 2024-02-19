import random

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import (confusion_matrix, roc_auc_score, matthews_corrcoef, accuracy_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file, save_jsonl_file
from lib.utils2 import balance_classes_in_dataset
from lib.visualizations import plot_confusion_matrix

# Initialize label map and class names
LABEL_MAP = {"continue": 0, "not_continue": 1}
class_names = list(LABEL_MAP.keys())
REVERSED_LABEL_MAP = {0: "continue", 1: "not_continue"}

# Initialize constants
MAX_LENGTH = 512  # the maximum sequence length that can be processed by the BERT model
SEED = 42  # 42, 1234, 2024
NUM_TRIALS = 20  # Number of trials for hyperparameter optimization
VALIDATION_STEP_INTERVAL = 20  # Interval for validation step for early stopping

print("############################################")
print(f"Number of trials for hyperparameter optimization: {NUM_TRIALS}")
print("############################################")


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


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
                                                      hidden_dropout_prob=0.2)
# Move model to device
model.to(device)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the dataset
dataset = load_jsonl_file("shared_data/dataset_2_5_pair_sentences_reclass.jsonl")
# Create a balanced dataset by undersampling the majority class
dataset = balance_classes_in_dataset(dataset, "continue", "not_continue", "label")
# Save the balanced dataset to a JSONL file
save_jsonl_file(dataset, "shared_data/dataset_2_6_2b.jsonl")

# Convert to pandas DataFrame for stratified splitting
df = pd.DataFrame({
  "id": [entry["id"] for entry in dataset],
  "text": [entry["text"] for entry in dataset],
  "label": [LABEL_MAP[entry["label"]] for entry in dataset],
  # "metadata": [entry["metadata"] for entry in dataset]
})

# Stratified split of the data to obtain the train and the remaining data
df_train, remaining_df = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=SEED)

# Split the remaining data equally to get a validation set and a test set
df_val, df_test = train_test_split(remaining_df, stratify=remaining_df["label"], test_size=0.5,
                                   random_state=SEED)


# Early stopping class for stopping the training when the validation loss does not improve
class EarlyStopping:
  """Early stops the training if validation loss doesn't improve after a given patience."""

  def __init__(self, patience=3, min_delta=0):
    """
    Args:
        patience (int): How many steps to wait after last time validation loss improved.
                        Default: 7
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
    """
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.best_score = None
    self.early_stop = False

  def __call__(self, val_loss):
    if self.best_score is None:
      self.best_score = val_loss
    elif val_loss > self.best_score - self.min_delta:
      self.counter += 1
      print(f'Early stopping counter: {self.counter} out of {self.patience}')
      if self.counter >= self.patience:
        # print('Early stopping')
        self.early_stop = True
    else:
      self.best_score = val_loss
      self.counter = 0


#
"""def create_dataset(_df):
  _texts = _df['text'].tolist()
  _labels = _df['label'].tolist()
  _ids = _df['id'].tolist()  # keep ids as a list of strings
  # Tokenize and preprocess texts
  _input_ids, _attention_masks = preprocess(_texts, tokenizer, device, max_length=MAX_LENGTH)
  # Create TensorDataset without ids, since they are strings
  return TensorDataset(_input_ids, _attention_masks, torch.tensor(_labels)), _ids


# Preprocess datapoints for BERT
def preprocess(_texts, _tokenizer, _device, max_length=MAX_LENGTH):
  # Tokenize and preprocess texts.
  inputs = _tokenizer(_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
  return inputs["input_ids"].to(_device), inputs["attention_mask"].to(_device)"""


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
    # Split the text into two sentences using the [SEP] token
    sentence1, sentence2 = text.split('[SEP]')
    encoded_input = _tokenizer.encode_plus(text=sentence1.strip(),
                                           text_pair=sentence2.strip(),
                                           add_special_tokens=True,
                                           max_length=max_length,
                                           padding='max_length',
                                           truncation=True,
                                           return_attention_mask=True,
                                           return_tensors='pt')

    input_ids.append(encoded_input['input_ids'])
    attention_masks.append(encoded_input['attention_mask'])

  # Convert lists to tensors and reshape to remove extra dimension
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


# Optuna objective function for hyperparameter optimization
def objective(_trial):
  # Initialize early stopping
  early_stopping = EarlyStopping(patience=7, min_delta=0.001)

  learning_rate = _trial.suggest_float("learning_rate", 1e-6, 3e-5, log=True)
  batch_size = _trial.suggest_categorical("batch_size", [16, 16])
  warmup_steps = _trial.suggest_int("warmup_steps", 0, 1000)
  num_epochs = _trial.suggest_int("num_epochs", 1, 3)

  # Create DataLoaders
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
  val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
  test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

  # Initialize optimizer and scheduler
  optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                              num_training_steps=len(train_dataloader) * num_epochs)

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
  print(f"- Learning Rate: {learning_rate}")
  print(f"- Batch Size: {batch_size}")
  print(f"- Warmup Steps: {warmup_steps}")
  print(f"- Number of Epochs: {num_epochs}")
  print("---")
  print(f"- Seed: {SEED}")

  epoch, step = 0, 0

  # Training Loop
  for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    print()

    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
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

      # Check for early stopping validation after specified steps
      if (step + 1) % VALIDATION_STEP_INTERVAL == 0 or (step + 1) == len(train_dataloader):
        # Switch to evaluation mode
        model.eval()
        val_loss = 0
        count = 0

        for val_batch in val_dataloader:
          b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in val_batch]
          with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            batch_loss = outputs[0].item()
            val_loss += batch_loss
            count += 1

        val_loss /= count
        print(f"Training loss: {total_train_loss / (step + 1):.4f} / Validation loss: {val_loss:.4f}")

        # Early Stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
          print("Early stopping triggered...")
          break

    if early_stopping.early_stop:
      break

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

  """print(len(train_losses))
  print(len(val_losses))"""

  # Plot training and validation losses
  plt.figure()
  plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", color="green")
  plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color="black")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training and Validation Losses per Epoch")
  plt.legend()
  plt.savefig("images/paper_b_3_bert_losses.png")
  plt.close()

  """ END of training /validation loop ------------------- """

  if predictions and true_labels:
    # precision, recall, f1, _ = precision_recall_fscore_support(true_labels,
    # predictions, average=None, zero_division=0)
    accuracy = accuracy_score(true_labels, predictions)

    print("\nMetrics for validation:")
    print(f"Accuracy: {accuracy}\n")

  # Test Loop
  model.eval()
  total_test_loss = 0
  test_predictions, test_true_labels = [], []
  all_probabilities = []

  # Create a softmax layer to convert logits to probabilities
  softmax = torch.nn.Softmax(dim=1)

  # Initialize JSONL file for misclassified examples
  misclassified_output_file = "shared_data/dataset_2_8_1b_misclassified_examples_reclass.jsonl"
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
      test_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())  # Move to CPU before conversion
      test_true_labels.extend(label_ids)

      # Identifying and saving misclassified examples
      for j, (pred, true) in enumerate(zip(test_predictions, label_ids)):  # no _id in zip
        if pred != true:
          # Access the correct id using the batch index and the offset within the batch
          example_id = test_ids[i * batch_size + j]
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
    print(f"Accuracy: {test_accuracy}\n")

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
                        "paper_b_2_bert_confusion_matrix.png",
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

  print()
  print("Hyperparameters:")
  print(f"- Learning Rate: {learning_rate}")
  print(f"- Batch Size: {batch_size}")
  print(f"- Warmup Steps: {warmup_steps}")
  print(f"- Number of Epochs: {num_epochs}")
  print("---")
  print(f"- Seed: {SEED}")
  # Report where early stopping was triggered
  if early_stopping.early_stop:
    print("---")
    print("Early stopping triggered at:")
    print(f"- Epoch {epoch + 1}")
    print(f" Step {step + 1}\n")

  # Save the best model only if the current trial achieves the best accuracy
  if test_accuracy > _trial.user_attrs.get("best_test_accuracy", 0):
    _trial.set_user_attr("best_test_accuracy", test_accuracy)
    best_model_path = f"models/2/paper_b_hop_bert_reclass.pth"
    torch.save(model.state_dict(), best_model_path)
    _trial.set_user_attr("best_model_path", best_model_path)

  return test_accuracy


# Create an Optuna study
study = optuna.create_study(direction="maximize")  # or "minimize" depending on your metric

# Optimize the study
study.optimize(objective, n_trials=NUM_TRIALS)  # Adjust the number of trials as needed

# Print best trial results
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
  print(f"    {key}: {value}")

"""
Model: BERT non-anonymized

- Accuracy: 0.857
- Precision: 0.857
- Recall: 0.857
- F1 Score: 0.857
- AUC-ROC: 0.917
- Matthews Correlation Coefficient (MCC): 0.714
- Confusion Matrix:
              continue  not_continue
continue           154            25
not_continue        26           152

Class-wise metrics:
continue: Precision = 0.856, Recall = 0.860, F1 = 0.858
not_continue: Precision = 0.859, Recall = 0.854, F1 = 0.856

Hyperparameters:
- Learning Rate: 1.0490130958581018e-05
- Batch Size: 16
- Warmup Steps: 466
- Number of Epochs: 1
---
- Seed: 2024

"""
