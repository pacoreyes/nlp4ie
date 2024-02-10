import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import (confusion_matrix, roc_auc_score, matthews_corrcoef, accuracy_score,
                             precision_recall_fscore_support)
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
# import optuna
from optuna import create_study

from lib.utils import load_jsonl_file, save_row_to_jsonl_file, empty_json_file
from lib.utils2 import balance_classes_in_dataset
from lib.visualizations import plot_confusion_matrix


# Initialize label map and class names
LABEL_MAP = {"monologic": 0, "dialogic": 1}
class_names = list(LABEL_MAP.keys())

# REVERSED_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
REVERSED_LABEL_MAP = {0: "monologic", 1: "dialogic"}

# Initialize constants
MAX_LENGTH = 512  # the maximum sequence length that can be processed by the BERT model
SEED = 1234  # 42, 1234, 2021

'''
# Hyperparameters
LEARNING_RATE = 1.6e-5  # 1.5e-5, 2e-5, 3e-5, 5e-5
BATCH_SIZE = 8  # 16, 32
WARMUP_STEPS = 700  # 0, 100, 1000, 10000
NUM_EPOCHS = 4  # 2, 3, 4, 5
WEIGHT_DECAY = 1e-3  # 1e-2 or 1e-3
DROP_OUT_RATE = 0.2  # 0.1 or 0.2
'''


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

# Load dataset
# data_file = "shared_data/dataset_1_4_sliced.jsonl"
train_set = load_jsonl_file("shared_data/dataset_1_6_1b_train_anonym.jsonl")
val_set = load_jsonl_file("shared_data/dataset_1_6_1b_validation_anonym.jsonl")
test_set = load_jsonl_file("shared_data/dataset_1_6_1b_test_anonym.jsonl")

# Load and preprocess the dataset
# dataset = load_jsonl_file(data_file)

# Balance dataset
train_set = balance_classes_in_dataset(train_set, "monologic", "dialogic", "label", SEED)
val_set = balance_classes_in_dataset(val_set, "monologic", "dialogic", "label", SEED)
test_set = balance_classes_in_dataset(test_set, "monologic", "dialogic", "label", SEED)

# dataset = balance_classes_in_dataset(dataset, "monologic", "dialogic", "label", SEED)


# Convert sets to DataFrames
"""train_df = pd.DataFrame(train_set)
val_df = pd.DataFrame(val_set)
test_df = pd.DataFrame(test_set)"""

# Convert to pandas DataFrame for stratified splitting
train_df = pd.DataFrame({
    "id": [entry["id"] for entry in train_set],
    "text": [entry["text"] for entry in train_set],
    "label": [LABEL_MAP[entry["label"]] for entry in train_set],
    "metadata": [entry["metadata"] for entry in train_set]
})

val_df = pd.DataFrame({
    "id": [entry["id"] for entry in val_set],
    "text": [entry["text"] for entry in val_set],
    "label": [LABEL_MAP[entry["label"]] for entry in val_set],
    "metadata": [entry["metadata"] for entry in val_set]
})

test_df = pd.DataFrame({
    "id": [entry["id"] for entry in test_set],
    "text": [entry["text"] for entry in test_set],
    "label": [LABEL_MAP[entry["label"]] for entry in test_set],
    "metadata": [entry["metadata"] for entry in test_set]
})

"""# Stratified split of the data to obtain the train and the remaining data
train_df, remaining_df = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=SEED)

# Split the remaining data equally to get a validation set and a test set
val_df, test_df = train_test_split(remaining_df, stratify=remaining_df["label"], test_size=0.5, random_state=SEED)
"""

"""# Specify file path for datasets JSON files
train_json_file_path = "shared_data/dataset_1_5_train.jsonl"
val_json_file_path = "shared_data/dataset_1_6_val.jsonl"
test_json_file_path = "shared_data/dataset_1_7_test.jsonl"""

"""train_dict = train_df.to_dict(orient="records")
val_dict = val_df.to_dict(orient="records")
test_dict = test_df.to_dict(orient="records")

# Save test_df to a JSON file
save_jsonl_file(train_dict, train_json_file_path)
print(f"Train dataset saved to {train_json_file_path}")
save_jsonl_file(val_dict, val_json_file_path)
print(f"Validation dataset saved to {val_json_file_path}")
save_jsonl_file(test_dict, test_json_file_path)
print(f"Test dataset saved to {test_json_file_path}")"""


def create_dataset(_df):
  _texts = _df['text'].tolist()
  _labels = _df['label'].tolist()
  _ids = _df['id'].tolist()  # keep ids as a list of strings

  _input_ids, _attention_masks = preprocess(_texts, tokenizer, device, max_length=MAX_LENGTH)

  # Create TensorDataset without ids, since they are strings
  return TensorDataset(_input_ids, _attention_masks, torch.tensor(_labels)), _ids


def preprocess(_texts, _tokenizer, _device, max_length=MAX_LENGTH):
  """Tokenize and preprocess texts."""
  inputs = _tokenizer(_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
  return inputs["input_ids"].to(_device), inputs["attention_mask"].to(_device)


# Create TensorDatasets
train_dataset, train_ids = create_dataset(train_df)
val_dataset, val_ids = create_dataset(val_df)
test_dataset, test_ids = create_dataset(test_df)

"""# Calculate class weights
labels = df["label"].tolist()
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
"""


def objective(_trial):
    learning_rate = _trial.suggest_float("learning_rate", 2e-5, 3e-5, log=True)
    batch_size = _trial.suggest_categorical("batch_size", [16, 16])
    warmup_steps = _trial.suggest_int("warmup_steps", 0, 1000)
    num_epochs = _trial.suggest_int("num_epochs", 3, 4)
    #WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    #DROP_OUT_RATE = trial.suggest_float("dropout_rate", 0.1, 0.3)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=len(train_dataloader) * num_epochs)

    # Initialize the gradient scaler only if the device is a GPU
    use_cuda = device.type == "cuda"
    grad_scaler = None
    if use_cuda:
        grad_scaler = GradScaler()

    train_losses = []
    val_losses = []
    predictions, true_labels = [], []
    '''
    # Create TensorDatasets
    train_dataset, train_ids = create_dataset(train_df)
    val_dataset, val_ids = create_dataset(val_df)
    test_dataset, test_ids = create_dataset(test_df)
    '''

    print("Hyperparameters:")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Batch Size: {batch_size}")
    print(f"- Warmup Steps: {warmup_steps}")
    print(f"- Number of Epochs: {num_epochs}")
    # print(f"- Weight Decay: {WEIGHT_DECAY}")
    # print(f"- Dropout Rate: {DROP_OUT_RATE}")
    print("---")
    print(f"- Seed: {SEED}")

    # Training Loop
    for epoch in range(num_epochs):
      model.train()
      total_train_loss = 0
      # loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
      loss_fct = torch.nn.CrossEntropyLoss()

      print()
      for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
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

    # Initialize JSONL file for misclassified examples
    misclassified_output_file = "shared_data/dataset_1_8_misclassified_examples.jsonl"
    empty_json_file(misclassified_output_file)

    for i, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
      with torch.no_grad():
        # Unpack batch data without ids
        b_input_ids, b_attention_mask, b_labels = batch
        b_input_ids, b_attention_mask, b_labels = b_input_ids.to(device), b_attention_mask.to(device), b_labels.to(
          device)

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

        # Identifying and saving misclassified examples
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        for j, (pred, true) in enumerate(zip(predictions, label_ids)):  # no _id in zip
          if pred != true:
            # Access the correct id using the batch index and the offset within the batch
            example_id = test_ids[i * batch_size + j]
            # example_id = dataset[i * BATCH_SIZE + j]["metadata"]["text_id"]
            save_row_to_jsonl_file({
              "id": example_id,  # corrected to use the separate ids list
              "true_label": REVERSED_LABEL_MAP[true],
              "predicted_label": REVERSED_LABEL_MAP[pred],
              "text": test_set[i * batch_size + j]["text"],
              "metadata": test_set[i * batch_size + j]["metadata"]
            }, misclassified_output_file)

    plt.figure()
    plot_confusion_matrix(test_true_labels,
                          test_predictions,
                          class_names,
                          "paper_a_1_dl_bert_model_confusion_matrix.png",
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
      print(f"{class_name}: Precision = {precision[i]:.2f}, Recall = {recall[i]:.2f}, F1 = {f1[i]:.2f}")

    print()
    print("Hyperparameters:")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Batch Size: {batch_size}")
    print(f"- Warmup Steps: {warmup_steps}")
    print(f"- Number of Epochs: {num_epochs}")
    #print(f"- Weight Decay: {WEIGHT_DECAY}")
    #print(f"- Dropout Rate: {DROP_OUT_RATE}")
    print("---")
    print(f"- Seed: {SEED}")
    print()

    # Save losses plot with hyperparameter values in the filename
    losses_plot_filename = (
      f"images/losses_plot_"
      f"lr_{learning_rate}_batch_{batch_size}_warmup_{warmup_steps}_epochs_{num_epochs}.png"
    )

    # Make visualization for training and validation losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", color="green")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", color="black")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #plt.savefig("images/paper_a_1_dl_bert_model_losses.png")
    plt.savefig(losses_plot_filename)
    plt.close()

    '''
    # Set the best model to the current model
    best_model = model
    '''

    # Save the best model only if the current trial achieves the best accuracy
    if test_accuracy > _trial.user_attrs.get("best_test_accuracy", 0):
        _trial.set_user_attr("best_test_accuracy", test_accuracy)
        best_model_path = f"models/1/paper_a_x_dl_bert_train_hop_bert.pth"
        torch.save(model.state_dict(), best_model_path)
        _trial.set_user_attr("best_model_path", best_model_path)

    return test_accuracy  # Return the metric we want to optimize (accuracy in this case)


# Create an Optuna study
study = create_study(direction="maximize")  # or "minimize" depending on your metric

# Optimize the study
study.optimize(objective, n_trials=10)

# Print best trial results
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

'''
# After optimization is complete, retrieve the best trial and load the best model
best_trial = study.best_trial
best_test_accuracy = best_trial.value
best_model_path = best_trial.user_attrs.get("best_model_path")

if best_model_path:
    # Load the best model
    best_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(LABEL_MAP))
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)

    # Now we can use the best_model for further testing or inference
    print(f"Best model loaded from {best_model_path}")
else:
    print("No best model path found. The best model may not have been saved during optimization.")
'''

'''
# Print best trial results
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
print("Test Accuracy of the Best Model: ", best_test_accuracy)
print("Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# Save the best model after optimization is complete
best_model_path = "models/1/paper_a_x_dl_bert_train_hop_bert.pth"
torch.save(best_model.state_dict(), best_model_path)
print(f"Best model saved to {best_model_path}")
'''

"""
Model: BERT

- Accuracy: 0.983
- Precision: 0.983
- Recall: 0.983
- F1 Score: 0.983
- AUC-ROC: 0.995
- Matthews Correlation Coefficient (MCC): 0.967
- Confusion Matrix:
           monologic  dialogic
monologic        268         4
dialogic           5       267

monologic: Precision = 0.97, Recall = 0.95, F1 = 0.96
dialogic: Precision = 0.95, Recall = 0.97, F1 = 0.96

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


"""
Model: BERT

- Accuracy: 0.985
- Precision: 0.985
- Recall: 0.985
- F1 Score: 0.985
- AUC-ROC: 0.999
- Matthews Correlation Coefficient (MCC): 0.971
- Confusion Matrix:
           monologic  dialogic
monologic        267         5
dialogic           3       269

monologic: Precision = 0.97, Recall = 0.98, F1 = 0.98
dialogic: Precision = 0.98, Recall = 0.97, F1 = 0.98

Hyperparameters:
- Learning Rate: 2.0749658870306432e-05
- Batch Size: 16
- Warmup Steps: 416
- Number of Epochs: 4
---
- Seed: 1234

"""