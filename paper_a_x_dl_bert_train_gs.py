import random
import itertools

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import (confusion_matrix, roc_auc_score, matthews_corrcoef,
                             accuracy_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
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

EXPERIMENT_NAME = "1-GRID-EXPERIMENT-paper_a_x_dl_bert_train_gs"

# Initialize constants
MAX_LENGTH = 512  # the maximum sequence length that can be processed by the BERT model
SEED = 42  # 42, 1234, 2021

"""
# Hyperparameters
LEARNING_RATE = 2e-5  # 1.5e-5, 2e-5, 3e-5, 5e-5
BATCH_SIZE = 16  # 16, 32
WARMUP_STEPS = 600  # 0, 100, 1000, 10000
NUM_EPOCHS = 3  # 2, 3, 4, 5
WEIGHT_DECAY = 1e-3  # 1e-2 or 1e-3
DROP_OUT_RATE = 0.1  # 0.1 or 0.2

"""

# Hyperparameters grid
hyperparameters_grid = {
  "learning_rate": [1.5e-5, 2e-5, 2.5e-5, 3e-5, 3.5e-5],
  "batch_size": [16, 32],
  "warmup_steps": [100, 600, 1000],
  "num_epochs": [2, 3, 4],
  "weight_decay": [0, 1e-2, 1e-3],
  "drop_out_rate": [0.1, 0.2]
}


# Generate all combinations of hyperparameters
hyperparameters_combinations = list(itertools.product(*hyperparameters_grid.values()))


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


def set_seed(seed_value):
  """Set seed for reproducibility."""
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def create_dataset(_df):
  _texts = _df['text'].tolist()
  _labels = _df['label'].tolist()

  _input_ids, _attention_masks = preprocess(_texts, tokenizer, device, max_length=MAX_LENGTH)
  return TensorDataset(_input_ids, _attention_masks, torch.tensor(_labels))


def preprocess(_texts, _tokenizer, _device, max_length=MAX_LENGTH):
  """Tokenize and preprocess texts."""
  inputs = _tokenizer(_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
  return inputs["input_ids"].to(_device), inputs["attention_mask"].to(_device)


# Run experiment for each combination of hyperparameters
for hyperparameters in hyperparameters_combinations:

  # Create a dictionary of the current hyperparameters
  hyperparameters_dict = dict(zip(hyperparameters_grid.keys(), hyperparameters))

  # Update the global variables with the current hyperparameters
  LEARNING_RATE = hyperparameters_dict["learning_rate"]
  BATCH_SIZE = hyperparameters_dict["batch_size"]
  WARMUP_STEPS = hyperparameters_dict["warmup_steps"]
  NUM_EPOCHS = hyperparameters_dict["num_epochs"]
  WEIGHT_DECAY = hyperparameters_dict["weight_decay"]
  DROP_OUT_RATE = hyperparameters_dict["drop_out_rate"]

  # Set device to CUDA, MPS, or CPU
  device = get_device()
  print(f"\nUsing device: {str(device).upper()}\n")

  # Load dataset
  data_file = "shared_data/dataset_1_4_sliced.jsonl"

  # Load BERT model
  model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(LABEL_MAP),
                                                        hidden_dropout_prob=DROP_OUT_RATE)
  # Move model to device
  model.to(device)

  # Load and preprocess the dataset
  dataset = load_jsonl_file(data_file)

  #dataset = dataset[:5]

  # Set seed for reproducibility
  set_seed(SEED)
  # torch.use_deterministic_algorithms(True)

  # Load the BERT tokenizer
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

  sentences = [entry["text"] for entry in dataset]
  labels = [entry["label"] for entry in dataset]

  # Convert labels from string to int
  labels = [LABEL_MAP[label] for label in labels]

  # Convert to pandas DataFrame for stratified splitting
  df = pd.DataFrame({"text": sentences, "label": labels})

  # Stratified split of the data to obtain the train and the remaining data
  train_df, remaining_df = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=SEED)

  # Split the remaining data equally to get a validation set and a test set
  val_df, test_df = train_test_split(remaining_df,
                                     stratify=remaining_df["label"], test_size=0.5, random_state=SEED)

  # Create TensorDatasets
  train_dataset = create_dataset(train_df)
  val_dataset = create_dataset(val_df)
  test_dataset = create_dataset(test_df)

  # Calculate class weights
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

  # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

  # Set MLflow experiment
  mlflow.set_experiment(EXPERIMENT_NAME)

  print()

  # Initialize the gradient scaler only if the device is a GPU
  use_cuda = device.type == "cuda"
  grad_scaler = None
  if use_cuda:
    grad_scaler = GradScaler()

  train_losses = []
  val_losses = []

  predictions, true_labels = [], []

  # Start MLflow run
  with mlflow.start_run(run_name="My Custom Run"):

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

    # Log metrics with mlflow or print them out
    mlflow.log_metric("val_accuracy", accuracy)
    mlflow.log_metric("Seed", SEED)

    # Log the hyperparameters with MLflow
    mlflow.log_params({
      "LEARNING_RATE": LEARNING_RATE,
      "BATCH_SIZE": BATCH_SIZE,
      "WARMUP_STEPS": WARMUP_STEPS,
      "NUM_EPOCHS": NUM_EPOCHS,
      "WEIGHT_DECAY": WEIGHT_DECAY,
      "DROP_OUT_RATE": DROP_OUT_RATE
    })

    """
    # Log the hyperparameters with MLflow 
    mlflow.log_params("LEARNING_RATE", LEARNING_RATE)
    mlflow.log_params("BATCH_SIZE", BATCH_SIZE)
    mlflow.log_params("WARMUP_STEPS", WARMUP_STEPS)
    mlflow.log_params("NUM_EPOCHS", NUM_EPOCHS)
    mlflow.log_params("WEIGHT_DECAY", WEIGHT_DECAY)
    mlflow.log_params("DROP_OUT_RATE", DROP_OUT_RATE)
    
    """

    print("\nAccuracy:", accuracy)
    print("Training Class-wise metrics:")
    for i, class_name in enumerate(class_names):
      print(f"{class_name}: Precision = {precision[i]:.2f}, Recall = {recall[i]:.2f}, F1 = {f1[i]:.2f}")
      mlflow.log_metric(f"precision_{class_name}", precision[i])
      mlflow.log_metric(f"recall_{class_name}", recall[i])
      mlflow.log_metric(f"f1_{class_name}", f1[i])

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

    confusion_matrix_filename = (
      f"confusion_matrix_"
      f"lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_warmup_{WARMUP_STEPS}_epochs_{NUM_EPOCHS}"
      f"_wd_{WEIGHT_DECAY}_dropout_{DROP_OUT_RATE}.png"
    )

    plt.figure()
    plot_confusion_matrix(test_true_labels,
                          test_predictions,
                          class_names,
                          confusion_matrix_filename,
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

    # Log metrics with mlflow or print them out
    mlflow.log_metric("test_accuracy", test_accuracy)

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

      mlflow.log_metric(f"test_precision_{class_name}", test_precision[i])
      mlflow.log_metric(f"test_recall_{class_name}", test_recall[i])
      mlflow.log_metric(f"test_f1_{class_name}", test_f1[i])

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

    # Optionally, you can also log the model
    mlflow.pytorch.log_model(model, "model")

    # Save losses plot with hyperparameter values in the filename
    losses_plot_filename = (
      f"images/losses_plot_"
      f"lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_warmup_{WARMUP_STEPS}_epochs_{NUM_EPOCHS}"
      f"_wd_{WEIGHT_DECAY}_dropout_{DROP_OUT_RATE}.png"
    )

    # Make visualization for training and validation losses
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training Loss", color="green")
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", color="black")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(losses_plot_filename)
    plt.close()

    # Log the image name with MLflow
    mlflow.log_params({
      "Confusion_Matrix": confusion_matrix_filename,
      "Losses_Plot": losses_plot_filename
    })

# End MLFlow run
mlflow.end_run()

"""
"""
