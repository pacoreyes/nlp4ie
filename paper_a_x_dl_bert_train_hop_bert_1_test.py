import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import (confusion_matrix, roc_auc_score, matthews_corrcoef, accuracy_score,
                             precision_recall_fscore_support)
from lib.visualizations import plot_confusion_matrix
import matplotlib.pyplot as plt



# Define constants and parameters
LABEL_MAP = {"monologic": 0, "dialogic": 1}
class_names = list(LABEL_MAP.keys())

MAX_LENGTH = 512  # Should match the value used during training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to preprocess texts
def preprocess(texts, tokenizer, device, max_length=MAX_LENGTH):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the saved model
saved_model_path = "models/1/paper_a_x_dl_bert_train_hop_bert.pth"
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(LABEL_MAP))
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.to(device)

# Load and preprocess the test dataset
data_file = "shared_data/dataset_1_4_sliced.jsonl"
test_dataset = pd.read_json(data_file, lines=True)

# Tokenize and preprocess test data
input_ids, attention_masks = preprocess(test_dataset['text'].tolist(), tokenizer, device, max_length=MAX_LENGTH)

# Create TensorDataset
test_labels = [LABEL_MAP[label] for label in test_dataset['label']]
test_dataset = TensorDataset(input_ids, attention_masks, torch.tensor(test_labels))

# Define test DataLoader
#batch_size = 32  # Adjust as needed
#test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=False)


# Function to preprocess texts
def preprocess(texts, tokenizer, device, max_length=MAX_LENGTH):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

# Perform inference on the test data
predictions = []
true_labels = []

# Test Loop
model.eval()
total_test_loss = 0
#test_predictions, test_true_labels = [], []
all_probabilities = []

softmax = torch.nn.Softmax(dim=1)

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing"):
        b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

        # Forward pass
        outputs = model(b_input_ids, attention_mask=b_attention_mask)
        logits = outputs.logits
        probabilities = softmax(logits)
        all_probabilities.append(probabilities.cpu().numpy())  # Move to CPU before conversion

        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())

        label_ids = b_labels.cpu().numpy()  # Move to CPU before conversion

        # Store predictions and true labels
        #test_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())  # Move to CPU before conversion
        #test_true_labels.extend(label_ids)

# Evaluate the model's performance on the test set
accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# Make confusion matrix
cm = confusion_matrix(true_labels, predictions)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
print(f"- Confusion Matrix:")
print(df_cm)

plt.figure()
plot_confusion_matrix(true_labels,
                        predictions,
                        class_names,
                        "paper_a_x_dl_bert_train_hop_bert.png",
                        "Confusion Matrix for BERT Model",
                        values_fontsize=22
                        )

# Concatenate all probabilities
all_probabilities = np.concatenate(all_probabilities, axis=0)

test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
      true_labels, predictions, zero_division=0)

# Get AUC ROC score
roc_auc = roc_auc_score(true_labels, all_probabilities[:, 1])

# Get MCC score
mcc = matthews_corrcoef(true_labels, predictions)

precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions)

print("\nModel: BERT\n")
print(f"- Accuracy: {accuracy:.3f}")
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

