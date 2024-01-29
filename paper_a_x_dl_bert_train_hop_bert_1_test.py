import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm

# Define constants and parameters
LABEL_MAP = {"monologic": 0, "dialogic": 1}
MAX_LENGTH = 512  # Should match the value used during training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to preprocess texts
def preprocess(texts, tokenizer, device, max_length=MAX_LENGTH):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the saved model
saved_model_path = "models/3/best_model.pth"
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(LABEL_MAP))
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.to(device)
model.eval()

# Load and preprocess the test dataset
data_file = "shared_data/dataset_1_4_sliced.jsonl"
test_dataset = pd.read_json(data_file, lines=True)

# Tokenize and preprocess test data
input_ids, attention_masks = preprocess(test_dataset['text'].tolist(), tokenizer, device, max_length=MAX_LENGTH)

# Create TensorDataset
test_labels = [LABEL_MAP[label] for label in test_dataset['label']]
test_dataset = TensorDataset(input_ids, attention_masks, torch.tensor(test_labels))

# Define test DataLoader
batch_size = 8  # Adjust as needed
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# Function to preprocess texts
def preprocess(texts, tokenizer, device, max_length=MAX_LENGTH):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

# Perform inference on the test data
predictions = []
true_labels = []

softmax = torch.nn.Softmax(dim=1)

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing"):
        b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

        # Forward pass
        outputs = model(b_input_ids, attention_mask=b_attention_mask)
        logits = outputs.logits
        probabilities = softmax(logits)
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())

# Evaluate the model's performance on the test set
accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")
