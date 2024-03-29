import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

from lib.utils import load_jsonl_file

test_set = load_jsonl_file("shared_data/adversarial/interview/2.json")

example = test_set[0]


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


class_names = ["speech", "interview"]


device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")

# Pre-trained model Name
MODEL_NAME = 'bert-base-uncased'

# Load Pre-trained Model
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Load Saved Weights
model.load_state_dict(torch.load('models/1/paper_a_x_dl_bert_train_hop_bert_anonym.pth',
                                 map_location=torch.device('cpu')))

# Move model to device
model.to(device)

# Prepare Model for Evaluation
model.eval()

# Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def predict(_model, _text):

  input_ids = tokenizer.encode(
    text=_text,
    add_special_tokens=True,
    max_length=512,
    truncation=True,
    return_tensors="pt"
  )

  # Assuming the model and tokenizer are on the same device, move input_ids to model's device
  input_ids = input_ids.to(device)

  # Prediction step without computing gradients for efficiency
  with torch.no_grad():
    outputs = _model(input_ids)
  logits = outputs.logits

  # Convert logits to probabilities
  _probabilities = torch.nn.functional.softmax(logits, dim=1)

  # Convert to numpy arrays for easier handling
  _probabilities = _probabilities.cpu().numpy()

  return _probabilities


# Example usage
text = example["text"]
probabilities = predict(model, text)
print("Probabilities:", probabilities)
# Find the index of the highest probability
max_index = np.argmax(probabilities, axis=1)
# Map the index to the corresponding class name
predicted_class = [class_names[index] for index in max_index]
print("Predicted class:", predicted_class)
