import torch
from transformers import BertTokenizer, BertForSequenceClassification


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

# Pre-trained model Name
MODEL_NAME = 'bert-base-uncased'

# Load Pre-trained Model
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Load Saved Weights
model.load_state_dict(torch.load('models/2/paper_b_hop_bert_reclass.pth'))

# Move model to device
model.to(device)

# Prepare Model for Evaluation
model.eval()

# Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def predict(_model, _text):
  # Split the input text into two sentences
  sentence1, sentence2 = _text.split('[SEP]')

  # Encode the pair of sentences, adding special tokens, and ensuring PyTorch tensor output
  input_ids = tokenizer.encode(
    text=sentence1.strip(),
    text_pair=sentence2.strip(),
    add_special_tokens=True,
    max_length=512,
    truncation=True,
    return_tensors="pt"
  )

  # Assuming the model and tokenizer are on the same device, move input_ids to model's device
  input_ids = input_ids.to(device)

  # If your model expects attention masks, you would need to generate them manually here
  # For simplicity, this step is omitted

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
text = "Tokyo is reach. [SEP] Yes it is reach"
probabilities = predict(model, text)
print("Probabilities:", probabilities)

# text = "That's the only explanation for why Saddam Hussein does not want inspectors in from the U.N. [SEP] Iraq continues to be unable to say that its neighbors have a right to exist, like Kuwait."
