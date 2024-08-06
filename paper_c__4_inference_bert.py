from transformers import BertTokenizer, BertForSequenceClassification
import torch


# Initialize label map and class names
LABEL_MAP = {"continue": 0, "not_continue": 1}
class_names = list(LABEL_MAP.keys())
REVERSED_LABEL_MAP = {0: "continue", 1: "not_continue"}


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


# Pre-trained model Name
MODEL_NAME = 'bert-base-uncased'

# Load Pre-trained Model
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Load Saved Weights
# model.load_state_dict(torch.load('models/11/topic_boundary_detection.pth'))
model.load_state_dict(torch.load('models/3/TopicContinuityBERT.pth'))

# Move model to device
model.to(device)

# Prepare Model for Evaluation
model.eval()

# Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


"""def prepare_sentence_pairs(_sentence_pairs, _tokenizer, max_length=512):
  input_ids = []
  attention_masks = []

  for pair in _sentence_pairs:
    sentence1, sentence2 = pair.split('[SEP]')
    encoded_input = _tokenizer.encode_plus(
      text=sentence1.strip(),
      text_pair=sentence2.strip(),
      add_special_tokens=True,
      max_length=max_length,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt'
    )
    input_ids.append(encoded_input['input_ids'])
    attention_masks.append(encoded_input['attention_mask'])

  # Convert list of tensors to a single tensor
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)

  return input_ids, attention_masks"""


def preprocess_pairs(_texts, _tokenizer, max_length=512):
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

  return input_ids, attention_masks


def classify_sentences(_model, _sentence_pairs, _tokenizer, _device):
  # model.eval()  # Ensure the model is in eval mode
  input_ids, attention_masks = preprocess_pairs(_sentence_pairs, _tokenizer)

  # Move tensors to the device where the model is
  input_ids = input_ids.to(_device)
  attention_masks = attention_masks.to(_device)

  with torch.no_grad():  # No need to track gradients for inference
    outputs = _model(input_ids, attention_mask=attention_masks)
    logits = outputs.logits
    _predictions = torch.argmax(logits, dim=-1)  # Get the predicted classes

  return _predictions


def inference_pair(_sentence_pairs):
  predictions = classify_sentences(model, _sentence_pairs, tokenizer, device)
  predictions = [REVERSED_LABEL_MAP[p.item()] for p in predictions]
  return predictions[0]


"""sentence1 = "And in numbers, they are not insignificant."
sentence2 = "In numbers like the hundreds of thousands, enough to change the election."

sentence_pairs = [sentence1 + '[SEP]' + sentence2]

print(inference_pair(sentence_pairs))"""

"""predictions = classify_sentences(model, sentence_pairs, tokenizer, device)
print(predictions)

# Convert predictions to class names
predictions = [REVERSED_LABEL_MAP[p.item()] for p in predictions]
print(predictions)"""
