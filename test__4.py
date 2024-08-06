import shap
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer

from lib.utils import load_jsonl_file
from lib.ner_processing import custom_anonymize_text

SEED = 42
BATCH_SIZE = 16
CLASS_NAMES = ['continue', 'not_continue']

# Load dataset
DATASET = load_jsonl_file("shared_data/topic_continuity_test.jsonl")


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


# Set device
device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")

# Initialize constants
BERT_MODEL = 'bert-base-uncased'
MODEL_PATH = 'models/3/TopicContinuityBERT.pth'

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=len(CLASS_NAMES))

# Move the model to the device
model = model.to(device)
# Load the model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# Set the model to evaluation mode
model.eval()

sentence_a = "And in numbers, they are not insignificant."
sentence_b = "In numbers like the hundreds of thousands, enough to change the election."

# Tokenization
encoding = tokenizer(sentence_a, sentence_b, padding="max_length", truncation=True, return_tensors="pt")
input_ids = encoding['input_ids'].numpy()
attention_mask = encoding['attention_mask'].numpy()
token_type_ids = encoding['token_type_ids'].numpy()


# Prediction function
def predict(input_data):
  input_ids = torch.tensor(input_data[:, :512], dtype=torch.long)
  token_type_ids = torch.tensor(input_data[:, 512:1024], dtype=torch.long)
  attention_mask = torch.tensor(input_data[:, 1024:], dtype=torch.long)

  with torch.no_grad():
    outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
  return outputs.logits.numpy()


# Prepare data for SHAP
input_data = np.concatenate([input_ids, token_type_ids, attention_mask], axis=1)

# SHAP KernelExplainer
explainer = shap.KernelExplainer(predict, input_data)
shap_values = explainer.shap_values(input_data)

