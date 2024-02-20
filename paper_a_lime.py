import os

# Disable upper limit for MPS memory allocations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import torch
import spacy
from tqdm import tqdm

from lib.utils import load_jsonl_file
# from lib.ner_processing import custom_anonymize_text

# Load Spacy NLP model
nlp = spacy.load("en_core_web_trf")

class_name = ['monologic', 'dialogic']

# Load dataset
dataset = load_jsonl_file("shared_data/dataset_1_6_1b_test.jsonl")
# Load mismatched datapoint
mismatched_datapoint = load_jsonl_file("dataset_1_8_2b_misclassified_examples.jsonl")

# datapoint = dataset[0]


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  """  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")"""
  return torch.device("cpu")


def predict_proba(_text):
  """
  Prediction function that takes a list of texts and returns model predictions.
  """
  # Tokenize text input for BERT
  inputs = tokenizer(_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
  # Get model predictions
  with torch.no_grad():
    outputs = model(**inputs)
  # Apply softmax to get probabilities from logits
  probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
  return probabilities.cpu().detach().numpy()


# Set device
device = get_device()
print(f"\nUsing device: {str(device).upper()}\n")

# Initialize constants
BERT_MODEL = 'bert-base-uncased'
MODEL_PATH = 'models/1/paper_a_x_dl_bert_train_hop_bert.pth'

# Load BERT Tokenizer
print("• Loading BERT Tokenizer...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
# Load Pre-trained Model
print("• Loading Pre-trained Model...")
model = BertForSequenceClassification.from_pretrained(BERT_MODEL)

# Move Model to Device
model.to(device)

# Load Saved Weights
print("• Loading Saved Weights...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Prepare Model for Evaluation
model.eval()

# Initialize LIME Text Explainer
explainer = LimeTextExplainer(class_names=class_name)

for mismatch in tqdm(mismatched_datapoint, desc="Generating Explanations"):
  datapoint = None
  for data in dataset:
    if data["id"] == mismatch["id"]:
      datapoint = data
      break

  text = datapoint["text"]
  # Generate explanation
  print("• Generating Explanation...")
  exp = explainer.explain_instance(
    text_instance=text, classifier_fn=predict_proba, num_features=5, num_samples=1000)

  print(f"True class: {mismatch['label']}")
  print(f"Predicted class: {datapoint['label']}")

  # Save the explanation to an HTML file
  exp.save_to_file(f'xnlp/lime_explanation_{datapoint["id"]}.html', labels=("monologic", "dialogic"))
