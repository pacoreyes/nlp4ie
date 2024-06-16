import torch
import shap

from setfit import SetFitModel
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


# Get best device
device = get_device()

model_id = "sentence-transformers/paraphrase-mpnet-base-v2"  # Example model_id

# Load model
model_setfit_path = "models/9"
model = SetFitModel.from_pretrained(model_setfit_path, local_files_only=True)

# Move the model to the chosen device
model.to(device)

# Load the corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Example usage of tokenizer
"""sample_text = "This is a sample sentence for tokenization."

# Tokenize the sample text
tokens = tokenizer(sample_text)

# Print the tokens
print("Tokenized output:", tokens)

# Decode the input_ids back to tokens
decoded_tokens = tokenizer.convert_ids_to_tokens(tokens['input_ids'])

# Print the decoded tokens
print("Decoded tokens:", decoded_tokens)"""

texts = ["This is a sample sentence for tokenization.", "This is another sample sentence for tokenization."]


# build an explainer using a token masker
explainer = shap.Explainer(model, tokenizer)

shap_values = explainer(texts[:1], fixed_context=1)

print(shap_values)

shap.plots.text(shap_values)
