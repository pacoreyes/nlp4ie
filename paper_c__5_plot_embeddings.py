from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from lib.utils import load_jsonl_file


def get_device():
  """Returns the appropriate device available in the system: CUDA, MPS, or CPU"""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")


print("Loading BERT model and test set...")

# Initialize BERT model
BERT_MODEL = 'bert-base-uncased'

# Assuming you've saved your best model to a path during training
MODEL_PATH = "models/3/TopicContinuityBERT.pth"
# MODEL_PATH = "models/1/paper_a_bert_solo.pth"

print(MODEL_PATH)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# Load Pre-trained Model
model = BertForSequenceClassification.from_pretrained(
  BERT_MODEL,
  output_hidden_states=True,
  num_labels=2
)

# Move Model to Device
device = get_device()
model.to(device)

# Load Saved Weights
print("• Loading Saved Weights...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Load test set
test_set = load_jsonl_file("shared_data/topic_boundary_test2.jsonl")
# test_set = load_jsonl_file("shared_data/dataset_1_6_1b_test.jsonl")

texts = [entry["text"] for entry in test_set]
labels = [entry["label"] for entry in test_set]

# Modified part to handle sentence pairs like in training
encoded_texts = []
for text in texts:
    sentence1, sentence2 = text.split('[SEP]')
    encoded_input = tokenizer.encode_plus(
        text=sentence1.strip(),
        text_pair=sentence2.strip(),
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    encoded_texts.append(encoded_input)

# Stack tensors to form a single input batch
input_ids = torch.cat([entry['input_ids'] for entry in encoded_texts], dim=0)
attention_masks = torch.cat([entry['attention_mask'] for entry in encoded_texts], dim=0)

# Forward pass to get hidden states
with torch.no_grad():
  model.eval()
  # Ensure tensors are on the correct device
  input_ids = input_ids.to(device)
  attention_masks = attention_masks.to(device)
  outputs = model(input_ids, attention_mask=attention_masks)
  hidden_states = outputs.hidden_states

# Extract embeddings (last layer's hidden state for [CLS] token)
embeddings = hidden_states[-1][:, 0, :].detach().cpu().numpy()


# Map labels to colors
colors = ['green' if label == "continue" else 'orange' for label in labels]

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=1234)
tsne_results = tsne.fit_transform(embeddings)

# Plot t-SNE with colors
plt.figure(figsize=(10, 6))
for i in range(len(tsne_results)):
    plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=colors[i],
                label=labels[i] if i == 0 or labels[i] != labels[i-1] else "")
plt.title('t-SNE visualization of BERT embeddings')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.legend()
plt.savefig('images/paper_c_plot_bert_embeddings_22_07.png')
# plt.show()
plt.close()
