import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Replace this with your model's name or path if you used a different model
MODEL_NAME = 'bert-base-uncased'

# 1. Load Pre-trained Model
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# 2. Load Saved Weights
model.load_state_dict(torch.load('models/2/paper_b_99_dl_bert_train.pth'))

# 3. Prepare Model for Evaluation
model.eval()

# 4. Preprocess Input Data
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def preprocess(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    return inputs


# 5. Make Predictions
def predict(text):
    inputs = preprocess(text)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits


# Example
text = "Your input text here"
prediction = predict(text)
print(prediction)
