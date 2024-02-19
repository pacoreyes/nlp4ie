import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from lib.utils import load_txt_file

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

file_path = 'new_text.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

def preprocess(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    return inputs


# 5. Make Predictions
def predict(text):
    inputs = preprocess(text)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    return probabilities

for line in lines:
    data_point = line.strip()
    result = predict(data_point)
    print(result)

''''# Example
text = "Tokiyo is reach. [SEP] Yes it is reach"
prediction = predict(text)
print(prediction)
'''
