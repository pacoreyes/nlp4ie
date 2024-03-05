from bertviz import head_view
from transformers import BertTokenizer, BertModel

# Load model
model_version = 'bert-base-uncased'
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version)

passage = "Abortion has been at the forefront of political debates for decades. Its supporters argue that women should have the right to choose, emphasizing personal autonomy. However, its opponents believe the procedure denies the fetus its own rights."

# Tokenize input and get attention weights
input_ids = tokenizer.encode(passage, return_tensors='pt', add_special_tokens=True)
attention = model(input_ids)[2]

# Convert token IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Use BERTViz's head_view function to visualize
head_view(attention, tokens)
