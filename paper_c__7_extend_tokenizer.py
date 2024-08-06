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


# Initialize the device
device = get_device()

# Initialize constants
BERT_MODEL = 'bert-base-uncased'
MODEL_PATH = 'models/3/TopicContinuityBERT.pth'

# Step 4: Load the model
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
# Load the model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Step 1: Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)  # , never_split=NEVER_SPLIT

# Step 2: Add new tokens
new_tokens = ["1,000", "2,000", "endures", "decency", "stockpile", "ventilators", "Blackwater", "standpoint",
              "dismantle", "empower", "frack", "polluters", "Saddamists", "rejectionists", "Qaida", "maiming",
              "torturing", "healthier", "massively", "asymptomatic", "Pocan", "unfairly", "1,400", "'s", "62,000",
              "hospitalizations", "490,050", "commend", "F-16", "opioid", "pushers", "peddling", "Ebola", "czar",
              "reiterate", "USAID", "maximally", "unwittingly", "'d", "Assad", "pandemic", "deadliest", "defunding",
              "ATF", "pressuring", "DACA", "U.S.", "basing", "hospitalization", "COVID", "incentivize", "reimagine",
              "dictate", "beneficiary", "closures", "lawmakers", "equipping", "vaccination", "retrain", "Hun-",
              "nutritious", "inhumane", "qualifies", "lifeblood", "forecasts", "vaccinated", "1619", "hundreds",
              "70,000", "legislating", "Javits", "childcare", "reemphasized", "destabilizing", "exporter", "COVID-19",
              "vaccinations", "ISR", "Abound", "1,500", "FDIC", "2.9", "IndyMac", "5,000", "borrowers", "foreclosure",
              "mortgages", "2.2", "mandating", "meddling", "11,000", "vaccinating", "Ivey", "belittling",
              "inclusiveness", "canceling", "d.c.", "docs", "harnessing", "refitting", "iranians", "Rouhani", "Forked",
              "4,000", "positives", "oftentimes", "NAFTA", "biofuels"]

#  "EO", "pand", "expand",

# Step 3: Add the new tokens to the tokenizer
tokenizer.add_tokens(new_tokens)

# Step 5: Resize the token embeddings
model.resize_token_embeddings(len(tokenizer))

# Step 6: Save the updated tokenizer
save_directory = 'models/3/tokenizer'
tokenizer.save_pretrained(save_directory)

"""# Step 7: Save the updated model
save_directory = 'models/3/resized_model'
model.save_pretrained(save_directory)"""
