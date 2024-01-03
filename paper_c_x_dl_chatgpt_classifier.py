from pprint import pprint

from lib.utils import load_json_file, load_jsonl_file

import openai

# The ID of your fine-tuned model
model_id = "ft:davinci-002:paco::8cjtj5uI"  # anonymized
# model_id = "ft:davinci-002:paco::8ck4FLNb"  # non-anonymized

# Your API key from OpenAI
openai.api_key = load_json_file("credentials/openai_credentials.json")["openai_api_key"]

# dataset = read_from_google_sheet(spreadsheet_4, "dataset_3")
dataset = load_jsonl_file("shared_data/dataset_3_3_openai.jsonl")

pprint(dataset)


def closest_to_zero(num1, num2):
  return min(num1, num2, key=abs)


# The text you want to classify
# text_to_classify = "I love the recent update, it made the law so much democratic and fairer for everyone"
# text_to_classify = "I have doubts on the recent bill, it made the migrant's situation so much antidemocratic and unfair for everyone"
# N/A
text_to_classify = "And I've asked Louis Moret to serve in the Department of Energy specially director for the minority impact."

# text_to_classify = "This can only mean the so-called normal of former reactionary administrations, the outstanding feature of which was a pittance for farm produce and a small wage for a long day of labor."
# support
# text_to_classify = "When that military objective is accomplished-and much of it has not yet been accomplished-the Italian people will be free to work out their own destiny, under a government of their own choosing."
# oppose
# text_to_classify = "We are more compassionate than a government that lets veterans sleep on our streets and families slide into poverty; that sits on its hands while a major American city drowns before our eyes."

res = openai.Completion.create(model=model_id, prompt=text_to_classify, max_tokens=1, temperature=0, logprobs=2)

pprint(res)

"""oppose_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["op"]
support_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["support"]
print(f"Oppose: {oppose_prediction}")
print(f"Support: {support_prediction}")

# Get the highest logprob
close_to_0 = closest_to_zero(oppose_prediction, support_prediction)
print(close_to_0)"""

"""print("--------------------------------")
if close_to_0 > -0.5:
  prediction = {
    "text": "oppose",
    "score": oppose_prediction
  }
  print(prediction)
elif close_to_0 > -0.5:
  prediction = {
    "text": "support",
    "score": support_prediction
  }
  print(prediction)
else:
  prediction = {
    "text": "undecided",
    "score": close_to_0
  }
  print(prediction)
print("--------------------------------")"""
