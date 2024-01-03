from pprint import pprint

from lib.utils import load_json_file, load_jsonl_file
from lib.utils2 import anonymize_text

import openai
import spacy

WITH_ANONYMIZATION = True

# The ID of your ChatGPT fine-tuned model
model_id = "ft:davinci-002:paco::8cjtj5uI"  # anonymized
# model_id = "ft:davinci-002:paco::8ck4FLNb"  # non-anonymized

# Your API key from OpenAI
openai.api_key = load_json_file("credentials/openai_credentials.json")["openai_api_key"]

# Load dataset
dataset = load_jsonl_file("shared_data/dataset_3_3_openai.jsonl")

# Load spaCy model
nlp_trf = spacy.load("en_core_web_trf")


def closest_to_zero(num1, num2):
  return min(num1, num2, key=abs)


support = [
  "I want to explain why we have taken military action against Saddam Hussein, and why we believe this action is in the interests of the Iraqi people and all the people of the Middle East.",
  "I'd like to see the day when every major river in our country is under control from its source to its mouth, when they are all wealth-givers instead of wealth-destroyers, when every one is running clean and pure and doing the work it ought to do for the people of this country.",
  "Since small business has such a stabilizing influence on recession, I think this is a good sign for all Americans.",
  "Our across-the-board cut in tax rates for a 3-year period will give them much of the incentive and promise of stability they need to go forward with expansion plans calling for additional employees.",
  "It has the authority to administer a gas rationing plan, and prior to decontrol it ran the oil price control program.",
  "And we will seek to cooperate with Russia to solve regional problems, while insisting that if Russian troops operate in neighboring states, they do so only when those states agree to their presence and in strict accord with international standards.",
  "To overcome dangers in our world, we must also take the offensive by encouraging economic progress and fighting disease and spreading hope in hopeless lands.",
  "These expenditure increases, let me stress, constitute practically all of the increases which have occurred under this administration, the remainder having gone to fight the recession we found in industry--mostly through the supplemental employment bill-and in agriculture."
]

oppose = [
  "His war against Iran cost at least half a million lives over 10 years.",
  "They have fought against advances in housing and health and education.",
  "At the same time, I think it cuts the defense budget too much, and by attempting to reduce the deficit through higher taxes, it will not create the kind of strong economic growth and the new jobs that we must have.",
  "Adding to our troubles is a mass of regulations imposed on the shopkeeper, the farmer, the craftsman, professionals, and major industry that is estimated to add $100 billion to the price of the things we buy, and it reduces our ability to produce.",
  "If we do nothing, American families will face a massive tax increase they do not expect and will not welcome.",
  "It's because they understand that when I get a tax break I don't need and the country can't afford, it either adds to the deficit or somebody else has to make up the difference, like a senior on a fixed income or a student trying to get through school or a family trying to make ends meet.",
  "Those with money and power will gain greater control over the decisions that could send a young soldier to war or allow another economic disaster or roll back the equal rights and voting rights that generations of Americans have fought, even died, to secure.",
  "What I am concerned about is the kind of deficit we would have if we had a recession, and while the prospects for a recession are not certainly imminent before us, we do have to look at our historical record and realize that any society such as ours, particularly with the tax structure such as ours, must face that prospect at some time.",
]

unclassified = [
  "Members of this Congress who went there as observers told me of a woman who was wounded by rifle fire on the way to the polls, who refused to leave the line to have her wound treated until after she had voted.",
  "Through the long years of conflict, four main issues have divided the parties involved.",
  "Federal civilian employment, for example, is actually lower today than it was in 1952, while State and local government employment over the same period has increased 67 percent.",
  "I hope that this lack will not make the veterans think I am any less deeply thrilled by the memory of their great comrades gone before - Grant, Hayes, Garfield, Harrison and McKinley - all sons of Ohio, who left records reflecting glory upon their State and Nation, or that my sympathies with the valor and courage and patriotism of those who faced death in the country's crises are any less earnest and sincere than they would be had I the right to wear a button of the Grand Army or of the veteran association of any of our country's wars.",
  "We meet at one of those defining moments - a moment when our nation is at war, our economy is in turmoil, and the American promise has been threatened once more.",
  "A little later he was made major general of Volunteers and placed in command of the Fifth Army Corps.",
  "When it is remembered that before this terrific engagement Meade had been in command of the Army but three days, his victory becomes the more wonderful.",
  "On the 1st of February, 1865, he was made a major general in the Regular Army."
]

print("################################")
print("Support class:\n")

for text in support:
  if WITH_ANONYMIZATION:
    text = anonymize_text(text, nlp_trf)

  res = openai.Completion.create(model=model_id, prompt=text, max_tokens=1, temperature=0, logprobs=2)

  pprint(res['choices'][0]["text"])
  pprint(res['choices'][0]['logprobs']['top_logprobs'][0])
  """oppose_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["op"]
  support_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["support"]
  print(f"Oppose: {oppose_prediction}")
  print(f"Support: {support_prediction}")"""

print("################################")
print("Oppose class:\n")

for text in oppose:
  if WITH_ANONYMIZATION:
    text = anonymize_text(text, nlp_trf)

  res = openai.Completion.create(model=model_id, prompt=text, max_tokens=1, temperature=0, logprobs=2)

  pprint(res['choices'][0]["text"])
  pprint(res['choices'][0]['logprobs']['top_logprobs'][0])
  """oppose_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["op"]
  support_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["support"]
  print(f"Oppose: {oppose_prediction}")
  print(f"Support: {support_prediction}")"""

print("################################")
print("Unclassified class:\n")

for text in unclassified:
  if WITH_ANONYMIZATION:
    text = anonymize_text(text, nlp_trf)

  res = openai.Completion.create(model=model_id, prompt=text, max_tokens=1, temperature=0, logprobs=2)

  pprint(res['choices'][0]["text"])
  pprint(res['choices'][0]['logprobs']['top_logprobs'][0])
  """oppose_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["op"]
  support_prediction = res['choices'][0]['logprobs']['top_logprobs'][0]["support"]
  print(f"Oppose: {oppose_prediction}")
  print(f"Support: {support_prediction}")"""


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
