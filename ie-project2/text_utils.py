import re
# from pprint import pprint


def preprocess_text(text: str, nlp,
                    with_remove_known_unuseful_strings: bool,
                    with_remove_parentheses_and_brackets: bool,
                    with_remove_text_inside_parentheses: bool,
                    with_remove_leading_patterns: bool,
                    with_remove_timestamps: bool,
                    with_replace_unicode_characters: bool,
                    with_expand_contractions: bool,
                    with_remove_links_from_text: bool,
                    with_put_placeholders: bool,
                    with_final_cleanup: bool
                    ) -> str:
  """
  Preprocesses the given text by applying a series of cleaning and formatting steps.

  Args:
      text (str): The input text to preprocess.
      nlp (spacy.Language): The Spacy language model for advanced text processing.
      with_put_placeholders (bool): Whether to replace certain text patterns with placeholders.
      with_remove_known_unuseful_strings (bool): Whether to remove known unuseful strings.
      with_remove_parentheses_and_brackets (bool): Whether to remove parentheses and brackets.
      with_remove_text_inside_parentheses (bool): Whether to remove text inside parentheses.
      with_remove_leading_patterns (bool): Whether to remove leading patterns.
      with_remove_timestamps (bool): Whether to remove timestamps.
      with_replace_unicode_characters (bool): Whether to replace unicode characters.
      with_expand_contractions (bool): Whether to expand contractions.
      with_remove_links_from_text (bool): Whether to remove links from text.
      with_final_cleanup (bool): Whether to perform a final cleanup.

  Returns:
      str: The preprocessed text.

  """
  if with_remove_known_unuseful_strings:
    text = remove_known_unuseful_strings(text)
  if with_remove_parentheses_and_brackets:
    text = remove_parentheses_and_brackets(text)
  if with_remove_text_inside_parentheses:
    text = remove_text_inside_parentheses(text)
  if with_remove_leading_patterns:
    text = remove_leading_patterns(text)
  if with_remove_timestamps:
    text = remove_timestamps(text)
  if with_replace_unicode_characters:
    text = replace_unicode_characters(text)
  if with_expand_contractions:
    text = expand_contractions(text)
  if with_remove_links_from_text:
    text = remove_links_from_text(text)
  if with_put_placeholders:
    text = put_placeholders(text, nlp)
  if with_final_cleanup:
    text = final_cleanup(text)
  return text


def remove_leading_patterns(text: str) -> str:
  """
  Removes leading patterns and punctuation from the given text.

  Args:
      text (str): The input text.

  Returns:
      str: The text with leading patterns and punctuation removed.
  """

  def _remove_leading_unopened_parenthesis_and_brackets(_text: str) -> str:
    """
    Helper function to remove unopened parentheses and brackets at the beginning of the text.

    Args:
        _text (str): The input text.

    Returns:
        str: The text with unopened parentheses and brackets removed.
    """
    count_parentheses = 0
    count_brackets = 0
    cut_position = 0
    for i, char in enumerate(_text):
      if char == '(':
        count_parentheses += 1
      elif char == ')':
        count_parentheses -= 1
      elif char == '[':
        count_brackets += 1
      elif char == ']':
        count_brackets -= 1
      if count_parentheses < 0 or count_brackets < 0:
        cut_position = i + 1
        count_parentheses = max(0, count_parentheses)
        count_brackets = max(0, count_brackets)
    return _text[cut_position:].strip()

  # Removes leading text before ":", typically label names found in interview transcripts.
  # text = re.sub(r'^[A-Z\s\-\$0-9]+(\s\(\s[A-Z\s\-]+\s\))?:\s', '', text)
  text = re.sub(r"^[^:]*:\s?", '', text)
  # pprint(text)

  # Remove negative lookbehind assertion, "first part) second part..." > "second part..."
  text = _remove_leading_unopened_parenthesis_and_brackets(text)

  # Removes leading punctuation patterns, e.g., from "# Example string" removes "# "
  text = re.sub(r'^[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]\s*', '', text)

  # Removes alphanumeric period patterns, e.g., from "1. Example string" removes "1. "
  text = re.sub(r'^\w+\.\s', '', text)

  # Remove leading hyphens, periods, and spaces
  text = re.sub(r'^[-\.,\s]+', '', text)

  return text


def remove_parentheses_and_brackets(text: str) -> str:
  """
  Removes parentheses and brackets from the given text.

  Args:
      text (str): The input text.

  Returns:
      str: The text with parentheses and brackets
  """
  # Removes whatever is inside a () or a [].
  text = re.sub(r'(\(.*?\)|\[.*?\])', '', text)
  return text


def remove_text_inside_parentheses(text: str) -> str:
  return re.sub(r'\(.*?\)', '', text)


def remove_timestamps(text: str) -> str:
  """
  Removes timestamps from a text string.

  Args:
      text (str): The input text.

  Returns:
      str: The text with timestamps removed.
  """
  pattern = r'[\[\(]\d{1,2}:\d{2}:\d{2}[\]\)]'
  return re.sub(pattern, '', text)


def replace_unicode_characters(text: str) -> str:
  """
  Replace various unicode and other characters with appropriate substitutes.

  Args:
      text (str): The input text.

  Returns:
      str: The text with replaced unicode and other characters.
  """
  replacements = {
    '\u200b': ' ',  # replace zero-width space Unicode with space
    '\u2013': '-',  # replace en dash Unicode with hyphen
    '\u2014': '-',  # replace em dash Unicode with hyphen
    '\u2019': "'",  # replace right single quotation mark Unicode with apostrophe
    '\u2018': "'",  # replace left single quotation mark Unicode with apostrophe
    '\n': ' ',  # replace newlines with spaces
    '\"': ' ',  # replace double quotes with spaces
    '\t': ' ',  # replace tabs with spaces
    '\r': ' ',  # replace carriage returns with spaces
    ',': ', ',  # add a space after every comma
  }

  for char, replacement in replacements.items():
    text = text.replace(char, replacement)

  text = text.encode().decode('unicode_escape')  # replace unicode characters with their closest ASCII representation
  text = text.encode('ascii', 'ignore').decode('utf-8')  # remove non-ascii, except for unicode characters
  return text


def remove_known_unuseful_strings(text: str) -> str:
  """
  Removes specific strings known to be unuseful from a text string.

  Args:
      text (str): The input text.

  Returns:
      str: The text with known unuseful strings removed.
  """
  frequent_strings = ["The American Presidency Project",
                      "AboutSearch",
                      "About Search",
                      "Click to watch",
                      "Return to Transcripts main page",
                      "CNN LIVE EVENT/SPECIAL",
                      "Transcribe Your Own Content Try Rev for free and save time transcribing & captioning.",
                      "Transcribe or caption speeches, interviews, meetings, town halls, phone calls, and more.",
                      "Rev is the largest, most trusted, fastest, and most accurate provider of transcription services and closed captioning & subtitling services in the world. Gov.",
                      "This website stores cookies on your computer."
                      "These cookies are used to improve your website experience and provide more personalized services to you, both on this website and through other media.",
                      "To find out more about the cookies we use, see our Privacy Policy.",
                      "We won't track your information when you visit our site. But in order to comply with your preferences, we'll have to use just one tiny cookie so that you're not asked to make this choice again."
                      ]
  for string in frequent_strings:
    text = text.replace(string, ' ')
  return text


def remove_links_from_text(text: str) -> str:
  """
  Removes links from a text string.

  Args:
      text (str): The input text.

  Returns:
      str: The text with links removed.
  """
  text = re.sub(r'http\S+', ' ', text)
  return text


def expand_contractions(text: str) -> str:
  """
  Expands contractions in English text.

  Args:
      text (str): The text containing contractions.

  Returns:
      str: The text with contractions expanded.
  """

  # contractions dictionary removed for brevity
  contractions = {
    "I've": "I have",
    "i've": "i have",
    "Haven't": "Have not",
    "haven't": "have not",
    "Isn't": "Is not",
    "isn't": "is not",
    "We've": "We have",
    "we've": "we have",
    "Don't": "Do not",
    "don't": "do not",
    "Doesn't": "Does not",
    "doesn't": "does not",
    "Can't": "Cannot",
    "can't": "cannot",
    "Couldn't": "Could not",
    "couldn't": "could not",
    "Won't": "Will not",
    "won't": "will not",
    "Wouldn't": "Would not",
    "wouldn't": "would not",
    "Shouldn't": "Should not",
    "shouldn't": "should not",
    "Didn't": "Did not",
    "didn't": "did not",
    "Hasn't": "Has not",
    "hasn't": "has not",
    "Hadn't": "Had not",
    "hadn't": "had not",
    "Aren't": "Are not",
    "aren't": "are not",
    "Weren't": "Were not",
    "weren't": "were not",
    "It's": "It is",
    "it's": "it is",
    "That's": "That is",
    "that's": "that is",
    "Let's": "Let us",
    "let's": "let us",
    "You're": "You are",
    "you're": "you are",
    "They're": "They are",
    "they're": "they are",
    "We're": "We are",
    "we're": "we are",
    "Who's": "Who is",
    "who's": "who is",
    "What's": "What is",
    "what's": "what is",
    "Where's": "Where is",
    "where's": "where is",
    "When's": "When is",
    "when's": "when is",
    "Why's": "Why is",
    "why's": "why is",
    "How's": "How is",
    "how's": "how is",
    "She's": "She is",
    "she's": "she is",
    "He's": "He is",
    "he's": "he is",
    "It'll": "It will",
    "She'll": "She will",
    "she'll": "she will",
    "He'll": "He will",
    "he'll": "he will",
    "here's": "here is",
    "Here's": "Here is",
    "There's": "There is",
    "there's": "there is",
    "I'm": "I am",
    "i'm": "i am",
    "We'll": "We will",
    "we'll": "we will",
    "They'll": "They will",
    "they'll": "they will",
    "I'll": "I will",
    "i'll": "i will",
    "You'll": "You will",
    "you'll": "you will",
    "You've": "You have",
    "you've": "you have",
    "They've": "They have",
    "they've": "they have",
    "wasn't": "was not",
  }

  pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')

  def expand_match(contraction):
    match = contraction.group(0)
    expanded = contractions.get(match)
    if expanded:
      expanded += " "  # add a space after expanding the contraction
    return expanded

  expanded_text = pattern.sub(expand_match, text)
  expanded_text = re.sub(' +', ' ', expanded_text)  # remove any double spaces that might have been introduced

  return expanded_text


def put_placeholders(text: str, nlp) -> str:
  """
  Replaces numbers, emojis, and specific entities with placeholders.

  Args:
      text (str): The input text.
      nlp (spacy.Language): The Spacy language model for named entity recognition.

  Returns:
      str: The text with placeholders.
  """
  # Defines the emoji pattern.
  emoji_pattern = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)

  # Defines the entities of interest.
  """entities_of_interest = {
    'PERSON',
    'ORG',
    'GPE',
    'LOC',
    'PRODUCT',
    'EVENT',
    'LAW',
    'FAC',
    'NORP',
    'WORK_OF_ART'
  }"""

  # Replace numbers and emojis.
  text = emoji_pattern.sub(r'EMOJI', text)

  # Replace stray numbers.
  text = re.sub(r'\b\d+\b', 'NUMBER', text)

  # Replace different numbers with "NUMBER".
  doc = nlp(text)
  for ent in doc.ents:
    if ent.label_ in ["PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"]:
      text = text.replace(ent.text, 'NUMBER')
    elif ent.label_ == "DATE":
      text = text.replace(ent.text, 'DATE')
    elif ent.label_ == "TIME":
      text = text.replace(ent.text, 'TIME')

  # replace year periods, e.g. 1990s, 90s, 1990's, 90's
  text = re.sub(r"(?<!-)\b\d{2,4}'?s\b(?!\-)", 'DATE', text)

  # replace dates with patterns like 01/01/2000, 01/01/00, 01-01-2000, 01-01-00
  text = re.sub(r"\b(0[1-9]|1[0-2])[-/](0[1-9]|[12][0-9]|3[01])[-/](\d{2}|\d{4})\b", 'DATE', text)

  # replace time with patterns like 12:34:56, 12:34 (24 hours format), 12:34 PM/AM (12 hours format)
  text = re.sub(r"\b((1[0-2]|0?[1-9]):([0-5][0-9])(:[0-5][0-9])?(\s?[APap][Mm])?)\b", 'TIME', text)

  # Replace undesirable NER entities.
  # doc = nlp(text)
  # for ent in doc.ents:
  #   if ent.label_ not in entities_of_interest:
  #    text = text.replace(ent.text, 'ENTITY')

  return text


def final_cleanup(text: str) -> str:
  """
  Performs general cleanup of the text by removing leading/trailing whitespace, extra spaces, HTML tags,
  and email addresses. It also removes content within parentheses or square brackets.

  Args:
      text (str): The input text.

  Returns:
      str: The cleaned text.
  """
  text = re.sub(r'<.*?>', '', text)  # removes HTML tags
  text = re.sub(r'\S*@\S*\s?', '', text)  # removes email addresses

  text = re.sub(r'\s+\.$', '.', text)  # remove trailing spaces before period
  text = text.strip()  # remove leading and trailing whitespace
  text = re.sub(r'\s+', ' ', text)  # remove extra spaces

  return text


def remove_leading_placeholders(text):
  text = re.sub(r'\[\w+\]\s*:\s*', '', text)  # Matches "[ANYWORD]: " or "[ANYWORD] :"
  return text
