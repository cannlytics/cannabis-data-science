"""
Open Cannabis AI | Prompt Engineering and Evaluation
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 2/2/2023
Updated: 3/29/2023
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>

References:

    - OpenAI Tokenizer Tool
    URL: <https://platform.openai.com/tokenizer?view=bpe>

    - OpenAI GPT Models
    URL: <https://platform.openai.com/docs/models/gpt-3>

"""
# External imports:
from dotenv import dotenv_values
import openai

# Get your OpenAI API key.
# URL: <https://platform.openai.com/account/api-keys>
config = dotenv_values('../../.env')
openai.api_key = config['OPENAI_API_KEY']

# TEST: Compare CoADoc to GPT-3.
import pdfplumber
from cannlytics.data.coas import CoADoc

# Read text from PDF.
text = pdfplumber.open('test_coa_2.pdf').pages[1].extract_text()

# Get data with CoA Doc.
parser = CoADoc()
data = parser.parse('test_coa.pdf')

# Define prompt.
prompt = f"""Find all cannabinoids percents as JSON given text: {text[:2000]}"""

# Create a completion prompt.
response = openai.Completion.create(
  model='text-davinci-003',
  prompt=prompt,
  max_tokens=3000,
  temperature=0.0,
  n=1,
)
print(response['choices'][0]['text'])
