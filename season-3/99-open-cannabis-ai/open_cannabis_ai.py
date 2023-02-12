"""
Open Cannabis AI
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 2/2/2023
Updated: 2/12/2023
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

# Create a completion prompt.
#
# 1. Select an appropriate GPT model:
#   - text-davinci-003: Complicated, long, high-quality tasks. Slowest. Highest cost.
#   - text-curie-001: Complex tasks. Fast. Lower cost than Davinci.
#   - text-babbage-001: Straightforward tasks. Faster. Lower cost than Curie.
#   - text-ada-001: Simple tasks. Fastest. Lowest cost.
#
# 2. Provide parameters:
#   - `max_tokens`: Provides a limit on the number of words to use.
#                   One token is approximately 4 characters of text.
#                   256 or more tokens are recommended per prompt.
#   - `temperature`: Use higher values (0 to 1) for more creativity.
#   - `top_p`: Use lower values (0 to 1) for more creativity.
#              It's suggested to alter either `top_p` or `temperature`, not both.
response = openai.Completion.create(
  model='text-davinci-003',
  prompt='Summary of Super Silver Haze',
  max_tokens=300,
  temperature=0.42,
  n=1,
#   best_of=3,
#   finish_reason='stop',
#   frequency_penalty=0.0,
#   presence_penalty=1
#   top_p=0.1,
)
print(response['choices'][0]['text'])

# Edit text.
response = openai.Edit.create(
  model='text-davinci-edit-001',
  input='Pesticide - Piperonyl Butoxide (ug/g)',
  instruction='analysis, compound, and units in JSON',
  n=1,
  temperature=0,
)
print(response['choices'][0]['text'])

# Find data in text.
prompt = 'analysis, compound, and units in JSON given:'
prompt += 'Pesticide - Piperonyl Butoxide (ug/g)'
print(prompt)
response = openai.Completion.create(
  model='text-davinci-003',
  prompt=prompt,
  max_tokens=100,
  n=1,
  temperature=0,
)
print(response['choices'][0]['text'])

# Summarize text.
body_of_text = response['choices'][0]['text']
prompt = body_of_text + '\n\ntl;dr'
print(prompt)
response = openai.Completion.create(
  model='text-davinci-003',
  prompt=prompt,
  max_tokens=100,
  n=1,
  temperature=0,
)
print(response['choices'][0]['text'])

# Create a table from text.
prompt = 'A two-column table of 5 '
prompt += 'terpenes found in cannabis and their boiling points'
prompt += '\nTerpene | Boiling point'
print(prompt)
response = openai.Completion.create(
  model='text-davinci-003',
  prompt=prompt,
  max_tokens=300,
  n=1,
  temperature=0,
)
print(response['choices'][0]['text'])

# Text to color.
prompt = 'Best color hex for: '
prompt += 'Super Silver Haze'
print(prompt)
response = openai.Completion.create(
  model='text-davinci-003',
  prompt=prompt,
  max_tokens=100,
  n=1,
  temperature=0,
)
print(response['choices'][0]['text'])

# Text to emoji.
prompt = 'Best emoji HTML hex for: '
prompt += 'Grape Ape'
print(prompt)
response = openai.Completion.create(
  model='text-davinci-003',
  prompt=prompt,
  max_tokens=100,
  n=1,
  temperature=0,
)
print(response['choices'][0]['text'])

# Create a recipe for cannabis edibles given ingredients, a product,
# and the desired dose per serving. Get the serving size and number of servings
# per dish.
prompt = """Write a recipe based on these ingredients and instructions:

cannabis oil
avocado
toast

Instructions:"""
print(prompt)
response = openai.Completion.create(
  model='text-davinci-003',
  prompt=prompt,
  max_tokens=100,
  n=1,
  temperature=0,
)
print(response['choices'][0]['text'])

# Create an image.
response = openai.Image.create(
  prompt='A small garden of homegrown cannabis',
  n=1,
  size='1024x1024',
)
image_url = response['data'][0]['url']
print(image_url)

# Edit an existing image.
image_file = 'imgs/plants.png'
mask_file = 'imgs/homebaking-mask.png'
response = openai.Image.create_edit(
  image=open(image_file, 'rb'),
  mask=open(mask_file, 'rb'),
  prompt='A journal for taking notes on cannabis',
  n=1,
  size='1024x1024'
)
image_url = response['data'][0]['url']
print(image_url)

# Create a variation of an existing image.
image_file = 'imgs/plants.png'
response = openai.Image.create_variation(
  image=open(image_file, 'rb'),
  n=1,
  size='1024x1024'
)
image_url = response['data'][0]['url']
print(image_url)
