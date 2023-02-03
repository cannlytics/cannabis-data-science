"""
Open Cannabis Data
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 2/2/2023
Updated: 2/2/2023
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>

References:

    - OpenAI Tokenizer Tool
    URL: <https://platform.openai.com/tokenizer?view=bpe>

    - OpenAI GPT Models
    URL: <https://platform.openai.com/docs/models/gpt-3>

"""

# Standard imports:
import os

# External imports:
from dotenv import dotenv_values
import openai

# Get your OpenAI API key.
# URL: <https://platform.openai.com/account/api-keys>
config = dotenv_values('.env')
openai.api_key = config['OPENAI_API_KEY']
user = config.get('OPENAI_USER')
# openai.organization = 'org-xyz'


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
openai.Completion.create(
  model='text-davinci-003',
  prompt='Write a summary of terpinolene',
  max_tokens=600,
  temperature=0.9,
  n=1,
  best_of=3,
  finish_reason='stop',
#   frequency_penalty=0.0,
#   presence_penalty=1
#   top_p=0.1,
)

# Edit text.
openai.Edit.create(
  model="text-davinci-edit-001",
  input="What day of the wek is it?",
  instruction="Fix the spelling mistakes",
  temperature=0.9,
  n=1,
)

# TODO: Summarize text.
body_of_text = ''
prompt = body_of_text + '\n\ntl;dr'

# TODO: Create a table from text.
# A two-column spreadsheet
# of top science fiction movies and the year 
# Title |  Year of release

# TODO: Text to color.
prompt = """The CSS code for a color like a blue sky at dusk:

background-color: #"""

# TODO: Create a recipe for cannabis edibles given ingredients, a product,
# and the desired dose per serving. Get the serving size and number of servings
# per dish.
prompt = """Write a recipe based on these ingredients and instructions:

Frito Pie

Ingredients:
Fritos
Chili
Shredded cheddar cheese
Sweet white or red onions, diced small
Sour cream

Instructions:"""

# Create an image.
openai.Image.create(
  prompt='A cute baby sea otter',
  n=1,
  size='1024x1024'
)


# Edit an existing image.
image_file = 'flower.png'
mask_file = 'jar.png'
openai.Image.create_edit(
  image=open(image_file, 'rb'),
  mask=open(mask_file, 'rb'),
  prompt='A cannabis bud in a jar in space',
  n=1,
  size='1024x1024'
)

# Create a variation of an existing image.
image_file = 'flower.png'
openai.Image.create_variation(
  image=open(image_file, 'rb'),
  n=1,
  size='1024x1024'
)
