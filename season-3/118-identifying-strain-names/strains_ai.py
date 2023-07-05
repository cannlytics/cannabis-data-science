"""
StrainsAI
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 6/29/2023
Updated: 7/4/2023
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>
"""
# Standard imports:
import json
import os
from time import sleep

# External imports:
from dotenv import dotenv_values
import pandas as pd
import openai


def initialize_openai(openai_api_key = None) -> None:
    """Initialize OpenAI."""
    if openai_api_key is None:
        openai_api_key = os.environ['OPENAI_API_KEY']
    openai.api_key = openai_api_key


print('Testing strains AI...')

# Initialize OpenAI.
config = dotenv_values('../../.env')
openai_api_key = config['OPENAI_API_KEY']

# Identify strain names in a product name.
text = 'GARCIA HAND PICKED DARK KARMA'

# Define model parameters.
model = 'gpt-4'
max_tokens = 1_000
temperature = 0.0
user = 'cannlytics'
verbose = True
retry_pause:  3.33
instructional_prompt = 'Only return JSON and always return at least an empty array, e.g. {"strains": []}, if no strains can be found.'
identification_prompt = 'Given the following cannabis product name or text, what is the strain(s) of cannabis? Return your answer as JSON, e.g. {"strains": ["X"]} or {"strains": ["Strain 1", "Strain 2"]} where strains is a list of the strains, where sometimes multiple strains are indicated by a cross, e.g. "Strain 1 x Strain 2".'
json_key = 'strains'

# Format the message.
messages = [
    {'role': 'system', 'content': instructional_prompt},
    {'role': 'system', 'content': identification_prompt},
    {'role': 'user', 'content': f'Text: {text}\n\nStrains:'}
]

# Make the request to OpenAI. 
try:
    initialize_openai(openai_api_key)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        user=user,
    )
except:
    if retry_pause:
        sleep(retry_pause)
        initialize_openai(openai_api_key)
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            user=user,
        )

# Get the content of the response.
if verbose:
    print('RESPONSE:', response)
content = response['choices'][0]['message']['content']
start_index = content.find('{')
end_index = content.rfind('}') + 1

extracted_data = []
try:
    obj = json.loads(content[start_index:end_index])
    extracted_data.extend(obj[json_key])
except:
    if verbose:
        print('Failed to extract data.')

# View the extracted data.
print(extracted_data)
