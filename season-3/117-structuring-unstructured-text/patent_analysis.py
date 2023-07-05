"""
Patent Analysis
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 6/5/2023
Updated: 6/21/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data sources:

    - [Patent Public Search Basic](https://ppubs.uspto.gov/pubwebapp/static/pages/ppubsbasic.html)

"""
# Standard imports:
from datetime import datetime
import json
import os
import re
from typing import Optional
import zlib

# External imports:
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import openai
import pandas as pd
import pdfplumber
from pytesseract import image_to_string
import seaborn as sns
import tiktoken


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (6, 4),
    'font.family': 'Times New Roman',
    'font.size': 18,
})


#-----------------------------------------------------------------------
# Explore aromas mentioned in patents.
# E.g. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/PP34802
#-----------------------------------------------------------------------

# Read patent data.
datafile = 'cannabis-plant-patents-2023-06-06-15-24-39.xlsx'
patents = pd.read_excel(datafile)
patents.set_index('patent_number', drop=False)
print('Found %i plant patents.' % len(patents))

# Count plant patents over time.
patents['date'] = pd.to_datetime(patents['date_published'])
group = patents.groupby(pd.Grouper(key='date', freq='Y'), as_index=True)
per_year = group['patent_number'].count()

# Plot patents over time.
sample = per_year[per_year.index >= pd.to_datetime('2016-01-01')]
ax = sample.plot(kind='bar')
plt.xticks(rotation=0)
labels = [c.strftime('%Y') for c in sample.index]
ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
plt.gca().set_yticks(plt.gca().get_yticks()[1:])
plt.xlabel('')
plt.title('Number of Cannabis Plant Patents by Year', pad=18)
plt.show()


#-----------------------------------------------------------------------
# EXAMPLE: Extract data from downloaded patents.
#-----------------------------------------------------------------------

# Define the output directories.
pdf_dir = 'D://data/strains/patents/pdfs/'
image_dir = 'D://data/strains/patents/images/'

# Extract the text from patents.
extracted_data = {}
for index, patent in patents.iterrows():
    patent_number = patent['patent_number']
    original_file = os.path.join(pdf_dir, patent_number + '.pdf')
    pdf_image = os.path.join(image_dir, patent_number + '.png')

    # Check the number of pages.
    report = pdfplumber.open(original_file)
    page_count = len(report.pages)

    # Save the first page as an image.
    im = report.pages[0].to_image(resolution=90)
    im.save(pdf_image, format='PNG')

    # Close the report.
    report.close()

    # Read the text of the first page.
    page_text = image_to_string(pdf_image)
    abstract = page_text.split('ABSTRACT')[-1].split('\n\n')[0]

    # TODO: Get additional patent data points:
    # - patent_number_formatted
    # - patent_type
    # - patent_link
    # - patent_issue_date
    # - patent_year
    # - patent_month
    # - app_type
    # - app_filing_date
    # - inventors
    # - inventor_name
    # - inventor_city
    # - inventor_state
    # - strain_name
    # - key
    # - type
    # - description
    # - lineage
    # - average total cannabinoids
    # - average total thc
    # - average total cbd
    # - average total terpenes
    # - major terpenes
    # - THCV to THC ratio.
    # - THCV to total cannabinoid ratio.
    # - THC / CBD ratio.
    # - CBD / CBC / THC ratio.
    # - All terpene ratios!!!

    # Record the extracted data.
    obs = {'abstract': abstract}
    extracted_data[patent_number] = {**patent.to_dict(), **obs}
    print('Extracted data for patent:', patent_number)

# Save the extracted patent data.
date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
extracted = pd.DataFrame(extracted_data).T
extracted.to_excel(f'cannabis-plant-patents-{date}.xlsx')


#-----------------------------------------------------------------------
# Setup OpenAI for data modelling.
#-----------------------------------------------------------------------

# OpenAI API model prices (as of 2023-06-06) per 1000 tokens.
PRICE_PER_1000_TOKENS = {
    'gpt-4': {'prompt': 0.03, 'completion': 0.06},
    'gpt-3.5-turbo': {'prompt': 0.002, 'completion': 0.002},
    'ada': {'prompt': 0.0004, 'completion': 0.0004, 'training': 0.0004, 'usage': 0.0016},
    'babbage': {'prompt': 0.0005, 'completion': 0.0005, 'training': 0.0006, 'usage': 0.0024},
    'curie': {'prompt': 0.002, 'completion': 0.002, 'training': 0.003, 'usage': 0.012},
    'davinci': {'prompt': 0.02, 'completion': 0.02, 'training': 0.03, 'usage': 0.12},
    'dalle_1024': {'usage': 0.02},
    'dalle_512': {'usage': 0.018},
    'dalle_256': {'usage': 0.016},
    'whisper': {'usage': 0.006},
}


def num_tokens_from_messages(messages: list, model: Optional[str] = 'gpt-4'):
    """Returns the number of tokens used by a list of messages.
    Credit: OpenAI
    License: MIT <https://github.com/openai/openai-cookbook/blob/main/LICENSE>
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tokens_from_string(string: str, model: Optional[str] = 'gpt-4') -> int:
    """Returns the number of tokens in a text string."""
    return num_tokens_from_messages([{'role': 'user', 'content': string},], model=model)


def get_prompt_price(prompt, model='gpt-4', prices=PRICE_PER_1000_TOKENS):
    """Returns the price to generate a prompt."""
    num_tokens = num_tokens_from_string(prompt, model)
    return num_tokens / 1_000 * prices[model]['prompt']


# Initialize the OpenAi API.
config = dotenv_values('../../.env')
openai.api_key = config['OPENAI_API_KEY']


#-----------------------------------------------------------------------
# Have a conversation with all of the plant patents.
#-----------------------------------------------------------------------

# Specify the model to use.
model = 'gpt-4'
temperature = 0.0
max_tokens = 4_000

# Primer.
cost = 0
system_prompt = """Please listen while I tell you about of the cannabis plant patents.
Only respond with "OK" after I tell you about a patent. Once I say "FINISHED",
then respond with "READY" and I will ask you about the patents."""
messages = [{'role': 'system', 'content': system_prompt}]
cost += get_prompt_price(system_prompt, model=model)

# Keep track of generated responses.
generated_data = []

# Specify the patent details to use in the prompt.
details = ['strain_name', 'patent_number', 'patent_title', 
           'inventor_name', 'date_published', 'abstract']

# Isolate a sample of patents.
sample = patents[5:10]

# Populate memory.
for patent_number, obs in sample.iterrows():

    # Format prompt.
    prompt = """Cannabis plant patent details: """
    prompt += obs[details].to_json()

    # TODO: Try the same prompt with compressed data.
    # prompt += zlib.compress(obs[details].to_json().encode())

    # Add the prompt.
    messages.append({'role': 'user', 'content': prompt})

    # Estimate the cost of training.
    cost += get_prompt_price(prompt, model=model)

# Add the question.
messages.append({'role': 'user', 'content': 'READY'})
question = """Do any of the cannabis plant patents mention soil?"""
messages.append({'role': 'user', 'content': question})
cost += get_prompt_price(question + 'READY', model=model)
print('Expected cost of training ≈ $%f' % cost)

# Submit all prompts at the same time.
response = openai.ChatCompletion.create(
    model=model,
    temperature=temperature,
    max_tokens=max_tokens,
    messages=messages
)
content = response['choices'][0]['message']['content']  
print(content)

# Record the prompt.
generated_data.append({'prompt': prompt, 'response': content})

# Save the responses.
date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
responses = pd.DataFrame(generated_data)
try:
    responses.to_excel(f'.datasets/ai/prompts/ai-prompts-responses-{date}.xlsx')
except:
    print('Failed to save responses.')


# Lookup the patent to confirm.
patents.loc[patents['patent_number'] == 'US-PP34924-P3']['patent_url'].values[0]


#-----------------------------------------------------------------------
# Try to extract aroma data from patents.
#-----------------------------------------------------------------------

# Setup the prompt.
system_prompt = """Please return the following JSON if any aromas are mentioned in the abstract:

{"aromas": ["Aroma One", "Aroma Two", ...]}

Otherwise, return: {"aromas": []}"""

# Define a sample.
sample = patents[:5]

# Explore the abstracts with OpenAi's ChatGPT model.
aromas = {}
for patent_number, obs in sample.iterrows():

    # Count the number of words.
    print('%s Words in abstract:' % patent_number, len(obs['abstract'].split(' ')))

    # Try to extract aromas from patents.
    prompt = """Abstract: """ + obs['abstract']
    response = openai.ChatCompletion.create(
        model='gpt-4',
        temperature=0,
        max_tokens=1000,
        messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
            ]
    )
    content = response['choices'][0]['message']['content']  
    print(content)

    # Clean the JSON afterwards.
    start_index = content.find('{')
    end_index = content.rfind('}') + 1
    extracted_data = json.loads(content[start_index:end_index])

    # Record the generated data.
    aromas[patent_number] = extracted_data['aromas']


# Lookup the patent to confirm.
patents.loc[patents['patent_number'] == 'US-PP34828-P2']['patent_url'].values[0]

# Save the responses.
date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
try:
    responses = pd.DataFrame([aromas]).T
    responses.columns = ['aromas']
    responses.to_excel(f'.datasets/ai/prompts/ai-prompts-responses-{date}.xlsx')
except:
    print('Failed to save responses.')


#-----------------------------------------------------------------------
# TODO: Visualize the extracted patent data.
#-----------------------------------------------------------------------
