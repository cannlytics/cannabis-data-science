"""
Cultivator Analysis | Flow Gardens
Copyright (c) 2024 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 1/28/2024
Updated: 2/7/2024
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Source:

    - [Flow Gardens](https://flowgardens.com/)

"""
# Standard imports:
import ast
import base64
import json
import os
import re
from time import sleep

# External imports:
from adjustText import adjust_text
from bs4 import BeautifulSoup
import cv2
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import numpy as np
from dotenv import dotenv_values
import requests
import pandas as pd
from pandas import Timestamp
from openai import OpenAI
import seaborn as sns
from skimage import color
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import spacy
from statistics import mean
from textacy.extract import ngrams


# === Setup ===

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})


# === Get Flow Gardens COAs ===
        
def get_flowgardens_coas(
        data_dir='./',
        pause = 3.33,
    ) -> pd.DataFrame:

    # Get the Flow Gardens website.
    base = 'https://flowgardens.com/flower'
    response = requests.get(base)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

    # Get the product URLs.
    product_list = soup.find_all('li', class_='product')

    # Get the number of pages.
    pagination = soup.find('a', class_='pagination-link')
    number_of_pages = pagination.get('aria-label', '')
    match = re.search(r'Page \d+ of (\d+)', number_of_pages)
    if match:
        total_pages = int(match.group(1))
    else:
        total_pages = None

    # Get the products from each page.
    for page in range(2, total_pages + 1):
        sleep(pause)
        url = f'{base}/?page={page}'
        response = requests.get(url)
        if response.status_code == 200:
            print(f'Getting products from: {url}')
            soup = BeautifulSoup(response.text, 'html.parser')
            product_list.extend(soup.find_all('li', class_='product'))

    # Get the product URLs.
    products = []
    for product in product_list:
        product_url = product.find('a').get('href')
        img_tag = product.find('img', class_='lazyload card-image')
        strain_name = product.find('h3').text.replace('\n', '').strip()
        strain_image = img_tag.get('src', '')
        products.append({
            'product_url': product_url,
            'strain_name': strain_name,
            'strain_image': strain_image,
        })

    # Get the details for each product.
    product_data = []
    for product in products:
        sleep(pause)
        product_url = product['product_url']
        strain_name = product['strain_name']
        strain_image = product['strain_image']
        if 'sample-pack' in product_url or 'chillum' in product_url:
            continue
        response = requests.get(product_url)
        if response.status_code == 200:
            print(f'Getting data from: {product_url}')
            soup = BeautifulSoup(response.text, 'html.parser')
            product_view = soup.find('dl', class_='productView-info').text
            try:
                strain_type = product_view.split('Type:')[1].split('\n')[1]
            except:
                print(f'Could not get strain type for: {product_url}')
                continue
            description = soup.find('div', class_='productView-description').text
            description = description.lstrip('\n').rstrip('\n\xa0\n')
            thumbnails = soup.find_all('li', class_='productView-thumbnail')
            image_urls = []
            for thumbnail in thumbnails:
                image_url = thumbnail.find('img').get('src')
                image_urls.append(image_url)

            # Get reviews.
            reviews = soup.find_all('li', class_='productReview')
            review_data = []
            for review in reviews:
                # Extract rating.
                rating = int(review.find('span', class_='productReview-ratingNumber').text.strip())

                # Extract review title (subject).
                subject = review.find('h5', class_='productReview-title').text.strip()

                # Extract review body (comment).
                comment = review.find('p', class_='productReview-body').text.strip()

                # Extract author and date.
                author_date = review.find('p', class_='productReview-author').text.strip()
                try:
                    author, date = author_date.split(' on ')
                    channel = 'website'
                except:
                    author, channel, date = author_date.split('on ')
                author = author.replace('Posted by ', '')
                date = pd.to_datetime(date)
                
                review_data.append({
                    'rating': rating,
                    'subject': subject,
                    'comment': comment,
                    'user': author,
                    'date': date,
                    'channel': channel,
                })
            
            # Record the product data.
            product_data.append({
                'strain_name': strain_name,
                'strain_type': strain_type,
                'description': description,
                'image_urls': image_urls,
                'product_url': product_url,
                'reviews': review_data,
            })

    # Save the product data.
    timestamp = pd.to_datetime('now').strftime('%Y-%m-%d')
    outfile = f'{data_dir}/flowgardens-products-{timestamp}.xlsx'
    df = pd.DataFrame(product_data)
    df.to_excel(outfile, index=False)
    return df


# Get Flow Gardens COAs.
data_dir = 'D://data/work/flowgardens/datasets'
report_dir = './cultivator-report/stats'
assets_dir = './cultivator-report/assets/figures'
coa_data = get_flowgardens_coas(data_dir=data_dir)


# === Parse COAs ===

# Base prompt.
INSTRUCTIONAL_PROMPT = 'You are a certificate of analysis (COA) parser designed to extract data from COA text. Only return JSON and always return at least an empty object, {}, if no data can be found. Return a value of `null` for any field that cannot be found.'
NOTES = """Note: Data was extracted from images of COAs using OpenAI's GPT models and may include incorrect values."""


def download_coa_image(url, image_dir):
    """Download a COA image."""
    filename = url.split('/')[-1].replace('?c=1', '')
    coa_image = os.path.join(image_dir, filename)
    if not os.path.exists(coa_image):
        response = requests.get(url)
        with open(coa_image, 'wb') as doc:
            doc.write(response.content)
    return coa_image


def encode_image(image_path):
        """Encode an image as a base64 string."""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        

def extract_completion_json(completion):
    """Extract JSON from a OpenAI completion."""
    content = completion.choices[0].message.content
    extracted_json = content.lstrip('```json\n').split('\n```')[0]
    try:
        extracted_data = json.loads(extracted_json)
    except:
        try:
            if not content.endswith('"'):
                content += '"'
            extracted_data = json.loads(extracted_json + '}')
        except:
            extracted_data = json.loads(','.join(extracted_json.split(',')[:-1]) + '}')
    return extracted_data


def extract_coa_metadata(
        client,
        coa,
        model = 'gpt-4-vision-preview', # Alt: gpt-4-0125-preview
        detail = 'high',
        max_tokens = 4_096,
        temperature = 0.0,
        user = 'cannlytics',  
        method = 'image',
        verbose = True,
    ) -> dict:
    if method == 'image':
        extraction_prompt = 'Given the attached image of a COA'
    else:
        extraction_prompt = 'Given text from a COA'
    extraction_prompt += """, extract JSON, where:\n
    | Field | Example | Description |
    |-------|---------|-------------|
    | `product_name` | "Blue Rhino Pre-Roll" | The name of the product. |
    | `product_type` | "flower" | The type of product. |
    | `sample_id` | "Sample-0001" | A lab-specific ID for the sample. |
    | `sample_received` | 2022-04-20T16:20 | An ISO-formatted time when the sample was received. |
    | `report_created` | 2022-04-20T16:20 | An ISO-formatted time when the report was created. |
    | `total_cannabinoids` | 14.20 | The analytical total of all cannabinoids measured. |
    | `total_thc` | 14.00 | The analytical total of THC and THCA. |
    | `total_cbd` | 0.20 | The analytical total of CBD and CBDA. |
    """
    if method == 'image':
        base64_image = encode_image(coa)
    else:
        material_prompt = 'COA text:\n\n' + coa
    messages = [
        {'role': 'system', 'content': INSTRUCTIONAL_PROMPT},
        {'role': 'system', 'content': extraction_prompt},
    ]
    if method == 'image':
        messages.append(
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{base64_image}',
                        'detail': detail,
                    },
                },
            ],
        })
    else:
        messages.append({'role': 'user', 'content': material_prompt})
    if verbose:
        print('\n\n'.join([x.get('content') for x in messages if isinstance(x.get('content'), str)]))
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        user=user,
    )
    if verbose:
        print('RESPONSE:', json.dumps(completion.dict()))
    return extract_completion_json(completion)


def extract_coa_cannabinoids(
        client,
        coa,
        model = 'gpt-4-vision-preview',
        detail = 'high',
        user = 'cannlytics',
        max_tokens = 4_096,
        temperature = 0.0,
        method = 'image',
        verbose = True,
        analytes = [],
        normalize = None,
    ) -> dict:
    if method == 'image':
        extraction_prompt = 'Given the attached image of a COA'
    else:
        extraction_prompt = 'Given text from a COA'
    extraction_prompt += """, extract JSON, where:\n
    | Field | Example | Description |
    |-------|---------|-------------|
    | `{cannabinoid}` | 0.20 | The percent (%) value for the cannabinoid. Acceptable values include "ND" and "<LOQ". |

    \nPlease use "snake_case" for fields, e.g. "Delta-9-Tetrahydrocannabinol (Δ-9 THC)" as "delta_9_thc".

    \nPlease make sure to use the percent value.

    \nPlease include all of the cannabinoids that you see. Below is a list of possible cannabinoids that may be encountered.

    \nDo not apply the decarboxylation rate (0.877) to any of the values.
    """
    extraction_prompt += '\n' + '\n'.join(analytes)
    if method == 'image':
        base64_image = encode_image(coa)
    else:
        material_prompt = 'COA text:\n' + coa
    messages = [
        {'role': 'system', 'content': INSTRUCTIONAL_PROMPT},
        {'role': 'system', 'content': extraction_prompt},
    ]
    if method == 'image':
        messages.append(
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{base64_image}',
                        'detail': detail,
                    },
                },
            ],
        })
    else:
        messages.append({'role': 'user', 'content': material_prompt})
    if verbose:
        print('\n\n'.join([x.get('content') for x in messages if isinstance(x.get('content'), str)]))
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        user=user,
    )
    if verbose:
        print('RESPONSE:', json.dumps(completion.dict()))
    
    # Format the completion.
    extracted_data = extract_completion_json(completion)

    # Calculate `sum_of_cannabinoids` as a double check.
    sum_of_cannabinoids = 0
    for key, value in extracted_data.items():
        if 'total' in key:
            continue
        try:
            sum_of_cannabinoids += float(value)
        except:
            pass
    extracted_data['sum_of_cannabinoids'] = sum_of_cannabinoids

    # Return the extracted data.
    return extracted_data


def extract_coa_terpenes(
        client,
        coa,
        model = 'gpt-4-vision-preview',
        detail = 'high',
        user = 'cannlytics',
        max_tokens = 4_096,
        temperature = 0.0,
        method = 'image',
        units = 'mg/g',
        normalize = 10,
        analysis = 'terpenes',
        example_name = 'Caryophyllene Oxide',
        example_key = 'caryophyllene_oxide',
        analytes = [],
        verbose = True,
    ) -> dict:
    if method == 'image':
        extraction_prompt = 'Given the attached image of a COA'
    else:
        extraction_prompt = 'Given text from a COA'
    extraction_prompt += """, extract JSON, where:

    | Field | Example | Description |
    |-------|---------|-------------|
    | `total_{analysis}` | 0.42 | The sum of all {analysis} measured. |
    | `{{terpene}}` | 0.20 | The {units} value for the {analysis}. Acceptable values include "ND" and "<LOQ". |
    | `primary_aromas` | ["earthy", "sweet", "gas"] | The primary aromas of the sample. |

    \nPlease use "snake_case" for fields, e.g. "{example_name}" as "{example_key}".

    \nPlease make sure to use the {units} value.

    \nPlease include all of the {analysis} that you see. Below is a list of possible cannabinoids that may be encountered.
    """.format(
        units=units,
        analysis=analysis,
        example_name=example_name,
        example_key=example_key,
    )
    extraction_prompt += '\n' + '\n'.join(analytes)
    if method == 'image':
        base64_image = encode_image(coa)
    else:
        material_prompt = 'COA text:\n' + coa
    messages = [
        {'role': 'system', 'content': INSTRUCTIONAL_PROMPT},
        {'role': 'system', 'content': extraction_prompt},
    ]
    if method == 'image':
        messages.append(
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{base64_image}',
                        'detail': detail,
                    },
                },
            ],
        })
    else:
        messages.append({'role': 'user', 'content': material_prompt})
    if verbose:
        print('\n\n'.join([x.get('content') for x in messages if isinstance(x.get('content'), str)]))
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        user=user,
    )
    if verbose:
        print('RESPONSE:', json.dumps(completion.dict()))

    # Format the completion.
    extracted_data = extract_completion_json(completion)

    # Convert mg/g to percent.
    if normalize:
        for key, value in extracted_data.items():
            try:
                extracted_data[key] = float(value) / normalize
            except:
                pass

    # Calculate `total_terpenes` as a double check.
    total_terpenes = 0
    for key, value in extracted_data.items():
        if 'total' in key:
            continue
        try:
            total_terpenes += float(value)
        except:
            pass
    extracted_data['sum_of_terpenes'] = total_terpenes

    # Return the extracted data.
    return extracted_data


# Define cannabinoids being sought.
specific_cannabinoids = [
    'cbc',
    'cbca',
    'cbd',
    'cbda',
    'cbg',
    'cbga',
    'cbn',
    'cbna',
    'delta_8_thc',
    'delta_9_thc',
    'thca',
    'thcv',
    'thcva',
    # 'thcp',
    # 'r_delta_10_thc',
    # 's_delta_10_thc',
    # '9r_hhc',
    # '9s_hhc',
    # 'thco',
]

# Define terpenes being sought.
specific_terpenes = [
    'alpha_bisabolol',
    'alpha_humulene',
    'alpha_pinene',
    'alpha_terpinene',
    'cineole',
    'beta_caryophyllene',
    'beta_myrcene',
    'borneol',
    'camphene',
    'delta_3_carene',
    'caryophyllene_oxide',
    'citral',
    'dihydrocarveol',
    'fenchone',
    'gamma_terpinene',
    'd_limonene',
    'linalool',
    'menthol',
    'nerolidol',
    'ocimene',
    'pulegone',
    'terpinolene',
]

# Initialize OpenAI.
config = dotenv_values('../../.env')
openai_api_key = config['OPENAI_API_KEY']
client = OpenAI()

# Image properties.
image_dir = 'D:/data/work/flowgardens/images'
image_size = '7200x7200'

# Download the images of the cannabinoids and terpenes COAs.
all_extracted_data = []
for index, obs in coa_data.iterrows():
    print('Extracting data for:', obs['strain_name'])

    # Download the cannabinoid COA image.
    try:
        cannabinoid_image_url = [x for x in obs['image_urls'] if 'COA' in x][0]
        cannabinoid_image_url = cannabinoid_image_url.replace('50x50', image_size)
        cannabinoid_coa = download_coa_image(cannabinoid_image_url, image_dir)
    except:
        print('No cannabinoid COA found.')
        cannabinoid_coa = None

    # Download the terpene COA image.
    sleep(3.33)
    try:
        terpene_image_url = [x for x in obs['image_urls'] if 'Terp' in x][0]
        terpene_image_url = terpene_image_url.replace('50x50', image_size)
        terpene_coa = download_coa_image(terpene_image_url, image_dir)
    except:
        print('No terpene COA found.')
        terpene_coa = None

    # Continue if no COAs are attached.
    if not cannabinoid_coa or not terpene_coa:
        continue

    # Optional: Extract the text from the COA images.
    # obs['cannabinoid_text'] = pytesseract.image_to_string(cannabinoid_coa, config='--oem 3 --psm 6')
    # obs['terpene_text'] = pytesseract.image_to_string(terpene_coa, config='--oem 3 --psm 6')
    # obs['metadata_text'] = obs['cannabinoid_text']

    # Extract COA metadata.
    if cannabinoid_coa:
        metadata = extract_coa_metadata(client, cannabinoid_coa)
    else:
        metadata = extract_coa_metadata(client, terpene_coa)
    obs = pd.Series({**obs, **metadata})

    # Extract cannabinoid data.
    if cannabinoid_coa:
        sleep(3.33)
        cannabinoid_data = extract_coa_cannabinoids(
            client,
            cannabinoid_coa,
            analytes=specific_cannabinoids,
        )
        obs = pd.Series({**obs, **cannabinoid_data})

    # Extract terpene data.
    if terpene_coa:
        sleep(3.33)
        terpene_data = extract_coa_terpenes(
            client,
            terpene_coa,
            analytes=specific_terpenes,
        )
        obs = pd.Series({**obs, **terpene_data})

    # Optional: Get total_terpenes and primary_aromas in a separate call.

    # Record the extracted data.
    all_extracted_data.append(obs.to_dict())

# Rename fields of the extracted data.
extraction = pd.DataFrame(all_extracted_data)
extraction.rename(columns={
    'report_created': 'date_tested',
    'sample_received': 'date_received',
    'sample_id': 'lab_id',
    'primary_aromas': 'predicted_aromas',
}, inplace=True)

# Save all of the extracted data.
timestamp = pd.to_datetime('now').strftime('%Y-%m-%d-%H-%M-%S')
outfile = f'{data_dir}/flowgardens-coa-data-{timestamp}.xlsx'
extraction.to_excel(outfile, index=False)


# === Format Reviews ===

# Format reviews.
reviews = []
for index, obs in coa_data.iterrows():
    try:
        strain_reviews = eval(obs['reviews'])
    except:
        strain_reviews = obs['reviews']
    for review in strain_reviews:
        reviews.append({**obs.to_dict(), **review})

# Save all of the reviews.
reviews = pd.DataFrame(reviews)
timestamp = pd.to_datetime('now').strftime('%Y-%m-%d-%H-%M-%S')
outfile = f'{data_dir}/flowgardens-reviews-{timestamp}.xlsx'
reviews.to_excel(outfile, index=False)


# === Summary Statistics ===

# Calculate the total number of strains.
total_strains = len(reviews['strain_name'].unique())

# Calculate the total number of reviews.
total_reviews = len(reviews)

# Count the number of parsed COAs.
total_parsed_coas = len(extraction)

# Output summary statistics to LaTeX.
summary = pd.DataFrame({
    'Total Strains': [total_strains],
    'Total Reviews': [total_reviews],
    'Total Parsed COAs': [total_parsed_coas],
})
output = summary.T
output.columns = ['Observations']
# output.to_latex(f'{report_dir}/summary.tex', index=True, header=False)
latex_code = output.to_latex(index=True, header=False)
latex_code = latex_code.replace('\n\\midrule', '')
with open(f'{report_dir}/summary.tex', 'w') as file:
    file.write(latex_code)

# Visualize strain types.
plt.figure(figsize=(14, 8.5))
colors = ["#78C850", "#F85888", "#F8D030", "#6890F0", "#705898"]
n_bins = 100
cmap_name = "strain_gradient"
strain_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
order = ['Sativa Dominant', 'Sativa', 'Hybrid', 'Indica', 'Indica Dominant']
coa_data['strain_type'] = coa_data['strain_type'].str.title()
sns.countplot(
    x='strain_type',
    data=coa_data,
    order=order,
    palette=np.array(strain_cmap(np.linspace(0, 1, len(order)))),
)
plt.xlabel('')
plt.ylabel('Count')
plt.title('Count of Strain Types')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(f'{assets_dir}/strain-types.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Visualize the distribution of ratings.
plt.figure(figsize=(15, 8))
ratings_distribution = reviews['rating'].value_counts().sort_index()
sns.barplot(
    x=ratings_distribution.index,
    y=ratings_distribution.values,
    palette="autumn"
)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'{assets_dir}/ratings.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Calculate the number of reviews and average rating for each strain in 'reviews'
strain_reviews_summary = reviews.groupby('strain_name').agg(
    Strain_Type=pd.NamedAgg(column='strain_type', aggfunc='first'),
    Number_of_Reviews=pd.NamedAgg(column='rating', aggfunc='count'),
    Average_Rating=pd.NamedAgg(column='rating', aggfunc='mean')
).reset_index()
strain_reviews_summary['COA Data'] = strain_reviews_summary['strain_name'].isin(extraction['strain_name']).map({True: '\\checkmark', False: ''})
strain_reviews_summary = strain_reviews_summary.sort_values(by='strain_name')
latex_table = strain_reviews_summary.to_latex(
    index=False,
    column_format='lcccc',
    header=['Strain Name', 'Strain Type', 'Number of Reviews', 'Average Rating', 'Parsed COA Data'],
    formatters={
        'Average_Rating': lambda x: f"{x:.1f}"
    },
    escape=False
)
latex_table = latex_table.replace('#', '\#')
print(latex_table)
with open(f'{report_dir}/strains.tex', 'w') as f:
    f.write(latex_table)


# === NLP Analysis ===

# Initialize the spacy model.
nlp = spacy.load('en_core_web_lg')
sia = SIA()

# Compile all of the reviews into a single corpus.
texts = reviews['comment'].dropna().tolist()
corpus = '. '.join(texts)
doc = nlp(corpus)

# Identify unique unigrams, bi-grams, trigrams to use as strain names.
unigrams = list(set([x.text for x in ngrams(doc, 1, min_freq=2)]))
bigrams = list(set([x.text for x in ngrams(doc, 2, min_freq=2)]))
trigrams = list(set([x.text for x in ngrams(doc, 3, min_freq=2)]))
print('Unique unigrams:', len(unigrams))
print('Unique bigrams:', len(bigrams))
print('Unique trigrams:', len(trigrams))

# Create a frequency distribution, normalizing all words to lowercase.
words = [w.lower() for w in corpus.split() if w.isalpha()]
stopwords = nltk.corpus.stopwords.words('english')
words = [w for w in words if w not in stopwords]
fd = nltk.FreqDist(words)
fd.tabulate(20)


def assign_sentiment(review_text):
    """ Analyze sentiment of a review text """
    return sia.polarity_scores(review_text)['compound']


def avg_positivity(text: str) -> bool:
    """Average of all sentence compound scores."""
    scores = [
        sia.polarity_scores(sentence)['compound']
        for sentence in nltk.sent_tokenize(text)
    ]
    try:
        return mean(scores)
    except:
        return 0


# Assign sentiment to each review.
reviews['sentiment'] = reviews['comment'].apply(avg_positivity)

# Assign sentiment to the descriptions.
reviews['description_sentiment'] = reviews['description'].apply(avg_positivity)

# Optional: Create a scatterplot of `sentiment` vs. other variables.
# sns.scatterplot(
#     x='description_sentiment',
#     y='sentiment',
#     hue='strain_name',
#     data=reviews,
#     legend=None,
# )


# === Effects Analysis ===

def extract_aromas(
        client,
        text,
        model = 'gpt-4-0125-preview',
        max_tokens = 4_096,
        temperature = 0.0,
        user = 'cannlytics',  
        verbose = True,
    ):
    extraction_prompt = """Given text, return any mentioned aromas in JSON. E.g

{"aromas": ["Aroma One", "Aroma Two", ...]}

Otherwise, return: {"aromas": []}"""
    user_prompt = 'Extract `aromas` JSON from the following text:\n\n' + text
    messages = [
        {'role': 'system', 'content': extraction_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        user=user,
    )
    if verbose:
        print('RESPONSE:', json.dumps(completion.dict()))
    return extract_completion_json(completion)


def extract_effects(
        client,
        text,
        model = 'gpt-4-0125-preview',
        detail = 'high',
        max_tokens = 4_096,
        temperature = 0.0,
        user = 'cannlytics',  
        verbose = True,
    ):
    extraction_prompt = """Given text, return any mentioned effects in JSON. E.g

{"effects": ["Effect One", "Effect Two", ...]}

Otherwise, return: {"effects": []}"""
    user_prompt = 'Extract `effects` JSON from the following text:\n\n' + text
    messages = [
        {'role': 'system', 'content': extraction_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        user=user,
    )
    if verbose:
        print('RESPONSE:', json.dumps(completion.dict()))
    return extract_completion_json(completion)


def standardize_aromas(
        client,
        text,
        model = 'gpt-4-0125-preview',
        max_tokens = 4_096,
        temperature = 0.0,
        user = 'cannlytics',  
        verbose = True,
        aromas = [],
    ):
    extraction_prompt = """Given a list of aromas, return a standardized list of aromas, combining highly similar entries.

E.g. given:

{"aromas": ["Strawberry", "strawberries", "Hot Pepper",...]}

Return:

{"aromas": ["strawberry", "hot-pepper"]}"""
    if aromas:
        extraction_prompt += '\n\nHere is a list of known aromas:\n' + '\n'.join(aromas)
    user_prompt = 'Standardize `aromas` JSON given the following list:\n\n' + str(text)
    messages = [
        {'role': 'system', 'content': extraction_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        user=user,
    )
    if verbose:
        print('RESPONSE:', json.dumps(completion.dict()))
    return extract_completion_json(completion)


def standardize_effects(
        client,
        text,
        model = 'gpt-4-0125-preview',
        max_tokens = 4_096,
        temperature = 0.0,
        user = 'cannlytics',  
        verbose = True,
        aromas = [],
    ):
    extraction_prompt = """Given a list of effects, return a standardized list of aromas, combining highly similar entries.

E.g. given:

{"effects": ["full-body relaxation"...]}

Return:

{"effects": ["body-high", "relaxed"]}"""
    if aromas:
        extraction_prompt += '\n\nHere is a list of known effects:\n' + '\n'.join(aromas)
    user_prompt = 'Standardize `effects` JSON given the following list:\n\n' + str(text)
    messages = [
        {'role': 'system', 'content': extraction_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        user=user,
    )
    if verbose:
        print('RESPONSE:', json.dumps(completion.dict()))
    return extract_completion_json(completion)


# Extract aromas and effects.
all_aromas, all_effects = [], []
for index, obs in reviews.iterrows():
    print('Extracting aromas and effects for:', obs['strain_name'])

    # Extract aromas.
    sleep(1)
    aromas = extract_aromas(client, obs['comment'])
    all_aromas.append(aromas['aromas'])

    # Extract effects.
    sleep(1)
    effects = extract_effects(client, obs['comment'])
    all_effects.append(effects['effects'])

# Assign aromas and effects to the reviews.
reviews['aromas'] = all_aromas
reviews['effects'] = all_effects

# Find all unique aromas and effects.
unique_aromas = list(set([x for y in all_aromas for x in y['aromas']]))
unique_effects = list(set([x for y in all_effects for x in y['effects']]))

# Effects NLP analysis.
texts = reviews['effects'].apply(lambda x: ', '.join(x)).tolist()
texts = [x for x in texts if x]
corpus = '. '.join(texts)
doc = nlp(corpus)
unigrams = list(set([x.text for x in ngrams(doc, 1, min_freq=2)]))
bigrams = list(set([x.text for x in ngrams(doc, 2, min_freq=2)]))
trigrams = list(set([x.text for x in ngrams(doc, 3, min_freq=2)]))
words = [w.lower() for w in corpus.split() if w.isalpha()]
stopwords = nltk.corpus.stopwords.words('english')
words = [w for w in words if w not in stopwords]
fd = nltk.FreqDist(words)
fd.tabulate(20)

# Aromas NLP analysis.
texts = reviews['aromas'].apply(lambda x: ', '.join(x)).tolist()
texts = [x for x in texts if x]
corpus = '. '.join(texts)
doc = nlp(corpus)
unigrams = list(set([x.text for x in ngrams(doc, 1, min_freq=2)]))
bigrams = list(set([x.text for x in ngrams(doc, 2, min_freq=2)]))
trigrams = list(set([x.text for x in ngrams(doc, 3, min_freq=2)]))
words = [w.lower() for w in corpus.split() if w.isalpha()]
stopwords = nltk.corpus.stopwords.words('english')
words = [w for w in words if w not in stopwords]
fd = nltk.FreqDist(words)
fd.tabulate(20)

# Define known aromas.
known_aromas = [
    "berries",
    "grape",
    "strawberry",
    "spicy",
    "cherry",
    "vanilla",
    "pine",
    "piney",
    "pinecones",
    "floral",
    "citrus",
    "fruity",
    "orange",
    "fruit",
    "lemon",
    "grapefruit",
    "blueberry",
    "skunk",
    "skunky",
    "apricot",
    "earthy",
    "sweet",
    "gas",
    "gassy",
]

# Define known effects.
known_effects = [
    "pain relief",
    "stress relief",
    "heady",
    "head high",
    "uplifting",
    "couch lock",
    "body high",
    "euphoric",
    "sleep",
    "relaxed",
    "relaxing",
    "energetic",
    "burns",
    "paranoid",
    "induced stress",
]

# TODO: Standardize effects and aromas.


# TODO: Frequency of certain effects/aromas etc.


# TODO: Look at the frequency of words like "smell", "aroma", "taste" (potentially by strain).


def calculate_percentage(df, column_name):
    """Function to calculate percentage of reviews mentioning any effect or aroma."""
    df[f'{column_name}_mentioned'] = df[column_name].apply(lambda x: 1 if len(x) > 0 else 0)
    summary = df.groupby('strain_name').agg(
        total_reviews=pd.NamedAgg(column='strain_name', aggfunc='count'),
        mentions=pd.NamedAgg(column=f'{column_name}_mentioned', aggfunc='sum')
    )
    summary[f'{column_name}_percentage'] = (summary['mentions'] / summary['total_reviews']) * 100
    return summary[[f'{column_name}_percentage']]


# Calculate percentages
effects_percentage = calculate_percentage(reviews, 'effects')
aromas_percentage = calculate_percentage(reviews, 'aromas')

# Join the two dataframes if you want them together
percentage_summary = effects_percentage.join(aromas_percentage, how='outer')
print(percentage_summary)

# Visualize the percentage of reviews mentioning aromas and effects by strain
n_reviews_per_strain = reviews.groupby('strain_name')['comment'].count()
percentage_summary_reset = percentage_summary.reset_index()
effects_data = percentage_summary_reset[['strain_name', 'effects_percentage']].copy()
effects_data['Type'] = 'Effects'
aromas_data = percentage_summary_reset[['strain_name', 'aromas_percentage']].copy()
aromas_data.rename(columns={'aromas_percentage': 'effects_percentage'}, inplace=True)
aromas_data['Type'] = 'Aromas'
combined_data = pd.concat([effects_data, aromas_data])
combined_data.rename(columns={'effects_percentage': 'Percentage'}, inplace=True)
effects_data_sorted = effects_data.sort_values(by='effects_percentage', ascending=False)
aromas_data_sorted = aromas_data.sort_values(by='effects_percentage', ascending=False)  # Note: effects_percentage column in aromas_data holds aroma percentages
combined_data_sorted = pd.concat([effects_data_sorted, aromas_data_sorted])
plt.figure(figsize=(15.5, 25))
plt.subplot(2, 1, 1)
effects_plot = sns.barplot(
    x='effects_percentage',
    y='strain_name',
    data=effects_data_sorted,
    color='skyblue',
)
effects_plot.set_title('Percentage of Reviews Mentioning Effects')
effects_plot.set_xlabel('Percentage')
effects_plot.set_ylabel('')
plt.subplot(2, 1, 2)
aromas_plot = sns.barplot(
    x='effects_percentage', # Note: Using 'effects_percentage' as it was renamed for aroma percentages
    y='strain_name',
    data=aromas_data_sorted,
    color='#ffb07c',
)
aromas_plot.set_title('Percentage of Reviews Mentioning Aromas')
aromas_plot.set_xlabel('Percentage')
aromas_plot.set_ylabel('')
plt.tight_layout()
plt.savefig(f'{assets_dir}/aromas-effects.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


def extract_classification(comment):
    """Extract mention of "sativa" / "indica" by strain."""
    mentions = {'sativa': 0, 'indica': 0}
    comment_lower = comment.lower()
    if 'sativa' in comment_lower:
        mentions['sativa'] = 1
    if 'indica' in comment_lower:
        mentions['indica'] = 1
    return pd.Series(mentions)


# Apply the function to each comment
mentions_df = reviews['comment'].apply(extract_classification)
reviews = pd.concat([reviews, mentions_df], axis=1)

# Step 2: Aggregate data by strain_name and strain_type
aggregated_data = reviews.groupby(['strain_name', 'strain_type']).agg(
    total_reviews=pd.NamedAgg(column='comment', aggfunc='count'),
    sativa_mentions=pd.NamedAgg(column='sativa', aggfunc='sum'),
    indica_mentions=pd.NamedAgg(column='indica', aggfunc='sum')
).reset_index()

# Calculate percentages of mentions
aggregated_data['sativa_mentions_pct'] = (aggregated_data['sativa_mentions'] / aggregated_data['total_reviews']) * 100
aggregated_data['indica_mentions_pct'] = (aggregated_data['indica_mentions'] / aggregated_data['total_reviews']) * 100

# Analysis of mentions vs. classification.
aggregated_data.sort_values(by='sativa_mentions', ascending=False, inplace=True)
aggregated_data.sort_values(by='indica_mentions', ascending=False, inplace=True)
sativa_mentions_latex = aggregated_data.sort_values(by='sativa_mentions', ascending=False)[['strain_name', 'strain_type', 'sativa_mentions']].head(5).to_latex(
    index=False,
    header=['Strain Name', 'Strain Type', 'Sativa Mentions'],
    column_format='lcr',
    escape=False
)
indica_mentions_latex = aggregated_data.sort_values(by='indica_mentions', ascending=False)[['strain_name', 'strain_type', 'indica_mentions']].head(5).to_latex(
    index=False,
    header=['Strain Name', 'Strain Type', 'Indica Mentions'],
    column_format='lcr',
    escape=False
)
sativa_mentions_latex = sativa_mentions_latex.replace('#', '\#')
indica_mentions_latex = indica_mentions_latex.replace('#', '\#')
print("Sativa Mentions:\n", sativa_mentions_latex)
print("Indica Mentions:\n", indica_mentions_latex)
with open(f'{report_dir}/sativa-mentions.tex', 'w') as f:
    f.write(sativa_mentions_latex)
with open(f'{report_dir}/indica-mentions.tex', 'w') as f:
    f.write(indica_mentions_latex)

# === User Analysis ===

# Count number of reviews per user.
n_reviews = reviews.groupby('user')['comment'].nunique()
n_reviews = n_reviews.sort_values(ascending=False)


# === Strain Analysis ===

# Count the number of reviews per strain.
n_reviews = reviews.groupby('strain_name')['comment'].nunique()
n_reviews = n_reviews.sort_values(ascending=False)

# Calculate average sentiment per strain.
avg_sentiment = reviews.groupby('strain_name')['sentiment'].mean()
avg_sentiment = avg_sentiment.sort_values(ascending=False)

# Visualize average sentiment per strain.
plt.figure(figsize=(12, 15))
sns.barplot(
    x=avg_sentiment,
    y=avg_sentiment.index,
    palette='Spectral_r',
)
plt.axvline(avg_sentiment.mean(), color='black', linestyle='--')
plt.xlabel('Average Review Sentiment by Strain')
plt.ylabel('')
for index, strain in enumerate(avg_sentiment.index):
    num_reviews = n_reviews[strain]
    sentiment_value = avg_sentiment[strain]
    text = f'{num_reviews} reviews' if num_reviews > 1 else f'{num_reviews} review'
    plt.text(
        sentiment_value,
        index,
        text,
        color='black',
        va='center',
        ha='left',
    )
plt.tight_layout()
plt.savefig(f'{assets_dir}/strain-sentiment.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


# === Colorimetry ===

def calculate_purpleness(rgb, how='scale'):
    """Purple is dominant in red and blue channels, and low in green.
    Args:
        rgb (list): A list of RGB values.
        how (str): How to calculate purpleness. Options are 'scale' and 'normalized'.
        Scale will return a value between 0 and 1. Normalized will return a value between -1 and 1.
    
    Returns:
        float: The purpleness score.
    
    Note: Adjust the formula for other shades of purple.
    """
    purpleness = (rgb[0] + rgb[2]) - 2*rgb[1]
    if how == 'scale':
        purpleness = purpleness + 510
        purpleness = purpleness / 1020
    elif how == 'normalized':
        purpleness = purpleness / 510
    return purpleness


def calculate_colourfulness(image, metric='M3'):
    # Convert the image from RGB to CIELab color space
    lab_image = color.rgb2lab(image)
    l, a, b = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]

    # Compute standard deviations and means in the CIELab space
    sigma_a, sigma_b = np.std(a), np.std(b)
    mu_a, mu_b = np.mean(a), np.mean(b)
    sigma_ab = np.sqrt(sigma_a**2 + sigma_b**2)
    mu_ab = np.sqrt(mu_a**2 + mu_b**2)
    # Aab = sigma_a * sigma_b

    # Compute Chroma and Saturation
    Chroma = np.sqrt(a**2 + b**2)
    _, mu_C = np.std(Chroma), np.mean(Chroma)
    # Saturation = Chroma / l
    # sigma_S, mu_S = np.std(Saturation), np.mean(Saturation)

    # Convert the image to a simplified opponent color space
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    sigma_rg, sigma_yb = np.std(rg), np.std(yb)
    mu_rg, mu_yb = np.mean(rg), np.mean(yb)
    sigma_rg_yb = np.sqrt(sigma_rg**2 + sigma_yb**2)
    mu_rg_yb = np.sqrt(mu_rg**2 + mu_yb**2)

    # Return the desired metric.
    if metric == 'M1':
        return sigma_ab + 0.37 * mu_ab
    elif metric == 'M2':
        return sigma_ab + 0.94 * mu_C
    elif metric == 'M3':
        return sigma_rg_yb + 0.3 * mu_rg_yb
    else:
        raise ValueError('Unknown metric: %s' % metric)


# Analyze the image for each strain.
purple_scores = []
colorfulness_scores = []
for index, obs in coa_data.iterrows():

    # Download the strain's image.
    try:
        sleep(1)
        image_url = obs['image_urls'][1]
        image_url = image_url.replace('50x50', image_size)
        image_file = download_coa_image(image_url, image_dir)
    except:
        print('No image found.')
        image_file = None

    # Analyze the image.
    image = cv2.imread(image_file)
    try:
        cropped_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        continue
    mean_color = cropped_img_rgb.mean(axis=0).mean(axis=0)
    purpleness = calculate_purpleness(mean_color)
    colorfulness = calculate_colourfulness(cropped_img_rgb)
    purple_scores.append(purpleness)
    colorfulness_scores.append(colorfulness)

# Assign the scores to the data.
coa_data['purpleness'] = purple_scores
coa_data['colorfulness'] = colorfulness_scores

# Visualize purpleness.
most_purple = coa_data.sort_values('purpleness', ascending=False).head(1).iloc[0]
most_green = coa_data.sort_values('purpleness', ascending=True).head(1).iloc[0]
plt.figure(figsize=(15, 8))
plt.hist(
    coa_data['purpleness'],
    bins=10,
    alpha=0.55,
    color='darkviolet',
)
plt.axvline(most_purple['purpleness'], color='darkviolet', linestyle='dashed', linewidth=2)
plt.text(
    most_purple['purpleness'], 3, f"Most purple: {most_purple['strain_name']}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
plt.axvline(most_green['purpleness'], color='mediumseagreen', linestyle='dashed', linewidth=2)
plt.text(
    most_green['purpleness'], 3, f"Most green: {most_green['strain_name']}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
plt.xlabel('Purpleness Score', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Flower Greenness to Purpleness', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.savefig(f'{assets_dir}/purpleness.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Visualize colorfulness.
most_colorful = coa_data.sort_values('colorfulness', ascending=False).head(1).iloc[0]
least_colorful = coa_data.sort_values('colorfulness', ascending=True).head(1).iloc[0]
plt.figure(figsize=(15, 8))
plt.hist(
    coa_data['colorfulness'],
    bins=10,
    alpha=0.55,
    color='darkturquoise',
)
plt.axvline(most_colorful['colorfulness'], color='darkviolet', linestyle='dashed', linewidth=2)
plt.text(
    most_colorful['colorfulness'], 3, f"Most colorful: {most_colorful['strain_name']}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
plt.axvline(least_colorful['colorfulness'], color='mediumseagreen', linestyle='dashed', linewidth=2)
plt.text(
    least_colorful['colorfulness'], 3, f"Least colorful: {least_colorful['strain_name']}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
plt.xlabel('Colorfulness Score', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Flower Colorfulness', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.savefig(f'{assets_dir}/colorfulness.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Colors table.
table_data = coa_data[['strain_name', 'purpleness', 'colorfulness']]
table_data = table_data.sort_values(['strain_name'], ascending=True)
latex_table = table_data.to_latex(index=False, float_format="%.2f", header=["Strain Name", "Purpleness", "Colorfulness"], column_format='lcc')
latex_table = latex_table.replace('#', '\#')
with open(f'{report_dir}/colors.tex', 'w') as file:
    file.write(latex_table)
print(latex_table)


# === Chemical Diversity Analysis ===

def calculate_shannon_diversity(df, compounds):
    """Calculate Shannon Diversity Index."""
    diversities = []
    for _, row in df.iterrows():
        proportions = [pd.to_numeric(row[compound], errors='coerce') for compound in compounds if pd.to_numeric(row[compound], errors='coerce') > 0]
        proportions = np.array(proportions) / sum(proportions)
        shannon_index = -np.sum(proportions * np.log2(proportions))
        diversities.append(shannon_index)
    return diversities


# Calculate diversity of terpenes.
extraction['cannabinoid_diversity'] = calculate_shannon_diversity(extraction, specific_cannabinoids)
extraction['terpene_diversity'] = calculate_shannon_diversity(extraction, specific_terpenes)

# Scatter plot of cannabinoid vs. terpene diversity
plt.figure(figsize=(14, 8.5))
sns.scatterplot(
    data=extraction,
    x='cannabinoid_diversity',
    y='terpene_diversity',
    hue='strain_type',
    style='strain_type', 
    palette='Set2',
    s=200,
)
plt.title('Terpene and Cannabinoid Diversity by Strain')
plt.xlabel('Cannabinoid Diversity')
plt.ylabel('Terpene Diversity')
leg = plt.legend(title='Strain Type', bbox_to_anchor=(1.05, 1), loc='upper left')
for leg_entry in leg.legendHandles: 
    leg_entry.set_sizes([200])
plt.grid(True)
plt.tight_layout()
for i in range(extraction.shape[0]):
    plt.text(
        extraction.iloc[i]['cannabinoid_diversity'],
        extraction.iloc[i]['terpene_diversity'],
        extraction.iloc[i]['strain_name'],
        ha='left',
        size='14',
        color='black',
    )
plt.tight_layout()
plt.savefig(f'{assets_dir}/diversity-scatterplot.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Sorting data for better visualization
sorted_cannabinoids = extraction.sort_values(by='cannabinoid_diversity', ascending=False)
sorted_terpenes = extraction.sort_values(by='terpene_diversity', ascending=False)
plt.figure(figsize=(18, 12))
plt.subplot(1, 2, 1)
sns.barplot(data=sorted_cannabinoids, x='cannabinoid_diversity', y='strain_name', palette='flare_r')
plt.title('Cannabinoid Diversity')
plt.xlabel('')
plt.ylabel('')
avg_cannabinoid_diversity = sorted_cannabinoids['cannabinoid_diversity'].mean()
plt.axvline(x=avg_cannabinoid_diversity, color='black', linestyle='--', label=f'Avg: {avg_cannabinoid_diversity:.2f}')
plt.legend(loc='lower right')
plt.subplot(1, 2, 2)
sns.barplot(data=sorted_terpenes, x='terpene_diversity', y='strain_name', palette='flare_r')
plt.title('Terpene Diversity')
plt.xlabel('')
plt.ylabel('')
avg_terpene_diversity = sorted_terpenes['terpene_diversity'].mean()
plt.axvline(x=avg_terpene_diversity, color='black', linestyle='--', label=f'Avg: {avg_terpene_diversity:.2f}')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{assets_dir}/diversity.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


# === Cannabinoid and Terpene Concentrations Analysis ===

# Analyze average concentrations.
stats = extraction.copy()
stats.replace({'<LOQ': 0, 'ND': np.nan}, inplace=True)
compounds = specific_cannabinoids + specific_terpenes
cannabinoid_stats = [(compound, detected, avg_concentration) for compound, detected, avg_concentration in stats if compound in specific_cannabinoids]
terpene_stats = [(compound, detected, avg_concentration) for compound, detected, avg_concentration in stats if compound in specific_terpenes]
cannabinoid_stats.sort(key=lambda x: x[0])
terpene_stats.sort(key=lambda x: x[0])
latex_table = """
\\begin{table}[H]
\\centering
\\begin{tabular}{lcc}
\\hline
Compound & Detection Rate & Average (\\%) \\\\
\\hline
\\multicolumn{3}{l}{Cannabinoids} \\\\
\\hline
"""
for compound, detected, avg_concentration in cannabinoid_stats:
    concentration_str = f"{avg_concentration:.2f}" if not np.isnan(avg_concentration) else "N/A"
    latex_table += f"{compound} & {detected:.0f}\\% & {concentration_str} \\\\\n"
latex_table += """
\\hline
\\multicolumn{3}{l}{Terpenes} \\\\
\\hline
"""
for compound, detected, avg_concentration in terpene_stats:
    concentration_str = f"{avg_concentration:.2f}" if not np.isnan(avg_concentration) else "N/A"
    latex_table += f"{compound} & {detected:.0f}\\% & {concentration_str} \\\\\n"

latex_table += """
\\hline
\\end{tabular}
\\end{table}
"""
latex_table = latex_table.replace('_', '\_')
print(latex_table)
with open(f'{report_dir}/compounds.tex', 'w') as file:
    file.write(latex_table)

# Average THC/CBD levels across strains
stats = extraction.copy()
stats.replace({'<LOQ': 0, 'ND': np.nan}, inplace=True)
average_thc_cbd = stats.groupby('strain_name')[['thca', 'total_cbd']].mean()
average_thc_cbd.columns = ['THCA', 'Total CBD']
average_thc_cbd.plot(kind='bar', figsize=(8, 5), color=['#17becf', '#b07aa1'])
plt.title('THCA and Total CBD Concentrations', fontsize=18)
plt.ylabel('Concentration (%)', fontsize=14)
plt.xticks(ha='right', rotation=45, fontsize=12)
plt.xlabel('')
plt.tight_layout()
plt.legend(fontsize=12, loc='upper right')
plt.yticks(np.arange(0, 21, 5), fontsize=12)
plt.ylim(0, 20)
plt.savefig(f'{assets_dir}/thca-cbd.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


# === Chemical similarity ===

# Clean the chemical data.
stats = extraction.copy()
stats.replace({'<LOQ': 0, 'ND': np.nan}, inplace=True)
for compound in specific_cannabinoids + specific_terpenes:
    stats[compound] = pd.to_numeric(stats[compound], errors='coerce')


# Visualize the average of the minor cannabinoids.
# Note: this is all `specific_cannabinoids` minus `thca`.
average_concentrations = stats[specific_cannabinoids].mean()
average_concentrations = average_concentrations.drop('thca', errors='ignore')
fig, ax = plt.subplots(figsize=(14, 8.5))
bar_width = 0.5
bar_l = np.arange(len(average_concentrations))
tick_pos = bar_l + bar_width / 2
ax.bar(bar_l, average_concentrations, width=bar_width, color='#ff796c', label='Average Concentration')
ax.set_xticks(tick_pos)
ax.set_xticklabels(average_concentrations.index, rotation=45, ha='right', fontsize=16)  # Increased font size
ax.set_title('Average Concentrations of Specific Cannabinoids', fontsize=28)  # Increased font size
ax.set_ylabel('Concentration (%)', fontsize=22)  # Increased font size
ax.legend(fontsize=20)  # Increased font size
plt.tight_layout()
plt.savefig(f'{assets_dir}/cannabinoids.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Visualize the average of the specific_terpenes.
average_terpene_concentrations = stats[specific_terpenes].mean()
fig, ax = plt.subplots(figsize=(14, 8.5))
bar_width = 0.5
bar_l = np.arange(len(average_terpene_concentrations))
tick_pos = bar_l + bar_width / 2
ax.bar(bar_l, average_terpene_concentrations, width=bar_width, color='mediumseagreen', label='Average Terpene Concentration')
ax.set_xticks(tick_pos)
ax.set_xticklabels(average_terpene_concentrations.index, rotation=45, ha='right', fontsize=16)  # Increased font size
ax.set_title('Average Concentrations of Specific Terpenes', fontsize=28)  # Increased font size
ax.set_ylabel('Concentration (%)', fontsize=22)  # Increased font size
ax.legend(fontsize=20)  # Increased font size
plt.tight_layout()
plt.savefig(f'{assets_dir}/terpenes.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


def plot_ratio(
        data,
        x_compound,
        y_compound,
        hue="strain_type",
        style="strain_type",
        palette="Set2",
        size=300,
        title=None,
        xlabel=None,
        ylabel=None,
        ymax=None,
        annotate='strain_name',
    ):
    """
    Plot a scatterplot showing the ratio of two compounds.

    Parameters:
    - data: DataFrame containing the data.
    - x_compound: The name of the compound for the x-axis.
    - y_compound: The name of the compound for the y-axis.
    - hue: Column name for color encoding.
    - style: Column name for style encoding.
    - palette: Color palette.
    - size: Marker size.
    - title: Plot title.
    - xlabel: x-axis label.
    - ylabel: y-axis label.
    """
    width = 16
    golden_ratio = (1 + 5 ** 0.5) / 2
    height = width / golden_ratio
    plt.figure(figsize=(width, height))
    ax = sns.scatterplot(
        data=data,
        x=x_compound,
        y=y_compound,
        hue=hue,
        style=style,
        palette=palette,
        s=size,
    )
    plt.title(title if title else f'{y_compound} to {x_compound}')
    plt.xlabel(xlabel if xlabel else x_compound.replace('_', '-').title() + ' (%)')
    plt.ylabel(ylabel if ylabel else y_compound.replace('_', '-').title() + ' (%)')
    leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, handletextpad=0.5, handlelength=2)
    for leg_entry in leg.legendHandles: 
        leg_entry.set_sizes([size * 0.8])
    plt.tight_layout()
    plt.xlim(0)
    if ymax:
        plt.ylim(0, ymax)
    texts = []
    if annotate:
        for _, row in data.iterrows():
            y_pos = row[y_compound]
            if ymax and y_pos > ymax:
                continue
            texts.append(plt.text(row[x_compound], row[y_compound], row[annotate], ha='left', size='small'))
    adjust_text(
        texts,
        avoid_points=False,
    )
    outfile = f'{assets_dir}/{y_compound.replace("_", "-")}-to-{x_compound.replace("_", "-")}.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight', transparent=False)
    print('Saved:', outfile)
    plt.show()


# Beta-Myrcene to Alpha-Pinene Ratio
plot_ratio(
    stats,
    'alpha_pinene',
    'beta_myrcene',
    title='Beta-Myrcene to Alpha-Pinene'
)

# Beta-Caryophyllene to D-Limonene Ratio
plot_ratio(
    stats,
    'd_limonene',
    'beta_caryophyllene',
    title='Beta-Caryophyllene to D-Limonene'
)

# Visualize ratio of `alpha_pinene` to `d_limonene`
plot_ratio(
    stats,
    'd_limonene',
    'alpha_pinene',
    title='Alpha-Pinene to D-Limonene'
)

# Visualize ratio of `terpinolene` to `beta_myrcene`
plot_ratio(
    stats,
    'beta_myrcene',
    'terpinolene',
    title='Terpinolene to Beta-Myrcene',
    ymax=0.065
)

# Terpene ratio: Alpha-Humulene to Beta-Caryophyllene
# Justification: common enzyme.
stats['alpha_humulene'] = pd.to_numeric(stats['alpha_humulene'], errors='coerce')
stats['beta_caryophyllene'] = pd.to_numeric(stats['beta_caryophyllene'], errors='coerce')
plot_ratio(
    stats,
    'beta_caryophyllene',
    'alpha_humulene',
    title='Alpha-Humulene to Beta-Caryophyllene'
)

# Terpene ratio: Beta-Myrcene to D-Limonene
plot_ratio(
    stats,
    'd_limonene',
    'beta_myrcene',
    title='Beta-Myrcene to D-Limonene'
)

# Camphene to D-Limonene
stats['camphene'] = pd.to_numeric(stats['camphene'], errors='coerce')
stats['d_limonene'] = pd.to_numeric(stats['d_limonene'], errors='coerce')
plot_ratio(
    stats,
    'd_limonene',
    'camphene',
    title='Camphene to D-Limonene'
)

# Visualize total cannabinoids and total terpenes.
plot_ratio(
    stats,
    'sum_of_terpenes',
    'sum_of_cannabinoids',
    title='Total Terpenes to Total Cannabinoids',
    xlabel='Total Terpenes (%)',
    ylabel='Total Cannabinoids (%)',
)

# Cannabinoid ratio: THCA to Delta-9 THC
stats['delta_9_thc'] = pd.to_numeric(stats['delta_9_thc'], errors='coerce')
stats['thca'] = pd.to_numeric(stats['thca'], errors='coerce')
plot_ratio(
    stats,
    'delta_9_thc',
    'thca',
    title='THCA to Delta-9 THC',
    xlabel='Delta-9 THC (%)',
    ylabel='THCA (%)',
)

# Look at total CBG.
stats['cbg'] = pd.to_numeric(stats['cbg'], errors='coerce')
stats['cbga'] = pd.to_numeric(stats['cbga'], errors='coerce')
stats['total_cbg'] = stats['cbg'] + stats['cbga'].mul(0.877)
plot_ratio(
    stats,
    'cbga',
    'thca',
    title='THCA to CBGA'
)

# CBG to CBD.
stats['cbd'] = pd.to_numeric(stats['cbd'], errors='coerce')
stats['cbda'] = pd.to_numeric(stats['cbda'], errors='coerce')
stats['calculated_total_cbd'] = stats['cbd'] + stats['cbda'].mul(0.877)
# plot_ratio(
#     stats,
#     'calculated_total_cbd',
#     'total_cbg',
#     title='Total CBG to Total CBD',
#     xlabel='Total CBD (%)',
#     ylabel='Total CBG (%)',
# )


# === Harvest Time Analysis ===

# Timeseries analysis of when strains were harvested.
stats = extraction.copy()
stats['date'] = pd.to_datetime(stats['date_received'])
stats['harvest_month'] = stats['date'].dt.month
stats['harvest_year'] = stats['date'].dt.year
harvest_time_data = stats.groupby(['harvest_year', 'harvest_month']).size().reset_index(name='counts')
plt.figure(figsize=(15, 8))
sns.lineplot(
    data=harvest_time_data,
    x="harvest_month",
    y="counts",
    # hue="harvest_year",
    marker="o",
)
plt.title("Harvest Time Distribution by Month and Year")
plt.xlabel("Month")
plt.ylabel("Number of Strains Harvested")
plt.legend(title='Harvest Year')
plt.grid(True)
plt.show()


# === Classification Analysis (PCA) ===

# Identify the dominant terpene for each strain.
for compound in specific_terpenes:
    stats[compound] = pd.to_numeric(stats[compound].replace({'ND': np.nan, '<LOQ': 0}), errors='coerce')
dominant_terpene = stats[specific_terpenes].idxmax(axis=1)
dominant_terpene_counts = dominant_terpene.value_counts()
dominant_terpene_percentages = (dominant_terpene_counts / dominant_terpene_counts.sum()) * 100
dominant_terpene_percentages.plot(
    kind='barh',
    figsize=(13, 8),
    color='mediumseagreen',
)
for index, value in enumerate(dominant_terpene_percentages):
    plt.text(value, index, f" {value:.0f}%", va='center')
plt.title('Dominant Terpenes Across Strains')
plt.xlabel('Percent of Strains with Dominant Terpene (%)')
plt.ylabel('')
plt.xticks(rotation=0, ha='right')
plt.tight_layout()
plt.savefig(f'{assets_dir}/dominant-terpenes.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Select relevant features for clustering
features = ['beta_caryophyllene', 'd_limonene', 'beta_myrcene', 'alpha_pinene', 'terpinolene', 'linalool']
X = stats[features].fillna(0)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca_scores, columns=['PC1', 'PC2'])
pca_df['Cluster'] = labels

# Visualize PCA scores with dominant terpenes.
pca_df['dominant_terpene'] = dominant_terpene
pca_df['strain_type'] = stats['strain_type']
pca_df['strain_name'] = stats['strain_name']
plt.figure(figsize=(15, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='dominant_terpene',
    data=pca_df.sort_values(by='dominant_terpene'),
    palette='tab10',
    s=200,
    alpha=0.7,
    legend='full',
    # style='strain_type'
)
plt.title('PCA of Strains by Dominant Terpenes')
plt.xlabel('PC1')
plt.ylabel('PC2')
leg = plt.legend(title='Dominant Terpene', bbox_to_anchor=(1.05, 1), loc='upper left')
for leg_entry in leg.legendHandles: 
    leg_entry.set_sizes([200])
texts = []
for _, row in pca_df.iterrows():
    y_pos = row['PC2']
    text = plt.text(
        row['PC1'],
        row['PC2'],
        row['strain_name'],
        ha='left',
        size='small',
    )
    texts.append(text)
adjust_text(
    texts,
    avoid_points=False,
)
plt.tight_layout()
plt.savefig(f'{assets_dir}/pca-dominant-terpenes.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()
