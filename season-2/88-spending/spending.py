"""
Spending Trending
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 11/2/2022
Updated: 11/2/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Setup:

    1. Install all Python dependencies:

    pip install cannlytics pdfplumber pytesseract spacy transformers

"""
# Standard imports.
import os

# External imports.
from cannlytics.utils import sorted_nicely


#------------------------------------------------------------------------------
# Read in all of the receipt images.
#------------------------------------------------------------------------------

# Specify where your receipts live.
RECEIPT_DIR = '../../.datasets/receipts'
receipt_filenames = []
filenames = os.listdir(RECEIPT_DIR)
for filename in filenames:
    if filename.endswith('.jpg'):
        receipt_filenames.append(f'{RECEIPT_DIR}/{filename}')


#------------------------------------------------------------------------------
# Parse the text from the images with OCR.
#------------------------------------------------------------------------------

import cv2
import pdfplumber
import pytesseract
import re


def rotate(image, center=None, scale=1.0):
    """Rotate an image with text into a readable position, as well as possible.
    Credit: Mousam Singh <https://stackoverflow.com/a/55122911/5021266>
    License: CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>
    """
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    degrees = int(re.search(
        '(?<=Rotate: )\d+',
        pytesseract.image_to_osd(image),
    ).group(0))
    angle = 360 - degrees
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def image_to_text(image_file):
    """Extract the text from an image, rotating it as necessary."""

    # Rotate the image as needed.
    image_data = rotate(cv2.imread(str(image_file), cv2.IMREAD_COLOR))

    # Convert the image to PDF.
    pdf = pytesseract.image_to_pdf_or_hocr(image_data, extension='pdf')
    pdf_file = image_file.replace('.jpg', '.pdf')
    with open(pdf_file, 'w+b') as f:
        f.write(pdf)
        f.close()

    # Extract the text from the PDF.
    receipt = pdfplumber.open(pdf_file)
    text = ''
    for page in receipt.pages:
        text += page.extract_text()
    return text


# Example: Extract the text from a receipt.
image_file = '../../.datasets/receipts/receipt-1.jpg'
image_text = image_to_text(image_file)

# Example: Extract the text from all receipts.
receipt_texts = []
for filename in receipt_filenames:
    image_text = image_to_text(filename)
    receipt_texts.append(image_text)

# Save the receipt text.
with open('receipt_texts.txt', 'w+') as f:
    for s in receipt_texts:
        f.write(str(s) + '\n')

#------------------------------------------------------------------------------
# Extract the key data from the text with NLP.
#------------------------------------------------------------------------------

import spacy
from spacy import displacy


# Create natural language processing client.
# Use `en_core_web_sm` for speed and `en_core_web_trf` for accuracy.
# For a blank model, use spacy.blank('en')
nlp = spacy.load('en_core_web_lg')

# Look at easily-recognized entities.
doc = nlp(receipt_texts[0])
displacy.render(doc, style='ent')


#------------------------------------------------------------------------------

# Extract all of the money entities.
time_entities = []
money_entities = []
for i in doc.ents:
    entry = str(i.lemma_).lower()
    if i.label_ in ['MONEY']:
        money_entities.append(entry)


#------------------------------------------------------------------------------

import pandas as pd

# TODO: Identify times.
time_entities = []
for i in doc.ents:
    entry = str(i.lemma_).lower()
    if i.label_ in ['CARDINAL']:
        try:
            date_time = pd.to_datetime(entry)
            time_entities.append(date_time)
        except:
            pass


#------------------------------------------------------------------------------

# FIXME: Identify quantities.
patterns = [
    {
        'label': 'QUANTITY',
        'pattern': [
            {'LIKE_NUM': True},
            {'LOWER': {'IN': [
                'g', 'gr', 'gram', 'grams', 'teenth', 'sixteenth', 'eighth',
                'quarter', 'ounce', 'ounces', 'oz', 'pound', 'lb', 'mg', 'kg',
                'milligram', 'milligrams', 'kilogram', 'kilograms',
                '1\/8 oz'
            ]}}
        ],
    },
]
try:
    ruler = nlp.add_pipe('entity_ruler', after='ner')
except ValueError:
    nlp.remove_pipe('entity_ruler')
    ruler = nlp.add_pipe('entity_ruler', after='ner')
ruler.add_patterns(patterns)

# Look at the receipts for quantities.
doc = nlp(receipt_texts[0])
displacy.render(doc, style='ent')


#------------------------------------------------------------------------------

# TODO: Identify strain names.


#------------------------------------------------------------------------------
# Analyze the data.
#------------------------------------------------------------------------------

from transformers import GPT2Tokenizer

# Use a pre-trained model to tokenize the receipts.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer(image_text)['input_ids']


#------------------------------------------------------------------------------
# Trend the spending!
#------------------------------------------------------------------------------

# Future work: Trend the spending amount by month!

