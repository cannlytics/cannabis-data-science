"""
Personality Prediction Models
Cannabis Data Science #75 | 2022-07-20
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 7/14/2022
Updated: 7/14/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    

Data Sources:

    - 

Resources:

    - https://github.com/huggingface/transformers

"""
# Standard imports.
import ast

# External imports.
from cannlytics.utils.utils import snake_case
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


# === Setup ===



# === Get the data. ===


# === Clean the data. ===


# === Process the data. ===


# === Augment the data. ===

# Features:
# - General information
# - Vocabulary


# === Analyze the data. ===



# Sentiment analysis
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
classifier('We are very happy to introduce pipeline to the transformers repository.')
# [{'label': 'POSITIVE', 'score': 0.9996980428695679}]


# Question answering.
question_answerer = pipeline('question-answering')
question_answerer({
    'question': 'What is the name of the repository ?',
    'context': 'Pipeline has been included in the huggingface/transformers repository'
})
# {'score': 0.30970096588134766, 'start': 34, 'end': 58, 'answer': 'huggingface/transformers'}

# === Use the data and models. ===

# - text classification
# - information extraction
# - question answering
# - summarization
# - translation
# - text generation


# === Save the results. ===

