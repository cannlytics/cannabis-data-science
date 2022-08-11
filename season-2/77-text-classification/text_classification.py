"""
Computer Vision Application
Cannabis Data Science #76 | 2022-07-27
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 8/2/2022
Updated: 8/2/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Can we build a model to predict the analysis given an analyte?

Algorithm:

    1. Annotate data.

    2. Use the model.

    3. Feed the model more annotated data.

""" 


#-----------------------------------------------------------------------
# Use NLP to parse CoAs!
#-----------------------------------------------------------------------

import spacy
from spacy import displacy

# Initialize a NLP client.
nlp = spacy.load("en_core_web_sm")

# Parse entities from a CoA!
doc = nlp(report.pages[1].extract_text())
displacy.serve(doc, style="ent")


#-----------------------------------------------------------------------
# Use Gensim
#-----------------------------------------------------------------------

import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

# Find the similarity between various words.
pairs = [
    ('car', 'minivan'),   # a minivan is a kind of car
    ('car', 'bicycle'),   # still a wheeled vehicle
    ('car', 'airplane'),  # ok, no wheels, but still a vehicle
    ('car', 'cereal'),    # ... and so on
]
for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))

# Find the 5 most similar words.
print(wv.most_similar(positive=['car', 'minivan'], topn=5))


# TODO: Create custom analyte / analysis mappings.
# for result in data['results']:
#     print(f"[('{result['name']}', '{result['analysis']}')]")


from gensim.test.utils import datapath
from gensim import utils

class CannlyticsCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line,
            # tokens separated by whitespace
            yield utils.simple_preprocess(line)


# Train a model on the corpus.
import gensim.models

sentences = CannlyticsCorpus()
model = gensim.models.Word2Vec(
    sentences=sentences,
    # corpus_file='', # preferred!
    min_count=1, # Remove infrequently used words.
)


# Save the NLP model.
# import tempfile

# with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
#     temporary_filepath = tmp.name
#     model.save(temporary_filepath)
#     #
#     # The model is now safely stored in the filepath.
#     # You can copy it to other machines, share it with others, etc.
#     #
#     # To load a saved model:
#     #
#     new_model = gensim.models.Word2Vec.load(temporary_filepath)


# Load the NLP model and train the model with more data.
# model = gensim.models.Word2Vec.load(temporary_filepath)
# more_sentences = [
#     ['Advanced', 'users', 'can', 'load', 'a', 'model',
#      'and', 'continue', 'training', 'it', 'with', 'more', 'sentences'],
# ]
# model.build_vocab(more_sentences, update=True)
# model.train(more_sentences, total_examples=model.corpus_count, epochs=model.epochs)

# # cleaning up temporary file
# import os
# os.remove(temporary_filepath)


# === Visualize the model predictions ===

# from sklearn.decomposition import IncrementalPCA    # inital reduction
# from sklearn.manifold import TSNE                   # final reduction
# import numpy as np                                  # array handling


# def reduce_dimensions(model):
#     num_dimensions = 2  # final num dimensions (2D, 3D, etc)

#     # extract the words & their vectors, as numpy arrays
#     vectors = np.asarray(model.wv.vectors)
#     labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

#     # reduce using t-SNE
#     tsne = TSNE(n_components=num_dimensions, random_state=0)
#     vectors = tsne.fit_transform(vectors)

#     x_vals = [v[0] for v in vectors]
#     y_vals = [v[1] for v in vectors]
#     return x_vals, y_vals, labels


# x_vals, y_vals, labels = reduce_dimensions(model)

# def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
#     from plotly.offline import init_notebook_mode, iplot, plot
#     import plotly.graph_objs as go

#     trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
#     data = [trace]

#     if plot_in_notebook:
#         init_notebook_mode(connected=True)
#         iplot(data, filename='word-embedding-plot')
#     else:
#         plot(data, filename='word-embedding-plot.html')


# def plot_with_matplotlib(x_vals, y_vals, labels):
#     import matplotlib.pyplot as plt
#     import random

#     random.seed(0)

#     plt.figure(figsize=(12, 12))
#     plt.scatter(x_vals, y_vals)

#     #
#     # Label randomly subsampled 25 data points
#     #
#     indices = list(range(len(labels)))
#     selected_indices = random.sample(indices, 25)
#     for i in selected_indices:
#         plt.annotate(labels[i], (x_vals[i], y_vals[i]))

# try:
#     get_ipython()
# except Exception:
#     plot_function = plot_with_matplotlib
# else:
#     plot_function = plot_with_plotly

# plot_function(x_vals, y_vals, labels)


#-----------------------------------------------------------------------
# Use NLP to standardize analyses.
#-----------------------------------------------------------------------

# Define known map of encountered analyses.
{
    'Foreign Material': 'foreign_matter',
    'Heavy Metals Screen': 'heavy_metals',
    'Microbial Screen': 'microbes',
    'Mycotoxin Screen': 'mycotoxins',
    'Percent Moisture (%)': 'moisture_content',
    'Potency Test Result': 'cannabinoids',
    'Pesticide Screen Result - Category 1': 'pesticides',
    'Pesticide Screen Result - Category 2': 'pesticides',
    'Residual Solvent Screen - Category 1': 'residual_solvents',
    'Residual Solvent Screen - Category 2': 'residual_solvents',
    'Terpene Test Result': 'terpenes',
    'Water Activity (Aw)': 'water_activity',
}

# TODO: Train the model.


# TODO: Test the model.


# TODO: Re-train the model with the full dataset.


# TODO: Use the model to predict analyses!


#-----------------------------------------------------------------------
# Use NLP to standardize analyses.
#-----------------------------------------------------------------------

# 1. Get a large map of analytes with analyses (from collected results!).


# 2. Build the model.

    # TODO: Train the model.


    # TODO: Test the model.


    # TODO: Re-train the model with the full dataset.


    # TODO: Use the model to predict analyses!


# 3. Apply the model to the list of SC Labs analytes.
