"""
Word Vectors
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 4/19/2023
Updated: 4/19/2023
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>

Credit: Maxim and Ananth Reddy <https://stackoverflow.com/a/48230654/5021266>
License: CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>

References:

    - Text generator based on LSTM model with pre-trained Word2Vec embeddings in Keras
    URL: <https://gist.github.com/maxim5/c35ef2238ae708ccb0e55624e9e0252b>

"""

from __future__ import print_function

import csv
import numpy as np
import gensim
import string

from keras.callbacks import LambdaCallback
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential

import pandas as pd
from gensim.models import KeyedVectors


#------------------------------------------------------------------------------
# Get the corpus.
#------------------------------------------------------------------------------

# Get Washington strain vocabulary.
strain_vocab_file = 'washington-strain-vocab.txt'
strain_data = pd.read_excel('ccrs-strain-statistics-2023-03-07.xlsx')
strain_data['strain_name'] = strain_data['strain_name'].apply(str.title)
strain_data.drop_duplicates(subset=['strain_name'], inplace=True)
strain_data.sort_values(by=['strain_name'], inplace=True)
strain_data.to_csv(
    strain_vocab_file,
    columns=['strain_name'],
    header=None,
    index=None,
    mode='w+',
    quoting=csv.QUOTE_NONE,
)


#------------------------------------------------------------------------------
# Train the model.
#------------------------------------------------------------------------------

# Format sentences.
max_sentence_len = 40
docs = strain_data['strain_name'].tolist()
translator = str.maketrans('', '', string.punctuation)
sentences = [[word for word in doc.lower().translate(translator).split()[:max_sentence_len]] for doc in docs]

# Train the word to vector model.
print('\nTraining word2vec...')
word_model = gensim.models.Word2Vec(
  sentences,
  vector_size=100,
  min_count=1,
  window=5,
#   iter=100
)


#------------------------------------------------------------------------------
# Explore the model.
#------------------------------------------------------------------------------

# Identify a few sample words.
sample_words = sentences[0]

# Analyze the model.
pretrained_weights = word_model.wv.vectors
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)
print('Checking similar words:')
for word in sample_words:
    most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.wv.most_similar(word)[:1])
    print('  %s -> %s' % (word, most_similar))


#------------------------------------------------------------------------------
# Train the model.
#------------------------------------------------------------------------------

def word2idx(word):
    return word_model.wv.key_to_index[word]

def idx2word(idx):
    return word_model.wv.index_to_key[idx]

# Prepare the data.
print('\nPreparing the data for LSTM...')
max_sentence_len = 40
train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
    if not sentence:
        continue
    for t, word in enumerate(sentence[:-1]):
        train_x[i, t] = word2idx(word)
    train_y[i] = word2idx(sentence[-1])
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)

# Train the model.
print('\nTraining LSTM...')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')


#------------------------------------------------------------------------------
# Use the model.
#------------------------------------------------------------------------------

def sample(preds, temperature=1.0):
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_next(text, num_generated=10):
    word_idxs = [word2idx(word) for word in text.lower().split()]
    for i in range(num_generated):
        prediction = model.predict(x=np.array(word_idxs))
        idx = sample(prediction[-1], temperature=0.7)
        word_idxs.append(idx)
    return ' '.join(idx2word(idx) for idx in word_idxs)


def on_epoch_end(epoch, _):
    print('\nGenerating text after epoch: %d' % epoch)
    texts = [
        'Yellow',
        'Submarine',
    ]
    for text in texts:
        sample = generate_next(text)
        print('%s... -> %s' % (text, sample))

# Fit the model.
model.fit(
    train_x,
    train_y,
    batch_size=128,
    epochs=5,
    callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)],
)
