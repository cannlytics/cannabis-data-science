"""
Word Vectors with Gensim
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 4/19/2023
Updated: 4/19/2023
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>
"""
# Standard imports:
import csv
import string

# External imports:
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd


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

# Train the model.
corpus = strain_data['strain_name'].tolist()
model = Word2Vec(sentences, vector_size=24, epochs=100)
word_vectors = model.wv

# Save the vectorized data to a file.
vector_file = 'washington-strain-vectors.kv'
word_vectors.save(vector_file)
reloaded_word_vectors = KeyedVectors.load(vector_file)


#------------------------------------------------------------------------------
# Analyze the model.
#------------------------------------------------------------------------------

# Get sample strains.
sample_words = strain_data['strain_name'].sample(5, random_state=420).tolist()

# Check the "most similar words", using the default "cosine similarity" measure.
for sentence in sentences[:5]:
    try:
        result = word_vectors.most_similar(positive=sentence)
        most_similar_key, similarity = result[0]
        print(sentence, '->', f"{most_similar_key} ({similarity:.4f})")
    except:
        pass


#------------------------------------------------------------------------------
# Visualize the model.
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 21,
})


# Visualize the results.
vocab = list(model.wv.key_to_index)
X = model.wv[vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

# Sample the data.
sample = df.sample(200, random_state=420)

# Create a 2D scatterplot of word vectors.
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=sample, x='x', y='y', s=100, alpha=0.8)
count = 0
for i, row in sample.iterrows():
    count +=1
    if count % 10 == 0:
        ax.annotate(
            row.name, 
            (row['x'], row['y']),
            fontsize=24, 
            ha='left',
        )
ax.set_title('Strain word vectors\n', fontsize=28)
sns.despine()
plt.show()
