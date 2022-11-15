"""
Product Subtypes
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 11/9/2022
Updated: 11/9/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

References:

    - Community Detection Algorithms
    https://towardsdatascience.com/community-detection-algorithms-9bd8951e7dae

    - What is Network Analysis?
    https://towardsdatascience.com/network-analysis-d734cd7270f8

"""

# External imports.
import pandas as pd
import spacy
import textacy


# Specify where your data lives.
DATA_DIR = '../../.datasets/lab_results/'


# === Louvain community detection algorithm ===

from cdlib import algorithms
import networkx as nx
G = nx.karate_club_graph()
coms = algorithms.louvain(G, weight='weight', resolution=1., randomize=False)
