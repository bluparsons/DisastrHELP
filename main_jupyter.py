#!/usr/bin/env python
# coding: utf-8

# ## Data Science Project
# #### Title: Event Identification and Text Summarisation using Open Source Small Language Models
# #### Author: Blu Parsons
# #### Date: 2024/2025
# 
# Key tables: 
# - df_default
# - df_tuned
# - df_topics
# - df_clean

# #### Import libraries

# In[285]:


# Import libraries
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
import random
import time
import textstat
from collections import Counter

# System libraries
import sys
import os
import twitter

# Import transformer models
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelWithLMHead
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from sentence_transformers import SentenceTransformer
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import seaborn as sns

# Dimension reduction, clustering, topic modelling
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pacmap
import trimap
import hdbscan
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from bertopic.representation._base import BaseRepresentation
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.model_selection import ParameterGrid
from hdbscan.validity import validity_index
from skopt import gp_minimize, forest_minimize, dummy_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer
from skopt.plots import plot_objective, plot_convergence, plot_evaluations, plot_regret
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances  # or euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy.stats import gaussian_kde
from nltk.corpus import stopwords
import nltk
from nltk.util import ngrams
#nltk.download('stopwords')
import spacy
from typing import Mapping, List, Tuple
nlp = spacy.load("en_core_web_sm")
from scipy.stats import ttest_rel

# T5 Fine Tuning
#from datasets import load_dataset, Dataset
#from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# APIs
import pprint
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
import openai
from huggingface_hub import login

# Evaluate
import evaluate
#from bert_score import score
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

from trl import DPOTrainer, DPOConfig
from transformers import TrainingArguments
from tqdm.auto import tqdm


# #### GitHub packages

# In[247]:


sys.path.append(r'C:\Users\mail\OneDrive\University\04 Bristol\07 Project') # To import procedures from other python files

# Import Github packages
# https://github.com/oduwsdl/tweetedat
import Code.source_third_party.tweetedat.script.TimestampEstimator as ta
# https://github.com/guyfe/Tweetsumm
import Code.source_third_party.Tweetsumm.tweet_sum_processor as tsp


# #### Functions

# In[248]:


# Data Cleansing

# Change date format from Unix timestamp to date as a string
def timestamp_to_str(tstamp):
    utcdttime = datetime.utcfromtimestamp(tstamp / 1000)
    utcdttime = utcdttime.strftime('%Y-%m-%d')
    return utcdttime

# Change date format from Unix timestamp to date and time as a string
def date_time_stamp_to_str(tstamp):
    utcdttime = datetime.utcfromtimestamp(tstamp / 1000)
    utcdttime = utcdttime.strftime('%Y-%m-%d, %H:%M:%S')  # H is in 24 hour clock
    return utcdttime

# Add Tweet DateTime in Unix epoch time format
def add_date_time_unix(df):
    df['DateTimeUnix'] = df.apply(lambda row: ta.find_tweet_timestamp(row['tweet_id']), axis=1)

# Add Tweet Date in string format
def add_date_str(df):
    df['DateTimeStr'] = df.apply(lambda row: timestamp_to_str(ta.find_tweet_timestamp(row['tweet_id'])), axis=1)

# Add Tweet DateTime in string format
def add_date_time(df):
    df['DateTime'] = df.apply(lambda row: date_time_stamp_to_str(ta.find_tweet_timestamp(row['tweet_id'])), axis=1)


# Pre-processing

# Output tweet information for a given tweet_id
def output_tweet_text(tweet_id, df):
    df_data_filtered = df[df['tweet_id'] == tweet_id]
    
    for tweet_date, text_original, text_clean in zip(
        df_data_filtered["DateTimeStr"],
        df_data_filtered["tweet_text"],
        df_data_filtered["tweet_text_clean"]):

        print("Tweet Date: ", tweet_date)
        print("-" * 10)  # separator line
        print("Original Tweet: ", text_original)
        print("-" * 10)  # separator line
        print("Clean Tweet: ", text_clean)
        print("-" * 80)  # separator line

USERNAME_PATTERN = re.compile(r'@[\w.-]+')
RETWEET_USERNAME_PATTERN = re.compile(r'RT @[\w.-]+: ')
EMOJI_PATTERN = [
    (re.compile(r'\:\-\)'), 'smiling_face'),
    (re.compile(r'\:\-\('), 'sad_face'),
    (re.compile(r'\:\-\/'), 'angry_face'),
    (re.compile(r'\:\-\\'), 'angry_face'),
    (re.compile(r'❤'), 'love'),
    (re.compile(r'❤️'), 'love'),
    (re.compile(r'♥'), 'love'),
    (re.compile(r'❤'), 'love'),
    (re.compile(r'♥'), 'love'),
    (re.compile(r'❣'), 'love'),
    (re.compile(r'⭐'), 'star'),
    (re.compile(r'✨'), 'star'),
    (re.compile(r'❗'), '!'),
    (re.compile(r'☹️'), 'sad_face'),
    (re.compile(r'‼'), '!'),
    (re.compile(r'✔'), 'yes'),
]
TEXT_REPLACE_PATTERN = [
    (re.compile(r'\&amp\;'), 'and')
]
URL_PATTERN = re.compile(r'https?://\S+')
DIGITS_PATTERN = re.compile(r'\d')
NON_ASCII_PATTERN = re.compile(r'[^\x00-\x7F]+')

# URLs
def preprocessing_count_urls(text):
    url = URL_PATTERN.findall(text)
    return len(url)

def preprocessing_remove_urls(text):
    url = RETWEET_USERNAME_PATTERN.sub('', text)
    return url

# Strings
def preprocessing_count_strings(text, strToRemove):
    digits = STRING_PATTERN.findall(text)
    return len(digits)

def preprocessing_remove_string(text, strToRemove):
    string_pattern = strToRemove
    string_removed = re.sub(string_pattern, '', text)
    return string_removed

# Digits
def preprocessing_count_digits(text):
    digits = DIGITS_PATTERN.findall(text)
    return len(digits)

def preprocessing_remove_digits(text):
    text_cleaned = DIGITS_PATTERN.sub('', text)
    return text_cleaned

# Emojis
def preprocessing_count_emoji(text):
    count = 0
    for pattern, _ in EMOJI_PATTERN:
        matches = pattern.findall(text)
        count += len(matches)
    return count

def preprocessing_replace_emoji(text):
    for pattern, replacement in EMOJI_PATTERN:
        text = pattern.sub(replacement, text)
    return text

# Ampersand, &
def preprocessing_replace_amp(text):
    for pattern, replacement in TEXT_REPLACE_PATTERN:
        text = pattern.sub(replacement, text)
    return text

# Retweet usernames
def preprocessing_count_retweet_username(text):
    retweet_usernames = RETWEET_USERNAME_PATTERN.findall(text)
    return len(retweet_usernames)

def preprocessing_remove_retweet_username(text):
    text_cleaned = RETWEET_USERNAME_PATTERN.sub('', text)
    return text_cleaned

# Usernames
def preprocessing_count_username(text):
    #usernames = USERNAME_PATTERN.findall(text)
    #return len(usernames)
    if text.startswith("RT "):
        return []  # Ignore, since it's a retweet
    else:
        return USERNAME_PATTERN.findall(text)

def preprocessing_remove_username(text):
    text_cleaned = USERNAME_PATTERN.sub('', text)
    return text_cleaned

def preprocessing_remove_non_ascii(text):
    text_cleaned = NON_ASCII_PATTERN.sub('', text)
    return text_cleaned

# Run the pre-processing functions
def custom_preprocessing(text):
    # Remove URLs
    text = preprocessing_remove_urls(text)

    # Remove all usernames
    text = preprocessing_remove_retweet_username(text)

    # Remove all usernames
    text = preprocessing_remove_username(text)

    # Remove special symbols
    text = preprocessing_remove_string(text, '#')
    text = preprocessing_remove_string(text, 'â€˜')
    text = preprocessing_remove_string(text, 'â€™')
    text = preprocessing_remove_string(text, 'âžœ')
    text = preprocessing_remove_string(text, 'âž')
    text = preprocessing_remove_string(text, 'â')
    text = preprocessing_remove_string(text, 'ž')
    text = preprocessing_remove_string(text, 'œ')
    text = preprocessing_remove_string(text, 'Â»')
    text = preprocessing_remove_string(text, '»')

    # Remove numerical digits
    text = preprocessing_remove_digits(text)

    # remove special chars
    # text = re.sub("\\W"," ",text)

    # Change icon emojis into describing word
    # text = emoji.demojize(text)

    # Change text emojis into work, e.g. :-)
    text = preprocessing_replace_emoji(text)

    # Change text &amp; into and
    text = preprocessing_replace_amp(text)

    # Remove non-ASCII, always do this last as this last otherwise it will remove ones we want to replace
    text = preprocessing_remove_non_ascii(text)

    return text


# Cluster embeddings and get cluster topics

def cluster_embeddings(embeddings, n_neighbors,min_dist,min_cluster_size,min_samples,n_components):
    # Apply UMAP for dimensionality reduction
    # Use n_components=50 for clustering
    # Use n_components=2 only for visualization
    umap_model_first_reduction = umap.UMAP(n_neighbors=n_neighbors,
                                           min_dist=min_dist,
                                           n_components=n_components,
                                           random_state=42,
                                           metric='cosine')
    X_umap_first_reduction = umap_model_first_reduction.fit_transform(embeddings)

    # Apply HDBSCAN clustering
    # Tuning options: 
    # Grid search, small datasets, all combinations, computationally expensive
    # Random search, random combinations, more efficient
    # Bayesian optimisation, uses past to select parameters, good for expensive functions
    # Decision: 
    # Start with random search to get a rough idea of parameter importance, then use
    # Bayesian Optimization for fine tuning after initial exploration
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,  # Lower values is more clusters but less meaningful ones
                                        min_samples=min_samples,  # How clusters are defined, higher values means clusters more conservative
                                        cluster_selection_epsilon=0.1,  # Fine tuning cluster boundaries
                                        metric='euclidean',
                                        #alpha=  # How mutual reachability distance is computed, higher means softer clustering
                                        cluster_selection_method='eom', #eom, leaf
                                        prediction_data=True)
    clusters = hdbscan_clusterer.fit_predict(X_umap_first_reduction) # Experiment with clusters = hdbscan_clusterer.fit_predict(embeddings)

    # Compute CBDV Score (only if there are at least 2 clusters)
    valid_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    # Compute DBCV Score (Density-Based Clustering Validation Score)
    #if valid_clusters > 1:
    #    dbcv_score = validity_index(X_umap_first_reduction.astype(np.float64), clusters)
    #else:
    #    dbcv_score = 1e6  # or some large positive value to indicate a bad score
    dbcv_score = validity_index(X_umap_first_reduction.astype(np.float64), clusters)
    print("Best DBCV score: ", dbcv_score)

    return umap_model_first_reduction, X_umap_first_reduction, hdbscan_clusterer, clusters, dbcv_score # Return the UMAP model and the clusters

def create_topics(cluster, embeddings, texts):

    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    
    # Initialize BERTopic with no embedding model
    topic_model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
        verbose=True
    )
    # Use fit_transform with my own clusters
    topics, probs = topic_model.fit_transform(texts, embeddings, y=cluster)

    return topic_model
    #return topics, probs


# Cluster visualisations

def scatter_vis(embeddings, clusters, cluster_colors, title, save_name, dr_model):    
    # Mailny used for multiple graphs, ensures a consistent colour for each cluster
    unique_cluster_ids = set(sorted(clusters))
    num_clusters = len(unique_cluster_ids)
    color_norm = mcolors.Normalize(vmin=min(unique_cluster_ids), vmax=max(unique_cluster_ids))
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap='tab20')
    cluster_colors = {cid: scalar_map.to_rgba(cid) for cid in unique_cluster_ids}
    
    fig, ax = plt.subplots(figsize=(4, 3))
    if dr_model == 'umap':
        umap_subplot_vis(embeddings, clusters, cluster_colors, title, ax)
    elif dr_model == 'tsne':
        tsne_subplot_vis(embeddings, clusters, cluster_colors, title, ax)
    fig.tight_layout()
    fig.savefig(save_name, format='pdf')
    fig.show()

# Visualise the results
def tree_vis(hdbscan_clusterer):
    hdbscan_clusterer.condensed_tree_.plot(select_clusters=True)

def umap_subplot_vis(X_umap_first_reduction, clusters, cluster_colors, title, ax):
    # Need to reduce dimensions to 2D for this
    # Define unique cluster labels (excluding noise points, which are labeled as -1)
    unique_clusters = set(clusters)
    #cluster_to_idx = {cid: idx for idx, cid in enumerate(unique_clusters)} # This ensure the colours work
    
    # UMAP
    umap_model = umap.UMAP(n_neighbors=n_neighbors,
                           n_components=2,
                           random_state=42,
                           metric='cosine')
    X_umap = umap_model.fit_transform(X_umap_first_reduction)
    
    # Plot the HDBSCAN
    #plt.figure(figsize=(5, 4))
    # Create a color map for clusters (-1 for noise will be gray)
    #colors = plt.get_cmap("tab20", len(unique_clusters))  # 'tab10' provides distinct colors
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        #idx = cluster_to_idx[cluster_id]
        color = "gray" if cluster_id == -1 else cluster_colors.get(cluster_id)  # Gray for noise points
        ax.scatter(X_umap[mask, 0],
                    X_umap[mask, 1],
                    label=f"Cluster {cluster_id}" if cluster_id != -1 else "Outliers", 
                    color=color,
                    alpha=0.7,
                    edgecolors="k",
                    linewidth=0.5)

    ax.set_title(f"{title} \nClusters: {len(unique_clusters)}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True)
    #ax.legend()

def tsne_subplot_vis(X_umap_first_reduction, clusters, cluster_colors, title, ax):
    # Need to reduce dimensions to 2D for this
    # Define unique cluster labels (excluding noise points, which are labeled as -1)
    unique_clusters = set(clusters)
    #cluster_to_idx = {cid: idx for idx, cid in enumerate(unique_clusters)} # This ensure the colours work
    perplexity = 30
    
    #t-SNE
    tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne_model.fit_transform(X_umap_first_reduction)
    
    # Plot clusters
    #plt.figure(figsize=(5, 4))
    # Create a color map for clusters (-1 for noise will be gray)
    #colors = plt.get_cmap("tab20", len(unique_clusters))  # 'tab10' provides distinct colors
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        #idx = cluster_to_idx[cluster_id]
        color = "gray" if cluster_id == -1 else cluster_colors.get(cluster_id)  # Gray for noise points
        ax.scatter(X_tsne[mask, 0],
                    X_tsne[mask, 1],
                    label=f"Cluster {cluster_id}" if cluster_id != -1 else "Outliers", 
                    color=color,
                    alpha=0.7,
                    edgecolors="k",
                    linewidth=0.5)
    
    ax.set_title(f"{title} \nClusters: {len(unique_clusters)}")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    #ax.legend()
    ax.grid(True)

"""
@article{JMLR:v22:20-1061,
  author  = {Yingfan Wang and Haiyang Huang and Cynthia Rudin and Yaron Shaposhnik},
  title   = {Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {201},
  pages   = {1-73},
  url     = {http://jmlr.org/papers/v22/20-1061.html}
}
"""
def pacmap_vis(X_umap_first_reduction, clusters, file_name):
    # Need to reduce dimensions to 2D for this
    # Define unique cluster labels (excluding noise points, which are labeled as -1)
    unique_clusters = set(clusters)
    cluster_to_idx = {cid: idx for idx, cid in enumerate(unique_clusters)}
    
    # paCMAP
    pacmap_model = pacmap.PaCMAP(n_components=2)
    X_pacmap = pacmap_model.fit_transform(X_umap_first_reduction)
    
    # Plot clusters
    plt.figure(figsize=(5, 4))
    # Create a color map for clusters (-1 for noise will be gray)
    colors = plt.get_cmap("tab20", len(unique_clusters))  # 'tab10' provides distinct colors
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        idx = cluster_to_idx[cluster_id]
        color = "gray" if cluster_id == -1 else colors(idx)  # Gray for noise points
        plt.scatter(X_pacmap[mask, 0],
                    X_pacmap[mask, 1],
                    label=f"Cluster {cluster_id}" if cluster_id != -1 else "Outliers", 
                    color=color,
                    alpha=0.7,
                    edgecolors="k",
                    linewidth=0.5)
    
    plt.xlabel("paCMAP Component 1")
    plt.ylabel("paCMAP Component 2")
    plt.title("paCMAP")
    #plt.legend()
    plt.grid(True)
    plt.savefig(file_name, format='pdf')
    plt.show()

def trimap_vis(X_umap_first_reduction, clusters, file_name):
    # Need to reduce dimensions to 2D for this
    # Define unique cluster labels (excluding noise points, which are labeled as -1)
    unique_clusters = set(clusters)
    cluster_to_idx = {cid: idx for idx, cid in enumerate(unique_clusters)}
    n_inliers = 15
    
    # TriMap
    trimap_model = trimap.TRIMAP(n_inliers=n_inliers, n_dims=2) # n_outliers=5, n_random=5
    X_trimap = trimap_model.fit_transform(X_umap_first_reduction)
    
    # Plot clusters
    plt.figure(figsize=(5, 4))
    # Create a color map for clusters (-1 for noise will be gray)
    colors = plt.get_cmap("tab20", len(unique_clusters))  # 'tab10' provides distinct colors
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        idx = cluster_to_idx[cluster_id]
        color = "gray" if cluster_id == -1 else colors(idx)  # Gray for noise points
        plt.scatter(X_trimap[mask, 0],
                    X_trimap[mask, 1],
                    label=f"Cluster {cluster_id}" if cluster_id != -1 else "Outliers", 
                    color=color,
                    alpha=0.7,
                    edgecolors="k",
                    linewidth=0.5)
    
    plt.xlabel("TriMap  Component 1")
    plt.ylabel("TriMap  Component 2")
    plt.title(f"TriMap (Nearest Neighbours = {n_inliers})")
    #plt.legend()
    plt.grid(True)
    plt.savefig(file_name, format='pdf')
    plt.show()

def count_groupby(lst):
    counts = {}  # Dictionary to store counts
    for item in lst:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    total_count = len(lst)  # Total number of elements
    return counts, total_count
    
def groupby_vis(X_umap_first_reduction, clusters):
    
    # Example usage
    data = clusters
    grouped_counts, total = count_groupby(data)
    
    # Plot the results
    plt.figure(figsize=(5, 4))
    plt.bar(grouped_counts.keys(), grouped_counts.values(), color=['red', 'yellow', 'orange'])
    plt.xlabel("Items")
    plt.ylabel("Count")
    plt.title("Item Frequency in List")
    plt.show()

def run_all_vis(embeddings_filtered, clusters, save_name):
    """
    save_name: what name to include in the save file
    """    
    umap_vis(embeddings_filtered, clusters, 'cluster_umap_' + save_name + '.pdf')
    pca_vis(embeddings_filtered, clusters, 'cluster_pca_' + save_name + '.pdf')
    tsne_vis(embeddings_filtered, clusters, 'cluster_tsne_' + save_name + '.pdf')
    pacmap_vis(embeddings_filtered, clusters, 'cluster_pacmap_' + save_name + '.pdf')
    trimap_vis(embeddings_filtered, clusters, 'cluster_trimap_' + save_name + '.pdf')
    groupby_vis(embeddings_filtered, clusters)

def run_all_vis_grid(embeddings_filtered, clusters, n_neighbors=15):
    reductions = {
        "UMAP": umap.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42, metric='cosine').fit_transform,
        "PCA": PCA(n_components=2).fit_transform,
        "t-SNE": lambda X: TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X),
        "paCMAP": pacmap.PaCMAP(n_components=2).fit_transform,
    }

    unique_clusters = set(clusters)
    cluster_to_idx = {cid: idx for idx, cid in enumerate(unique_clusters)}
    colors = plt.get_cmap("tab20", len(unique_clusters))

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.flatten()

    for ax, (name, reducer) in zip(axes, reductions.items()):
        X_reduced = reducer(embeddings_filtered)

        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            idx = cluster_to_idx[cluster_id]
            color = "gray" if cluster_id == -1 else colors(idx)
            ax.scatter(X_reduced[mask, 0],
                       X_reduced[mask, 1],
                       label=f"Cluster {cluster_id}" if cluster_id != -1 else "Outliers",
                       color=color,
                       alpha=0.7,
                       edgecolors="k",
                       linewidth=0.5)

        ax.set_title(name)
        ax.set_xlabel(f"{name} 1")
        ax.set_ylabel(f"{name} 2")
        ax.grid(True)

    # Only one legend outside
    handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.show()


# Load sentence transformer model for the embeddings and create tweet embeddings
def embed_tweets(embedding_model_id, tweet_text):
    embedding_model = SentenceTransformer(embedding_model_id)
    # Convert series to list if needed
    if isinstance(tweet_text, pd.Series):
        tweet_text = tweet_text.tolist()
    embeddings = embedding_model.encode(tweet_text, show_progress_bar=False)  # tweet_text = tweet_text_clean

    # Normalise so all rows have length 1, this is common for clustering and using cosine similarity/other distance metrics
    #embeddings = normalize(embeddings)

    return embeddings


# Train DBCV for global clustering, so cluster once and not daily

def compute_dbcv(X, labels):
    """
    Compute the DBCV score for a clustering solution.
    Higher scores indicate better clustering quality.
    """
    if len(set(labels)) <= 1:  # Ensure more than one cluster is found
        return -1
    return validity_index(X, labels)


# Run clustering process for daily clustering
def run_cluster_process_with_global_cluster_id(n_neighbors,
                                               min_dist,
                                               min_cluster_size,
                                               min_samples,
                                               n_components,
                                               df):
    
    #df = df[df['DateTimeStr'].isin(['2019-03-20', '2019-03-18'])]
    #df['DateTimeDt'] = pd.to_datetime(df['DateTimeStr']).dt.date
    #df['embeddings'] = list(embeddings)
    
    # Threshold to consider two medoids as the same topic
    SIMILARITY_THRESHOLD = 0.70

    print(f'Running clustering and matching on global cluster ID where medoid similarity is {SIMILARITY_THRESHOLD}')
    print('Parameters: ')
    print(f'Nearest neighbors: {n_neighbors}')
    print(f'Minimum distance {min_dist}')
    print(f'Minimum cluster size {min_cluster_size}')
    print(f'Minimum samples {min_samples}')
    print(f'Number of components {n_components}')
    
    # Store global medoids and ID counter
    global_medoids = {}
    next_cluster_id = 0
    
    df['global_cluster_id'] = -1  # Initialize
    
    # Store values
    records = []
    daily_clusters = {}
    
    for day, group in df.groupby('DateTimeDt'):
        print(f"Processing date: {day}")
    
        if len(group) < 40:
            continue
    
        day_embeddings = np.vstack(group['embeddings'].values)
        umap_model_first_reduction, X_umap_first_reduction, hdbscan_clusterer, labels, dbcv_score = cluster_embeddings(day_embeddings,
                                                                                       n_neighbors,
                                                                                       min_dist,
                                                                                       min_cluster_size,
                                                                                       min_samples,
                                                                                       n_components
                                                                                      )
        df.loc[group.index, 'cluster'] = labels
    
        # Compute and store cluster medoids
        clusters = {}
        for label in set(labels):
            if label == -1:
                continue
    
            cluster_mask = labels == label
            cluster_embs = day_embeddings[cluster_mask]
            # This uses centroid
            #centroid = cluster_embs.mean(axis=0).reshape(1, -1)
            # This uses medoid
            distance_matrix = pairwise_distances(cluster_embs, metric='cosine')
            medoid_index = distance_matrix.sum(axis=1).argmin()
            medoid = cluster_embs[medoid_index].reshape(1, -1)

    
            # Check against global medoids
            assigned_cluster_id = None
            if global_medoids:
                all_global_medoids = np.vstack(list(global_medoids.values()))
                similarities = cosine_similarity(medoid, all_global_medoids)[0] # Changed
                max_sim_idx = np.argmax(similarities)
                if similarities[max_sim_idx] > SIMILARITY_THRESHOLD:
                    assigned_cluster_id = list(global_medoids.keys())[max_sim_idx]
    
            # If no match, assign new global_cluster_id
            if assigned_cluster_id is None:
                assigned_cluster_id = next_cluster_id
                global_medoids[next_cluster_id] = medoid # Changed
                next_cluster_id += 1
    
            # Assign to tweets in DataFrame
            df.loc[group.index[cluster_mask], 'global_cluster_id'] = assigned_cluster_id
    
            # Store cosine distance to daily medoid
            distances = cosine_distances(cluster_embs, medoid).flatten() # Medoid
            df.loc[group.index[cluster_mask], 'cosine_distance_to_medoid'] = distances
    
            # Store daily cluster info
            clusters[assigned_cluster_id] = medoid # Changed
    
        daily_clusters[day] = clusters
    
        records.append({
            'DateTimeDt': day,
             'DBCV': dbcv_score,
        })
    
        df_cluster_quality = pd.DataFrame(records)

    arr_all_global_medoids = all_global_medoids
    #print(f'Cluster list: {df["global_cluster_id"].unique()}')
    #print(f'Cluster shape: {all_global_centroids.shape}')

    number_clusters = df['global_cluster_id'].unique()
    print('Completed!')
    print(f'Clusters created: {number_clusters}')
    return df, df_cluster_quality, arr_all_global_medoids

# Top n c-TF-IDF words per cluster
def ctfiidf_top_words(df,group_col="global_cluster_id",text_col="tweet_text_clean",top_n=5,ngram_range=(1,2),min_df=1,):
    # Add some stop words and other words that do not add much value to the list
    #stop_words="english"
    stopwords_additions = ['http',
                       'https',
                       'amp',
                       'com',
                       'rt',
                       'hurricane',
                       'peurto',
                       'rico',
                       'maria',
                       'puerto',
                       'hurricanemaria',
                       'need',
                       'puertorico',
                       'marias', 
                       'hurricaneirma',
                       'juan',
                       'san',
                       'san juan',
                       'irma',
                       'juans',
                       'please',
                       'us',
                       'ricos',
                       'along',
                       'help',
                       '']
    stop_words = list(stopwords.words('english')) + stopwords_additions

    # Put words together
    grouped = (
        df.groupby(group_col, as_index=False)[text_col]
          .apply(lambda s: " ".join(map(str, s)))
          .rename(columns={text_col: "doc"})
    )

    # Term counts per cluster
    vectorizer = CountVectorizer(ngram_range=ngram_range,
                                 min_df=min_df,
                                 stop_words=stop_words)
    X = vectorizer.fit_transform(grouped["doc"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # c-TF-IDF calculation
    tf = normalize(X, norm="l1", axis=1) # L1 normalise rows (class-level TF)
    df_term = np.asarray((X > 0).sum(axis=0)).ravel()  # in how many groups term appears
    n_groups = X.shape[0]
    idf = np.log((n_groups + 1) / (df_term + 1)) + 1.0  # smooth

    ctfidf = tf.multiply(idf)  # (n_groups x n_terms), sparse

    # Top-n per group
    top_words = {}
    for row_idx, group_key in enumerate(grouped[group_col].tolist()):
        row = ctfidf.getrow(row_idx).toarray().ravel()
        if row.sum() == 0:
            top_words[group_key] = []
            continue
        top_idx = np.argpartition(-row, range(min(top_n, row.size)))[:top_n]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        words = feature_names[top_idx]
        scores = row[top_idx]
        top_words[group_key] = list(zip(words, scores))

    return top_words

# Bar graph
def plot_cluster_bars(top_words_dict, title_prefix="Cluster", cols=4, figsize=(14, 8)):
    clusters = sorted(top_words_dict.keys())
    n = len(clusters)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, cid in zip(axes, clusters):
        pairs = top_words_dict[cid]
        if not pairs:
            ax.axis("off")
            continue

        words, scores = zip(*pairs)
        color = cluster_colors.get(cid, 'gray')
        ax.barh(range(len(words)), scores, color = color, edgecolor = 'black', alpha = 0.9)
        ax.set_xlim(0,0.28)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_title(f"{title_prefix} {cid}")
        ax.invert_yaxis()
        
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5)

    # Hide leftover axes
    for ax in axes[len(clusters):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("cluster_bar_test_tuned.pdf", format='pdf')
    plt.show()

# Output sample text summaries for comparisons
# Parameters: 
# summary_model_id_list - a list of LLMs that will be compared
# cluster_list - a list of clusters that will be compared
def output_summaries(summary_model_id_list = None, cluster_list = None):
    if cluster_list is None:
        cluster_list = df_events_all_group["global_cluster_id"].unique().tolist()

    if summary_model_id_list is None:
        summary_model_id_list = [
            col for col in df_events_all_group.columns
            if col.startswith("tweet_summary") or col.startswith("human_summary")
        ]
    
    df_events_all_group_filtered = df_events_all_group[df_events_all_group['global_cluster_id'].isin(cluster_list)]
    df_events_all_group_filtered.sort_values(by=['DateTimeDt', 'global_cluster_id', 'event_type'],ascending=[False, True, True])
    
    for idx, row in df_events_all_group_filtered.iterrows():
        event_date = row["DateTimeDt"]
        cluster = row["global_cluster_id"]
        event_type = row["event_type"]
        text = row["tweet_text_clean"]

        if event_type == 'new_event':
            event_type_print = 'New Event'
        elif event_type == 'event_spike':
            event_type_print = 'Spike Event'
        else:
            event_type_print = event_type

        print("Event Date: ", event_date)
        print("Event Type: ", event_type_print)
        print("Cluster: ", cluster)
        print("-" * 10)
        print("Original Tweets: ", text)
        print("-" * 10)

        # Print each summary model column
        for col in summary_model_id_list:
            print(f"{col}: {row[col]}")
            print("-" * 10)

        print("End")
        print("-" * 80)

# Run the LLM summariser
def summarise_text(text, model, tokenizer, model_type="t5"):
    if model_type == "t5":
        # Optionally truncate text
        max_input_length = 512
        words = text.split()
        if len(words) > max_input_length - 5:
            text = ' '.join(words[:max_input_length])
        
        # T5 expects prefix "summarize: "
        input_text = "summarize: " + text

        inputs = tokenizer.encode(
            input_text, 
            return_tensors='pt', 
            max_length=400, 
            truncation=True
        )
        output = model.generate(
            inputs,
            # T5 base settings used from https://s3.amazonaws.com/models.huggingface.co/bert/t5-base-config.json
            length_penalty=5,
            min_length=20,
            max_length=400,
            num_beams=4, #https://www.reddit.com/r/LanguageTechnology/comments/igz9ul/what_are_beams_and_how_does_their_number_affect_a/
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        summary = tokenizer.decode(output[0], skip_special_tokens=True)

    elif model_type == "pegasus":
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            length_penalty=2.0,
            max_length=60,
            min_length=10,
            no_repeat_ngram_size=3,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    elif model_type == "bart":
        inputs = tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
        outputs = model.generate(
            inputs['input_ids'],
            num_beams=4,
            length_penalty=2.0,
            max_length=130,
            min_length=30,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary



# Get ROUGE and BLEU scores
def rouge_bleu_scores(text_references, text_candidates):
    bleu_results = bleu_metric.compute(predictions=text_candidates, references=text_references)
    rouge_results = rouge_metric.compute(predictions=text_candidates, references=text_references)
    return bleu_results, rouge_results

# Get the Maximal Marginal Relevance (MMR)
def mmr(embeddings, medoid_embedding, lambda_param=0.7, top_k=5):
    relevance = cosine_similarity(embeddings, medoid_embedding.reshape(1, -1)).flatten()
    selected = []
    remaining = list(range(len(embeddings)))

    # MMR rank 1 will always be closest to the medoid
    first = np.argmax(relevance)
    selected.append(first)
    remaining.remove(first)

    for _ in range(top_k - 1):
        mmr_scores = []
        for idx in remaining:
            sim_to_medoid = relevance[idx]
            sim_to_selected = max(cosine_similarity(
                embeddings[idx].reshape(1, -1),
                embeddings[selected]
            ).flatten())
            mmr_score = lambda_param * sim_to_medoid - (1 - lambda_param) * sim_to_selected
            mmr_scores.append((idx, mmr_score))

        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def open_ai_summarise_text(text_to_summarize):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize the following in the third person like a news report:\n{text_to_summarize}"}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    # Get the summary
    summary = response.choices[0].message.content
    return summary


def summarise_each_llm(df, text_input, text_output, model_id, model_family):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, return_dict=True)
    df[text_output] = df[text_input].apply(
        lambda x: summarise_text(x, model, tokenizer, model_type=model_family)
    )
    print(f'{model_id} done!')
    return df

# Preprocess
def preprocess(preprocess_sample, model_family=None):
    max_input_length = 512
    max_target_length = 128

    # T5 models only
    if model_family == 't5':
        text_input = "summarize: " + preprocess_sample["input_text"]
    elif model_family == 'T5':
        text_input = "summarize: " + preprocess_sample["input_text"]
    else:
        text_input = preprocess_sample["input_text"]
    inputs = tokenizer(
        text_input,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    targets = tokenizer(
        preprocess_sample["target_text"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0],
        "labels": targets["input_ids"][0],
    }


def build_dpo_dataset(df_human_rlhf):
    data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }

    for _, row in df_human_rlhf.iterrows():
        if row["reverse_better"]:
            data["prompt"].append(row["tweet_text_clean"])
            data["chosen"].append(row["summary_text_reverse"])
            data["rejected"].append(row["summary_text"])
        else:
            data["prompt"].append(row["tweet_text_clean"])
            data["chosen"].append(row["summary_text"])
            data["rejected"].append(row["summary_text_reverse"])

    return Dataset.from_dict(data)

# We have already built summarisation Python code, however it is very slow, so use some AI inspired code to help us improve speed
def load_summarizer(model_id, model_family="auto", device=None):
    """Load once, reuse."""
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tok, model, device, model_family.lower()

@torch.no_grad()
def summarise_dataframe(df, input_col, output_col, tok, model, device,
                        model_family="auto", batch_size=16,
                        max_input_length=512, max_new_tokens=128,
                        num_beams=4, do_sample=False):
    """Batch inference; writes summaries to df[output_col]."""
    texts = df[input_col].astype(str).tolist()
    summaries = []

    # Optional prefix for T5
    use_t5_prefix = model_family.lower() == "t5"
    prefix = "summarize: " if use_t5_prefix else ""

    for i in tqdm(range(0, len(texts), batch_size), desc="Summarizing"):
        batch_texts = [prefix + t for t in texts[i:i+batch_size]]
        enc = tok(batch_texts,
                  max_length=max_input_length,
                  padding=True,
                  truncation=True,
                  return_tensors="pt").to(device)

        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample
            )

        batch_summaries = tok.batch_decode(out, skip_special_tokens=True)
        summaries.extend(batch_summaries)

    df[output_col] = summaries
    return df


# In[255]:


# Define the objective function to optimize
# Global
@use_named_args(space)
def objective_global(n_neighbors, min_dist, min_cluster_size, min_samples, n_components):

    umap_model_first_reduction, X_umap_first_reduction, hdbscan_clusterer, clusters, dbcv_score = cluster_embeddings(embeddings,n_neighbors,min_dist,min_cluster_size,min_samples,n_components)

    # Compute CBDV Score (only if there are at least 2 clusters)
    valid_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    # Compute DBCV Score (Density-Based Clustering Validation Score)
    if valid_clusters > 1:
        dbcv_score = validity_index(X_umap_first_reduction.astype(np.float64), clusters)
        if dbcv_score is None:
            dbcv_score =  1e6
    else:
        dbcv_score = 1e6  # or some large positive value to indicate a bad score

    # Return DBCV score
    return -dbcv_score


# Tasks
# UMAP to reduce dimensions from about 1,000D to about 50D, this ensure we preserve as
# much informaiton as possible
# HDBSCAN to custer
# Reduce to 2D for visalisation, can use UMAP, PCA, t-SNE

# We reduce the to 50D dataset and not the original dataset, this is so we: 
# 1) preserve cluster structur as HDBSCAN was applied to the 50D dataset and this contains
# the most meaningful clusture structure, reducing this to 2D ensures we are visualising
# the HDBSCAN structure that was actually used
# 2) Avoid information loss, if we reduce the original high-dimension dataset (before UMAP)
# directly to 2D we lose important patterns that UMAP had captured in the 50D representation

# Define the objective function to optimize
# Daily
@use_named_args(space)
def objective_daily(n_neighbors, min_dist, min_cluster_size, min_samples, n_components):

    dbcv_score_list = []

    unique_days = df_data['DateTimeStr'].unique()
    sampled_days = random.sample(list(unique_days), 2) # Select 2 random days for processing

    # Run a for loop to check some or all days, this ensures we have a good DBCV score acorss all or some days
    for day_data in sampled_days:
        group = df_data[df_data['DateTimeStr'] == day_data] # group is the subset of dataframe for this day
        day_embeddings = np.vstack(group['embeddings'].values)
        print(f"Processing date: {day_data}")

        # Check number of samples and skip day if this is below n_components
        n_samples = day_embeddings.shape[0]
        if n_samples <= n_components:
            print(f"Skipping {day_data}: n_samples ({n_samples}) <= n_components ({n_components})")
            continue
        
        umap_model_first_reduction, X_umap_first_reduction, hdbscan_clusterer, clusters, dbcv_score = cluster_embeddings(day_embeddings, n_neighbors,min_dist,min_cluster_size,min_samples,n_components)

        # Compute CBDV Score (only if there are at least 2 clusters)
        valid_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        # Compute DBCV Score (Density-Based Clustering Validation Score)
        if valid_clusters > 1:
            dbcv_score = validity_index(X_umap_first_reduction.astype(np.float64), clusters)
            if dbcv_score is None:
                dbcv_score =  1e6
        else:
            dbcv_score = 1e6  # or some large positive value to indicate a bad score

        dbcv_score_list.append(dbcv_score)

    # Return DBCV score (higher is better)
    # Option: 1) best of the worst performance 2) average performance
    return -np.mean(dbcv_score_list)  # Maximize average DBCV
    #return -np.min(dbcv_score_list)  # Maximize worst-case DBCV


# #### Import data
# 
# - Train: used to train the models
# - Dev: used to tune the hyper-parameters
# - Test: final testing on the live data

# In[69]:


# Directories
os.chdir(r'C:\Users\mail\OneDrive\University\04 Bristol\07 Project')
current_dir = os.path.dirname(os.path.abspath('project.py'))
module_dir = os.path.join(current_dir, 'modules')
data_dir = os.path.join(current_dir, r'Code\data')
visualisations_dir = os.path.join(current_dir, r'Code\visualisations')
models_dir = os.path.join(current_dir, r'Code\models')

# Import the data, this is in TSV (Tab Separated Values) format
file_address_train = os.path.join(data_dir, r'HumAID_data_events_set1_47K\events_set1\hurricane_maria_2017\hurricane_maria_2017_train.tsv')
file_address_dev = os.path.join(data_dir, r'HumAID_data_events_set1_47K\events_set1\hurricane_maria_2017\hurricane_maria_2017_dev.tsv')
file_address_test = os.path.join(data_dir, r'HumAID_data_events_set1_47K\events_set1\hurricane_maria_2017\hurricane_maria_2017_test.tsv')

df_data_train = pd.read_csv(file_address_train, sep='\t')
df_data_dev = pd.read_csv(file_address_dev, sep='\t')
df_data_test = pd.read_csv(file_address_test, sep='\t')


# #### Data Cleanse
# Get Date Time from Snowflafe ID
# Preprocess Tweets
# 
# Get Unix timestamp by inputting a twitter snowflake ID
# tstamp = ta.find_tweet_timestamp(721630546711986178)
# 
# Get a date format datetime by inputting a Unix timestamp
# utcdttime = datetime.utcfromtimestamp(tstamp / 1000)

# In[70]:


# Add tweet date time in Unix format
add_date_time_unix(df_data_train)
add_date_time_unix(df_data_dev)
add_date_time_unix(df_data_test)

# Add tweet date in string format
add_date_str(df_data_train)
add_date_str(df_data_dev)
add_date_str(df_data_test)

# Add tweet date and time in string format
add_date_time(df_data_train)
add_date_time(df_data_dev)
add_date_time(df_data_test)

df_data_train['DateTime'] = pd.to_datetime(df_data_train['DateTime'], format='%Y-%m-%d, %H:%M:%S')
df_data_train['DateTimeHour'] = df_data_train['DateTime'].dt.hour
df_data_train['DateTimeDt'] = pd.to_datetime(df_data_train['DateTimeStr'])
df_data_train['DateTimeDayWeek'] = df_data_train['DateTimeDt'].dt.day_name()

df_data_dev['DateTime'] = pd.to_datetime(df_data_dev['DateTime'], format='%Y-%m-%d, %H:%M:%S')
df_data_dev['DateTimeHour'] = df_data_dev['DateTime'].dt.hour
df_data_dev['DateTimeDt'] = pd.to_datetime(df_data_dev['DateTimeStr'])
df_data_dev['DateTimeDayWeek'] = df_data_dev['DateTimeDt'].dt.day_name()

df_data_test['DateTime'] = pd.to_datetime(df_data_test['DateTime'], format='%Y-%m-%d, %H:%M:%S')
df_data_test['DateTimeHour'] = df_data_test['DateTime'].dt.hour
df_data_test['DateTimeDt'] = pd.to_datetime(df_data_test['DateTimeStr'])
df_data_test['DateTimeDayWeek'] = df_data_test['DateTimeDt'].dt.day_name()


# #### Preprocessing and Normalisation of Tweets
# 
# Common preprocessing methods for social media text
# - Lowercase
# - Remove punctuation
# - Remove stopwords
# - Remove URLs or replace with standard token, this is a form of normalisation, e.g. <url>
# - Remove HTML tags
# - Known social media features, e.g. @user, RT, #hashtags
# - Consider #hashtag words and not just replace them, e.g. #whataday to what a day
# - Change emojis to words
# - Change emoticons to words
# - Lemmatisation
# - Stemming
# - Remove numbers
# - Tokenisation
# 
# References
# - https://developers.google.com/edu/python/regular-expressions
# - https://github.com/DavidBert/Tweet_normalizer/blob/master/normalize_tweets.py
# - https://github.com/XuanyiZ/Text-Normalization

# In[71]:


# Pre-process each dataset
df_data_train['tweet_text_clean'] = df_data_train.apply(lambda row: custom_preprocessing(row['tweet_text']), axis=1)
df_data_dev['tweet_text_clean'] = df_data_dev.apply(lambda row: custom_preprocessing(row['tweet_text']), axis=1)
df_data_test['tweet_text_clean'] = df_data_test.apply(lambda row: custom_preprocessing(row['tweet_text']), axis=1)

# Concatenate into the datasets we need
df_data = pd.concat([df_data_train, df_data_dev], ignore_index=True)
df_data_all = pd.concat([df_data, df_data_test], ignore_index=True)

train_dev_count = df_data['tweet_text'].count()
all_count = df_data_all['tweet_text'].count()

print(f'{train_dev_count:,} train and dev tweets imported!')
print(f'{all_count:,} train, dev, and test tweets imported!')


# In[72]:


tweet_list = df_data_all['tweet_text'].tolist()

# Emojis
counts = [preprocessing_count_emoji(t) for t in tweet_list]
total_emojis = sum(counts)
tweets_with_emoji = sum(1 for count in counts if count > 0)
print("Total emojis used:", total_emojis)
print("Tweets containing at least one emoji:", tweets_with_emoji)

# Usernames
counts = [preprocessing_count_retweet_username(t) for t in tweet_list]
total_usernames = sum(counts)
tweets_usernames = sum(1 for count in counts if count > 0)
print("Total usernames used:", total_usernames)
print("Tweets containing at least one usernames:", tweets_usernames)

# Retweet usernames
counts = [preprocessing_count_retweet_username(t) for t in tweet_list]
total_retweet_usernames = sum(counts)
tweets_retweet_usernames = sum(1 for count in counts if count > 0)
print("Total retweet usernames used:", total_retweet_usernames)
print("Tweets containing at least one retweet usernames:", tweets_retweet_usernames)

# URLs
counts = [preprocessing_count_urls(t) for t in tweet_list]
total_urls = sum(counts)
tweets_urls = sum(1 for count in counts if count > 0)
print("Total URLs used:", total_urls)
print("Tweets containing at least one URL:", tweets_urls)

# Digits
counts = [preprocessing_count_digits(t) for t in tweet_list]
total_digits = sum(counts)
tweets_digits = sum(1 for count in counts if count > 0)
print("Total digits used:", total_digits)
print("Tweets containing at least one digit:", tweets_digits)


# In[715]:


# Test pre-processing
output_tweet_text(910571399399559168, df_data_all)
output_tweet_text(910535316544610305, df_data_all)
output_tweet_text(910570785047228416, df_data_all)
output_tweet_text(910614483717914624, df_data_all)
output_tweet_text(910712228680155136, df_data_all)
output_tweet_text(914138245839060992, df_data_all)
output_tweet_text(912272394974265344, df_data_all)

# Check non-ASCII characters
all_text = ''.join(df_data_all['tweet_text_clean'].dropna())
unicode_chars = set(re.findall(r'[^\x00-\x7F]', all_text))
print("Unique Unicode characters found:", unicode_chars)


# #### Tweet Analysis Visualisations
# 
# We will output graphs such as distribution of word counts, this will help with: 
# 
# - Identify if long tweets, as long tail
# - Distribution is normal, skewed or bimodal
# - Threshold for chunking, trimming or padding

# In[177]:


# Example: assuming df is your DataFrame
# Calculate word counts
df_data_all['word_count'] = df_data_all['tweet_text'].str.split().apply(len)
df_data_all['word_count_clean'] = df_data_all['tweet_text_clean'].str.split().apply(len)
mean_val = df_data_all['word_count'].mean()
std_val = df_data_all['word_count'].std()

# Plot histogram
plt.figure(figsize=(5, 3))
plt.hist(df_data_all['word_count'], bins=30, edgecolor='black', color='skyblue', alpha=0.75)
#plt.hist(df_data['word_count_clean'], bins=30, edgecolor='black', color='pink', alpha=0.5)

# Add lines for mean and ±1 standard deviation
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean = {mean_val:.1f}')
plt.axvline(mean_val + std_val, color='green', linestyle='dotted', linewidth=1, label=f'+1 SD = {mean_val + std_val:.1f}')
plt.axvline(mean_val - std_val, color='green', linestyle='dotted', linewidth=1, label=f'-1 SD = {mean_val - std_val:.1f}')
plt.axvline(mean_val + (2 * std_val), color='orange', linestyle='dotted', linewidth=1, label=f'+2 SD = {mean_val + std_val:.1f}')
plt.axvline(mean_val - (2 * std_val), color='orange', linestyle='dotted', linewidth=1, label=f'-2 SD = {mean_val - std_val:.1f}')

plt.title('Word Count Distribution of Tweets')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.xlim(0,40)
plt.ylim(0,1500)
plt.grid(axis='y', alpha=0.7)
plt.tight_layout()
plt.savefig("word_count_distribution.pdf", format='pdf')
plt.show()


# In[178]:


print(df_data_all['word_count'].describe())


# In[179]:


volume_per_day = df_data_all['DateTimeDayWeek'].value_counts().sort_index()
volume_per_day = volume_per_day.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.figure(figsize=(5, 3))
volume_per_day.plot(kind='bar', color='skyblue')
plt.title("Tweet Volume per Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[180]:


conditions = [
    (df_data_all['DateTimeHour'] >= 6) & (df_data_all['DateTimeHour'] < 12),
    (df_data_all['DateTimeHour'] >= 12) & (df_data_all['DateTimeHour'] < 18),
    (df_data_all['DateTimeHour'] >= 18) & (df_data_all['DateTimeHour'] < 24),
    (df_data_all['DateTimeHour'] >= 24) | (df_data_all['DateTimeHour'] < 6)
]
choices = ['Morning', 'Afternoon', 'Evening', 'Nighttime']
df_data_all['DateTimePeriod'] = np.select(conditions, choices, default='Unknown')

grouped = df_data_all.groupby(['DateTimeDayWeek', 'DateTimePeriod']).size().reset_index(name='count')
pivot_table = grouped.pivot(index='DateTimeDayWeek', columns='DateTimePeriod', values='count').fillna(0)

ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ordered_periods = ['Morning', 'Afternoon', 'Evening', 'Nighttime']
pivot_table = pivot_table.reindex(ordered_days)
pivot_table = pivot_table[ordered_periods]

ax = pivot_table.plot(kind='bar', stacked=True, figsize=(5, 4), colormap='Blues', edgecolor='black', linewidth=1)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title='Time Period', bbox_to_anchor=(1.05, 1), loc='upper left') # Reverse legend order
plt.title("Tweet Volume by Day of Week and Time Period")
plt.xlabel("Day of Week")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("tweet_by_day.pdf", format='pdf')
plt.show()


# In[181]:


# New for percentage
pivot_table_percent = pivot_table.div(pivot_table.sum(axis=1), axis=0)

ax = pivot_table_percent.plot(kind='bar', stacked=True, figsize=(5, 4), colormap='Blues', edgecolor='black', linewidth=1)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title='Time Period', bbox_to_anchor=(1.05, 1), loc='upper left') # Reverse legend order
plt.title("Tweet Distribution by Day of Week and Hour (100%)")
plt.xlabel("Day of Week")
plt.ylabel("Proportion of Tweets")
plt.tight_layout()
plt.savefig("tweet_distribution_by_day.pdf", format='pdf')
plt.show()


# In[506]:


test = df_data_all[df_data_all['DateTimeStr'] == '2017-09-26'].copy()
test


# In[182]:


# Notice there is a gap in the data, can write about using synthetic data [https://arxiv.org/abs/2310.07849]

# Count tweets per day
tweets_per_day = df_data_all.groupby(df_data_all['DateTimeDt'].dt.date).size().reset_index(name='tweet_count')

# Sort by date (just in case)
tweets_per_day = tweets_per_day.sort_values('DateTimeDt')

# Plot
plt.figure(figsize=(12, 5))
plt.plot(tweets_per_day['DateTimeDt'], tweets_per_day['tweet_count'], marker='o')
plt.title("Timeline of Hurricane Maria with Tweet Volume")
plt.xlabel("Date")
plt.ylabel("Number of Tweets")

# Add vertical dotted line on a specific date
highlight_date_1 = pd.to_datetime("2017-09-16") # Maria forms
plt.axvline(x=highlight_date_1, color='red', linestyle='--', linewidth=1)

highlight_date_2 = pd.to_datetime("2017-09-18") # Maria achieves category 5 strength
plt.axvline(x=highlight_date_2, color='red', linestyle='--', linewidth=1)

highlight_date_3 = pd.to_datetime("2017-09-20") # Maria makes landfall in Puerto Rico as a high-end Category 4 hurricane
plt.axvline(x=highlight_date_3, color='red', linestyle='--', linewidth=1)

highlight_date_4 = pd.to_datetime("2017-09-30") # Maria becomes extratropical
plt.axvline(x=highlight_date_4, color='red', linestyle='--', linewidth=1)

highlight_date_5 = pd.to_datetime("2017-10-02") # Maria dissipates
plt.axvline(x=highlight_date_5, color='red', linestyle='--', linewidth=1)

# Find y position for annotation (e.g., max y or fixed value)
y_value = tweets_per_day['tweet_count'].max() * 0.9

# Add annotation
plt.annotate(
    "Maria forms as\na tropical wave",
    xy=(highlight_date_1, y_value),   # Point to annotate
    xytext=(highlight_date_1 + pd.Timedelta(days=1), y_value + 100),  # Position of text
    arrowprops=dict(arrowstyle="->", color='black'),
    fontsize=10,
    color='black',
    ha='center'
)
plt.annotate(
    "Maria achieves\ncategory 5 strength",
    xy=(highlight_date_2, y_value),   # Point to annotate
    xytext=(highlight_date_2 + pd.Timedelta(days=1), y_value - 300),  # Position of text
    arrowprops=dict(arrowstyle="->", color='black'),
    fontsize=10,
    color='black',
    ha='center'
)
plt.annotate(
    "Maria made landfall and weakened\nto a high-end category 4 hurricane",
    xy=(highlight_date_3, y_value),   # Point to annotate
    xytext=(highlight_date_3 + pd.Timedelta(days=2), y_value + 100),  # Position of text
    arrowprops=dict(arrowstyle="->", color='black'),
    fontsize=10,
    color='black',
    ha='center'
)
plt.annotate(
    "Maria becomes\nextratropical",
    xy=(highlight_date_4, y_value),   # Point to annotate
    xytext=(highlight_date_4 + pd.Timedelta(days=1), y_value + 100),  # Position of text
    arrowprops=dict(arrowstyle="->", color='black'),
    fontsize=10,
    color='black',
    ha='center'
)
plt.annotate(
    "Maria dissipates",
    xy=(highlight_date_5, y_value),   # Point to annotate
    xytext=(highlight_date_5 + pd.Timedelta(days=1), y_value - 200),  # Position of text
    arrowprops=dict(arrowstyle="->", color='black'),
    fontsize=10,
    color='black',
    ha='center'
)

plt.xlim(pd.to_datetime('2017-09-15'), pd.to_datetime('2017-10-03'))
plt.ylim(0,1400)
plt.xticks(rotation=45)
#plt.gca().yaxis.tick_right()
#plt.gca().yaxis.set_label_position("right")
plt.grid(True)
plt.tight_layout()
plt.savefig("maria_timeline.pdf", format='pdf')
plt.show()


# In[183]:


# 1. Combine all tweet text into one string
all_text = ' '.join(df_data_all['tweet_text_clean'].dropna()).lower()

# 2. Tokenize (split into words, removing punctuation)
words = re.findall(r'\b\w+\b', all_text)

# 3. Count word frequencies
word_counts = Counter(words)

# 4. Convert to DataFrame
df_word_freq = pd.DataFrame(word_counts.items(), columns=['word', 'count'])
df_word_freq = df_word_freq.sort_values(by='count', ascending=False)

top_n = 20
top_words = df_word_freq.head(top_n)

plt.figure(figsize=(6, 4))
plt.bar(top_words['word'], top_words['count'], color='tab:blue')
plt.xticks(rotation=45)
plt.title(f"Top {top_n} Most Frequent Words")
#plt.xlabel("Word")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("word_frequency.pdf", format='pdf')
plt.show()


# In[184]:


#nltk.download('punkt')

# Clean and tokenize all tweets
all_text = ' '.join(df_data_all['tweet_text_clean'].dropna()).lower()
tokens = nltk.word_tokenize(all_text)

# Create bigrams (change 2 → 3 for trigrams, etc.)
bigram_list = list(ngrams(tokens, 2))

# Count bigrams
bigram_counts = Counter(bigram_list)

# Convert to DataFrame
df_bigrams = pd.DataFrame(bigram_counts.items(), columns=['bigram', 'count'])
df_bigrams = df_bigrams.sort_values(by='count', ascending=False)

# Select top N
top_n = 20
top_bigrams = df_bigrams.head(top_n)

# Convert bigram tuples to strings for display
top_bigrams['bigram_str'] = top_bigrams['bigram'].apply(lambda x: ' '.join(x))

# Plot
plt.figure(figsize=(6, 4))
plt.bar(top_bigrams['bigram_str'], top_bigrams['count'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title(f"Top {top_n} Most Frequent Bigrams")
plt.xlabel("Bigram")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# #### NLP Pipeline
# This is a sequence of interconnected steps to transform raw tweets into a
# desired output, this enables machines to better comprehend and understand
# human text.
# 
# Preprocessing has already been done, e.g. removing punctuation
# 
# BERTopic is a topic modelelling framework that allows users to create a
# customised topic model, it is flexible and modular, this means that models
# can be changed and updated easily.
# 
# - Tokenise tweets
# - Embedding of tweets - we need to choose a Sentence Transformer that offers high-quality dense sentence embeddings, good preservation of semantics and fast inference on dataset. Each sentece transformer has a dimensionality, this refers to the size of the vector representation (embedding) generated for each sentence, each sentence is converted into a dense numerical vector of fixed length, which is used for similarity and clustering. 
# - Reducing dimensionality of embeddings
# - Clustering reduced embeddings into topics
# - Tokenization of topics
# - Weight tokens
# - Represent topics with one or multiple representations (optional)

# ### Tweet Embeddings
# 
# Use Sentence Transformer for tweet embeddings, three options are considered: 
# - All MiniLM L6 V2 - fast, general-purpose semantic similarity and clustering—great for event detection/timelines
# - All MiniLM L12 V2 - higher accuracy without going full MPNet
# - All Mpnet Base V2 - best-in-class semantic performance, especially in semantic search, but be prepared for slower inference and higher resource needs
# 
# <img src="attachment:cf81a074-12ad-46fd-a682-70a1ef9c1814.png" width="60%" height="60%">
# 

# In[249]:


# Define the Tweet text to use

tweet_text = df_data['tweet_text_clean']
#tweet_text = df_data_test['tweet_text_clean']

# Define the Tweet date to use
tweet_date = df_data['DateTimeStr']
#tweet_date = df_data_test['DateTimeStr']

# Define embedding model for clustering
#embedding_model_id = 'all-MiniLM-L6-v2'
#embedding_model_id = 'all-MiniLM-L12-v2'
embedding_model_id = 'all-mpnet-base-v2'

# Embed the tweets
start = time.time()
embeddings = embed_tweets(embedding_model_id,tweet_text)
end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")

df_data['embeddings'] = list(embeddings)
#df_data_test['embeddings'] = list(embeddings)


# In[991]:


# Data
categories = ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'all-mpnet-base-v2']
values = [83, 164, 419]
colors = ['tab:blue', 'tab:cyan', 'tab:purple']

# Plot
plt.figure(figsize=(4, 3))
plt.bar(categories, values, width=0.4, color=colors)

# Labels and title
plt.xlabel('Word Embedding Models')
plt.ylabel('Runtime (Seconds)')
plt.ylim(0,500)
plt.xticks(rotation=45, ha='right')
plt.title('Embedding Model Runtime\nfor 7,278 Tweets')
#plt.grid()

for i, v in enumerate(values):
    plt.text(i, v + 0.5, str(v), ha='center')

plt.tight_layout()
plt.savefig("model_run_time.pdf", format='pdf')
plt.show()


# #### Compute CBDV
# 
# This calculates the Density-Based Clustering Validation (DBCV), which is a score designed for dentity-based clustering algorithms. We aim to maximise this score by running different parameters defined in a hyperperametre space, this will adjust the UMAP and HDBSCAN values. See below for the minimisation (maximisation) process. 

# #### Minimisation Process
# 
# Runa process to minimise the DBCV by applying different hyperperamaters, then use these best results for final test data. 
# 
# - Random Process miniisation (dummy_minimize)
# - Gaussian Process minimization (gp_minimize): Very smart, models the unknown function as a smooth surface using a "probabilistic model" (Gaussian Process), good for smooth hills and valleys
# - Random Forest minimization (forest_minimize): Models the function using a Random Forest (decision trees), good for noisy or non-smooth problems, wild jungle of random outcomes
# 
# #### Results
# - results.x	= The best parameter values found
# - results.fun = The best objective value (in your case, -DBCV, because you negated it)
# - results.func_vals = The list of all function evaluations (all -DBCV values tried)
# - results.x_iters = The list of all parameter values tried

# In[1057]:


# Train global

# Define hyperparameters search space for UMAP and HDBSCAN
space_global = [
    Integer(10, 50, name='n_neighbors'),      # UMAP parameter
    Real(0.00, 0.1, name='min_dist'),       # UMAP parameter
    Integer(45, 100, name='n_components'),  # UMAP parameter, dimensions
    Integer(20, 50, name='min_cluster_size'), # HDBSCAN parameter
    #Integer(20, 50, name='max_cluster_size'), # HDBSCAN parameter
    Integer(10, 50, name='min_samples'),       # HDBSCAN parameter
]

result_forest = forest_minimize(objective_global, space_global, n_calls=10, random_state=42)

minimisation_results = result_forest

# Print the best parameters and the best DBCV score, these will be used for the final clustering generation
best_params = dict(zip([dim.name for dim in space_global], minimisation_results.x))
#print("Best parameters found: ", best_params)
print("Best DBCV score: ", -minimisation_results.fun) # Best DBCV score (since objective returned -dbcv, flip it back)

# Assign best parameters to variables for future use
n_neighbors = best_params.get('n_neighbors')
min_dist = best_params.get('min_dist')
min_cluster_size = best_params.get('min_cluster_size')
min_samples = best_params.get('min_samples')
n_components = best_params.get('n_components')

print("No. neighbours", n_neighbors)
print("Min distance", min_dist)
print("Min clusters", min_cluster_size)
print("Min samples", min_samples)
print("No. components", n_components)


# In[256]:


# Train daily

# Find best parameters using different methods
# Best parameters and scores for each iteration are kept in the results object
#result_random = dummy_minimize(objective, space, n_calls=10, random_state=42)
#result_gp = gp_minimize(objective, space, n_calls=10, random_state=42)

# Define hyperparameters search space for UMAP and HDBSCAN
space = [
    Integer(10, 50, name='n_neighbors'),      # UMAP parameter
    Real(0.00, 0.1, name='min_dist'),       # UMAP parameter
    Integer(45, 100, name='n_components'),  # UMAP parameter, dimensions
    Integer(5, 50, name='min_cluster_size'), # HDBSCAN parameter
    #Integer(20, 50, name='max_cluster_size'), # HDBSCAN parameter
    Integer(5, 50, name='min_samples'),       # HDBSCAN parameter
]

result_forest = forest_minimize(objective_daily, space, n_calls=10, random_state=42)

results = [
    #('random_results', result_random),
    #('result_gp', result_gp),
    ('result_forest', result_forest)
    ]

minimisation_results = result_forest

# Print the best parameters and the best DBCV score, these will be used for the final clustering generation
best_params = dict(zip([dim.name for dim in space], minimisation_results.x))
#print("Best parameters found: ", best_params)
print("Best DBCV score: ", -minimisation_results.fun) # Best DBCV score (since objective returned -dbcv, flip it back)

# Assign best parameters to variables for future use
n_neighbors = best_params.get('n_neighbors')
min_dist = best_params.get('min_dist')
min_cluster_size = best_params.get('min_cluster_size')
min_samples = best_params.get('min_samples')
n_components = best_params.get('n_components')

print("No. neighbours", n_neighbors)
print("Min distance", min_dist)
print("Min clusters", min_cluster_size)
print("Min samples", min_samples)
print("No. components", n_components)


# ##### Plot Fine-tuning of DBCV

# In[257]:


# Some fine tuning plots
# plot_convergence: Progress of optimization by showing the best to date result at each iteration
# plot_evaluations: Shows evolution of the search, for each hyperparameter we see the histogram of 
# explored values, for each pair of hyperparameters, the scatter plot of sampled values is plotted 
# with the evolution represented by color, from blue to yellow, e.g. for random search
# we see no evolution, it just randomly searches, however for forest we see it converges
# to certain points in the space that it explores more heavily
# plot_objective: we gain intuition into the score sensitivity with respect to hyperparameters, 
# we can decide which parts of the space may require a more fine-grained search and which 
# hyperparameters barely affect the score and can potentially be dropped from the search

#plot_convergence(minimisation_results)
#plot_evaluations(minimisation_results)
plot_objective(minimisation_results)
#plot_regret(minimisation_results)

# DBCV over iterations
#plt.plot(range(len(minimisation_results.func_vals)), minimisation_results.func_vals, marker="o")
#plt.xlabel("Iteration")
#plt.ylabel("DBCV Score")
#plt.title("DBCV Score Optimization Over Iterations")
#plt.show()


# #### Run the Clustering for Real using Best Parameters
# 
# This is where we actually run the clustering algorithm, DBCV score is recorded in a df so that results can be output into a graph. 
# 
# Dataframe in use is now called: df
# 
# We run for below scenarios, then we output clusters to show which one is best
# 1) Daily clustering
# 2) Global clustering
# 
# <img src="attachment:4ac34ce2-3b88-45a7-937c-db850dc67e65.png" width="60%" height="60%">
# 

# ##### Daily Clustering

# In[210]:


# Default values
n_neighbors = 15
min_dist = 0.1
min_cluster_size = 5
min_samples = None
n_components = 2

df_default, df_cluster_quality_default, all_global_centroids_default = run_cluster_process_with_global_cluster_id(n_neighbors,
                                                                                                                  min_dist,
                                                                                                                  min_cluster_size,
                                                                                                                  min_samples,
                                                                                                                  n_components,
                                                                                                                  df_data_test.copy()
                                                                                                                  #df_data.copy()
                                                                                                                 )
print('Default done!')


# Tuned values

# Tuned without human checks
# Best DBCV score:  0.3094117718640187
#n_neighbors = 35
#min_dist = 0.01
#min_cluster_size = 47
#min_samples = 16
#n_components = 64

# Tuned with human checks
n_neighbors = 12
min_dist = 0.1 # Was 0.038
min_cluster_size = 8
min_samples = 8
n_components = 75 # Changing from 60 to 70 reduces heatmap overlap, now we try 80

df_tuned, df_cluster_quality_tuned, all_global_centroids_tuned = run_cluster_process_with_global_cluster_id(n_neighbors,
                                                                                                            min_dist,
                                                                                                            min_cluster_size,
                                                                                                            min_samples,
                                                                                                            n_components,
                                                                                                            df_data_test.copy()
                                                                                                            #df_data.copy()
                                                                                                           )
print('Tuned done!')
print(f'{df_tuned["global_cluster_id"].unique()} clusters created.')


# In[171]:


# Averate tweets per cluster and day

df_tuned.groupby(['DateTimeStr','global_cluster_id'])['tweet_id'].aggregate('count')

avg_tweets = (
    df_tuned
    .groupby(['DateTimeStr','global_cluster_id'])['tweet_id']
    .count()
    .mean()
)

print(f"Average tweets per date+cluster: {avg_tweets:.2f}")


# In[283]:


# The below will only work when the summaries have been generated, this is used to plot a histogram of tweets in the cluster to the 
# summary, the intention is that this is similar to the same tweet comparison but to the medoid, this will prove that the tweets, medoid
# and summary have a similar distribution

# Copy df
df_events_all_group_new_event = df_events_all_group.copy()
df_events_all_group_spike_event = df_events_all_group.copy()
# New events
df_events_all_group_new_event = df_events_all_group_new_event[df_events_all_group_new_event['event_type'] == 'new_event']
df_events_all_group_new_event = df_events_all_group_new_event[['global_cluster_id','event_type','tweet_summary_bart-base-finetuned-gpt_1000_2_5']]
# Spike events
df_events_all_group_spike_event = df_events_all_group_spike_event[df_events_all_group_spike_event['event_type'] == 'event_spike']
df_events_all_group_spike_event = df_events_all_group_spike_event[['global_cluster_id','event_type','tweet_summary_bart-base-finetuned-gpt_1000_2_5']]

# New event embeddings
tweet_text_summary = df_events_all_group_new_event['tweet_summary_bart-base-finetuned-gpt_1000_2_5']
embedding_model_id = 'all-mpnet-base-v2'
embeddings_summary = embed_tweets(embedding_model_id,tweet_text_summary.tolist())
df_events_all_group_new_event['embeddings'] = list(embeddings_summary)

# Spike event embeddings
tweet_text_summary = df_events_all_group_spike_event['tweet_summary_bart-base-finetuned-gpt_1000_2_5']
embedding_model_id = 'all-mpnet-base-v2'
embeddings_summary = embed_tweets(embedding_model_id,tweet_text_summary.tolist())
df_events_all_group_spike_event['embeddings'] = list(embeddings_summary)


# In[307]:


cluster_colors


# In[327]:


# Function to compute the medoid of a set of vectors
def compute_medoid(vectors):
    sims = cosine_similarity(vectors)
    medoid_idx = np.argmax(np.sum(sims, axis=1))
    return vectors[medoid_idx]

# Make sure we have the colours selected to consistent across clusters
num_clusters = len(unique_cluster_ids)
color_norm = mcolors.Normalize(vmin=min(unique_cluster_ids), vmax=max(unique_cluster_ids))
scalar_map = cm.ScalarMappable(norm=color_norm, cmap='tab20')
cluster_colors = {cid: scalar_map.to_rgba(cid) for cid in unique_cluster_ids}

# Set up grid for 11 clusters (0–10)
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
axes = axes.flatten()
common_bins = np.linspace(0,1,20)

for cluster_id in range(15):
    df_cluster = df_tuned[df_tuned['global_cluster_id'] == cluster_id]
    df_text_summary_new_event_cluster = df_events_all_group_new_event[df_events_all_group_new_event['global_cluster_id'] == cluster_id]
    df_text_summary_spike_event_cluster = df_events_all_group_spike_event[df_events_all_group_spike_event['global_cluster_id'] == cluster_id]

    if df_cluster.empty:
        axes[cluster_id].set_visible(False)
        continue

    # Medoid similarities
    vectors = np.vstack(df_cluster['embeddings'].values)
    medoid = compute_medoid(vectors)
    similarities = cosine_similarity(vectors, [medoid]).flatten()
    # Best performing text summary similarities for new events
    text_summary_new_event_embedding = np.vstack(df_text_summary_new_event_cluster['embeddings'].values)
    similarities_2 = cosine_similarity(vectors, text_summary_new_event_embedding).flatten()
    # Best performing text summary similarities for spikes events
    if not df_text_summary_spike_event_cluster.empty:
        text_summary_spike_event_embedding = np.vstack(df_text_summary_spike_event_cluster['embeddings'].values)
        similarities_3 = cosine_similarity(vectors, text_summary_spike_event_embedding).flatten()

    # Mean and standard deviation of medoid
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    # Mean of text summaries
    mean_sim_2 = np.mean(similarities_2)
    # T-test
    #try:
        # Check arrays are same length
    #    if len(similarities) == len(similarities_2):
    #        t_stat, p_val = ttest_rel(similarities, similarities_2)
    #        ax.text(0.05, 0.9, f"p = {p_val:.3f}", transform=ax.transAxes, fontsize=8, color='black')
    #    else:
    #        ax.text(0.05, 0.9, "Mismatch in lengths", transform=ax.transAxes, fontsize=8, color='red')
    #except Exception as e:
    #    ax.text(0.05, 0.9, f"Error: {e}", transform=ax.transAxes, fontsize=8, color='red')

    mean_medoid = np.mean(similarities)
    mean_summary = np.mean(similarities_2)
    #mean_diff = mean_summary - mean_medoid
    #ax.text(0.05, 0.8, f"Δμ = {mean_diff:.4f}", transform=ax.transAxes, fontsize=8)
    
    ax = axes[cluster_id]
    color = cluster_colors.get(cluster_id, 'gray')
    # Plot histogram
    ax.hist(similarities, bins=common_bins, color=color, alpha=0.8, density=True, edgecolor='black')
    ax.axvline(mean_sim, color='black', linestyle='-', linewidth=1, label=f'Mean: {mean_sim:.2f}')
    #ax.axvline(mean_sim + std_sim, color='green', linestyle=':', linewidth=1, label=f'+1 SD: {mean_sim + std_sim:.2f}')
    #ax.axvline(mean_sim - std_sim, color='green', linestyle=':', linewidth=1, label=f'-1 SD: {mean_sim - std_sim:.2f}')
    #ax.axvline(mean_sim + (2 * std_sim), color='orange', linestyle=':', linewidth=1, label=f'+2 SD: {mean_sim + std_sim:.2f}')
    #ax.axvline(mean_sim - (2 * std_sim), color='orange', linestyle=':', linewidth=1, label=f'-2 SD: {mean_sim - std_sim:.2f}')
    ax.axvline(mean_sim_2, color='black', linestyle='--', linewidth=1, label=f'Mean: {mean_sim_2:.2f}')

    # Overlay KDE curve for medoid
    kde = gaussian_kde(similarities)
    x_vals = np.linspace(0, 1, 200)
    ax.plot(x_vals, kde(x_vals), color='black', linewidth=1)

    # Overlay KDE curve for text summary new event
    kde_2 = gaussian_kde(similarities_2)
    x_vals_2 = np.linspace(0, 1, 200)
    ax.plot(x_vals, kde_2(x_vals_2), color='black', linewidth=1, linestyle='dashed')

    # Overlay KDE curve for text summary spike event
    #if not df_text_summary_spike_event_cluster.empty:
    #    kde_3 = gaussian_kde(similarities_3)
    #    x_vals_3 = np.linspace(0, 1, 200)
    #    ax.plot(x_vals, kde_3(x_vals_3), color='darkred', linewidth=1, linestyle='dashed')

    # Add title and axis labels
    ax.set_title(f'Cluster {cluster_id}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 8)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')

# Remove unused subplot if less than 12 used
for i in range(15, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig("cluster_histogram.pdf", format='pdf')
plt.show()


# In[277]:


# Compare stats between default and tuned
df_cluster_quality_default[['DBCV']].describe()
df_cluster_quality_tuned[['DBCV']].describe()


# In[236]:


df_tuned.to_csv('test3.csv')


# In[311]:


# Daily clustering visualisations
# Choose whether to use the default or tuned df, then run the code

df_filtered = df_default.copy()
df_filtered = df_tuned.copy()
df_filtered = df_filtered[df_filtered['global_cluster_id'] != -1]

unique_cluster_ids = sorted(df_filtered['global_cluster_id'].unique())
num_clusters = len(unique_cluster_ids)
color_norm = mcolors.Normalize(vmin=min(unique_cluster_ids), vmax=max(unique_cluster_ids))
scalar_map = cm.ScalarMappable(norm=color_norm, cmap='tab20')
cluster_colors = {cid: scalar_map.to_rgba(cid) for cid in unique_cluster_ids}
unique_days = sorted(df_filtered['DateTimeStr'].unique())[:15] # Run for up to 15 days

fig, ax = plt.subplots(2, 6, figsize=(18, 6.5))
ax = ax.flatten()
for idx, day in enumerate(unique_days):
    df_filtered_day = df_filtered[df_filtered['DateTimeStr'] == day].copy()
    embeddings_filtered = np.vstack(df_filtered_day['embeddings'].values)
    clusters = df_filtered_day['global_cluster_id'].values
    
    #umap_subplot_vis(embeddings_filtered, clusters, cluster_colors, f'Day: {day}', ax[idx])
    tsne_subplot_vis(embeddings_filtered, clusters, cluster_colors, f'Day: {day}', ax[idx])

for i in range(len(unique_days), len(ax)):
    ax[i].set_visible(False)
    
fig.tight_layout()
#fig.savefig('cluster_umap_train_default_daily.pdf', format='pdf')
#fig.savefig('cluster_tsne_train_default_daily.pdf', format='pdf')
#fig.savefig('cluster_umap_train_tuned_daily.pdf', format='pdf')
#fig.savefig('cluster_tsne_train_tuned_daily.pdf', format='pdf')
fig.savefig('cluster_tsne_test_tuned_daily.pdf', format='pdf')
fig.show()


# In[80]:


# Daily clustering combined visualisations

df_filtered = df_default.copy()
#df_filtered = df_tuned.copy()
df_filtered = df_filtered[df_filtered['global_cluster_id'] != -1]

unique_cluster_ids = sorted(df_filtered['global_cluster_id'].unique())
num_clusters = len(unique_cluster_ids)
color_norm = mcolors.Normalize(vmin=min(unique_cluster_ids), vmax=max(unique_cluster_ids))
scalar_map = cm.ScalarMappable(norm=color_norm, cmap='tab20')
cluster_colors = {cid: scalar_map.to_rgba(cid) for cid in unique_cluster_ids}
unique_days = sorted(df_filtered['DateTimeStr'].unique())[:15] # Run for up to 15 clusters

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
embeddings_filtered = np.vstack(df_filtered['embeddings'].values)
clusters = df_filtered['global_cluster_id'].values
    
umap_subplot_vis(embeddings_filtered, clusters, cluster_colors, '', ax)
#tsne_subplot_vis(embeddings_filtered, clusters, cluster_colors, 'test', ax)
    
fig.tight_layout()
fig.savefig('cluster_umap_train_tuned_daily_all.pdf', format='pdf')
fig.show()


# ##### Global Clustering

# In[258]:


# Default values
n_neighbors = 15
min_dist = 0.1
min_cluster_size = 5
min_samples = None
n_components = 2

df_global_default = df_data_test.copy()
#df_global_default = df_data.copy()
embeddings_global = np.vstack(df_global_default['embeddings'].values)

umap_model_first_reduction, X_umap_first_reduction, hdbscan_clusterer, labels, dbcv_score = cluster_embeddings(embeddings_global,
                                                                               n_neighbors,
                                                                               min_dist,
                                                                               min_cluster_size,
                                                                               min_samples,
                                                                               n_components
                                                                              )

labels = np.array(labels)
df_global_default['global_cluster_id'] = labels

# Plot
# You can also pass embeddings instead of X_umap_first_reduction
#run_all_vis(X_umap_first_reduction, labels, 'global_test_default')

# Tuned values

# Large word embeddings and values tweaked for better clustering
# This gives stationary ADF and KPSS
n_neighbors = 16
min_dist = 0.1
min_cluster_size = 15
min_samples = 12
n_components = 40

# Optimised values but no human visualisation changes
# Best DBCV score:  0.6045924017534546
# Highest DBCV score but not good in t-SNE
n_neighbors = 31
min_dist = 0.0056411579027100265
min_cluster_size = 31
min_samples = 39
n_components = 68

# Tuned values and human checked for better performance
#Best DBCV score:  0.4725966407722493
n_neighbors = 12
min_dist = 0.006 # 0.02
min_cluster_size = 12
min_samples = 20 # 14
#max_samples = 40
n_components = 68 # 50

n_neighbors = 12
min_dist = 0.1 # Was 0.038
min_cluster_size = 8
min_samples = 8
n_components = 75 # Changing from 60 to 70 reduces heatmap overlap, now we try 80

df_global_tuned = df_data_test.copy()
#df_global_tuned = df_data.copy()
embeddings_global = np.vstack(df_global_tuned['embeddings'].values)
umap_model_first_reduction, X_umap_first_reduction, hdbscan_clusterer, labels, dbcv_score = cluster_embeddings(embeddings_global,
                                                                               n_neighbors,
                                                                               min_dist,
                                                                               min_cluster_size,
                                                                               min_samples,
                                                                               n_components
                                                                              )
labels = np.array(labels)
df_global_tuned['global_cluster_id'] = labels

# Plot
# You can also pass embeddings instead of X_umap_first_reduction
#run_all_vis(X_umap_first_reduction, labels, 'global_test_tuned')


# In[169]:


# Averate tweets per cluster and day

df_global_tuned.groupby(['DateTimeStr','global_cluster_id'])['tweet_id'].aggregate('count')

avg_tweets = (
    df_global_tuned
    .groupby(['DateTimeStr','global_cluster_id'])['tweet_id']
    .count()
    .mean()
)

print(f"Average tweets per date+cluster: {avg_tweets:.2f}")


# In[259]:


# Global clustering visualisations

df_filtered = df_global_default.copy()
df_filtered = df_global_tuned.copy()
df_filtered = df_filtered[df_filtered['global_cluster_id'] != -1]
df_filtered = df_filtered[df_filtered['DateTimeStr'] != '2017-09-26'] # Only 4 tweets so remove

unique_cluster_ids = sorted(df_filtered['global_cluster_id'].unique())
num_clusters = len(unique_cluster_ids)
color_norm = mcolors.Normalize(vmin=min(unique_cluster_ids), vmax=max(unique_cluster_ids))
scalar_map = cm.ScalarMappable(norm=color_norm, cmap='tab20')
cluster_colors = {cid: scalar_map.to_rgba(cid) for cid in unique_cluster_ids}
unique_days = sorted(df_filtered['DateTimeStr'].unique())[:15] # Run for up to 15 clusters

fig, ax = plt.subplots(2, 6, figsize=(18, 6.5))
ax = ax.flatten()
for idx, day in enumerate(unique_days):
    df_filtered_day = df_filtered[df_filtered['DateTimeStr'] == day].copy()
    embeddings_filtered = np.vstack(df_filtered_day['embeddings'].values)
    clusters = df_filtered_day['global_cluster_id'].values
    
    #umap_subplot_vis(embeddings_filtered, clusters, cluster_colors, f'Day: {day}', ax[idx])
    tsne_subplot_vis(embeddings_filtered, clusters, cluster_colors, f'Day: {day}', ax[idx])

for i in range(len(unique_days), len(ax)):
    ax[i].set_visible(False)
    
fig.tight_layout()
#fig.savefig('cluster_umap_train_default_global.pdf', format='pdf')
#fig.savefig('cluster_umap_train_tuned_global.pdf', format='pdf')
#fig.savefig('cluster_tsne_train_tuned_global.pdf', format='pdf')
fig.savefig('cluster_tsne_test_tuned_global.pdf', format='pdf')
fig.show()


# In[223]:


df_global_tuned.to_csv('df_global_tuned.csv', index=False)


# In[747]:


# Global clustering visualisations

df_filtered = df_global_default.copy()
df_filtered = df_global_tuned.copy()
df_filtered = df_filtered[df_filtered['global_cluster_id'] != -1]

unique_cluster_ids = sorted(df_filtered['global_cluster_id'].unique())
num_clusters = len(unique_cluster_ids)
color_norm = mcolors.Normalize(vmin=min(unique_cluster_ids), vmax=max(unique_cluster_ids))
scalar_map = cm.ScalarMappable(norm=color_norm, cmap='tab20')
cluster_colors = {cid: scalar_map.to_rgba(cid) for cid in unique_cluster_ids}
unique_days = sorted(df_filtered['DateTimeStr'].unique())[:15] # Run for up to 15 clusters

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
embeddings_filtered = np.vstack(df_filtered['embeddings'].values)
clusters = df_filtered['global_cluster_id'].values
    
umap_subplot_vis(embeddings_filtered, clusters, cluster_colors, '', ax)
#tsne_subplot_vis(embeddings_filtered, clusters, cluster_colors, 'test', ax)
    
fig.tight_layout()
fig.savefig('cluster_umap_train_tuned_global_all.pdf', format='pdf')
fig.show()

# Run for a grid of visualisations
#save_name = 'global_test_tuned'
#run_all_vis_grid(embeddings_filtered, clusters)


# In[278]:


counts_df = (
    df_default.groupby(['DateTimeDt', 'global_cluster_id'])
    .size()
    .reset_index(name='count')
)

avg_counts = (
    counts_df.groupby('global_cluster_id')['count']
    .mean()
    .reset_index(name='avg_count_per_day')
)

print(avg_counts)


# In[486]:


counts_df = (
    df_tuned.groupby(['DateTimeDt', 'global_cluster_id'])
    .size()
    .reset_index(name='count')
)

counts_df = counts_df[counts_df['global_cluster_id'] != -1]
counts_df['DateTimeDt'] = pd.to_datetime(counts_df['DateTimeDt'])
counts_df['DateTimeDt'] = counts_df['DateTimeDt'].dt.date

heatmap_df = counts_df.pivot(index='global_cluster_id', columns='DateTimeDt', values='count')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_df, cmap='BuPu', annot=True, fmt='.0f', linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Tweet Count'})

plt.title('Tweet Volume per Cluster per Date')
plt.xlabel('Date')
plt.ylabel('Cluster')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("tweet_count_heatmap.pdf", format='pdf')
plt.show()


# In[280]:


df_cluster_quality_default['parameters'] = 'default'
df_cluster_quality_tuned['parameters'] = 'fine-tuned'
df_cluster_quality_all = pd.concat([df_cluster_quality_default,df_cluster_quality_tuned],axis = 0)

df_cluster_quality_all.head()


# In[86]:


# Graph of cluster centroids

# Default parameters
n = all_global_centroids_default.shape[0]
cluster_list = list(range(1, n + 1))
cluster_list = np.array(cluster_list)

# Visualise default centroids
umap_vis(all_global_centroids_default, cluster_list, 'centroid_test_default.pdf')

# Tuned parameters
n = all_global_centroids_tuned.shape[0]
cluster_list = list(range(1, n + 1))
cluster_list = np.array(cluster_list)

# Visualise tuned centroids
umap_vis(all_global_centroids_tuned, cluster_list, 'centroid_test_tuned.pdf')


# In[281]:


df_filtered = df_tuned.copy()
df_filtered['cluster_type'] = df_filtered['global_cluster_id'].apply(lambda x: 'Noise' if x == -1 else 'Not Noise')
counts = df_filtered.groupby('cluster_type').size().reset_index(name='count')
print(counts)

df_filtered = df_global_tuned.copy()
df_filtered['cluster_type'] = df_filtered['global_cluster_id'].apply(lambda x: 'Noise' if x == -1 else 'Not Noise')
counts = df_filtered.groupby('cluster_type').size().reset_index(name='count')
print(counts)


# In[287]:


# Example: Assuming your DataFrame is called df and cluster column is 'cluster'
#df_filtered = df_default[df_default['global_cluster_id'] != -1].copy()
df_filtered = df_default.copy()
#df_filtered = df_global_tuned.copy()
df_filtered = df_tuned.copy()
df_filtered['cluster_type'] = df_filtered['global_cluster_id'].apply(lambda x: 'Noise' if x == -1 else 'Not Noise')

grouped = df_filtered.groupby(['DateTimeStr', 'cluster_type']).size().reset_index(name='count')
pivot_table = grouped.pivot(index='DateTimeStr', columns='cluster_type', values='count').fillna(0)
pivot_table = pivot_table.sort_index()

ordered_periods = ['Not Noise', 'Noise']
pivot_table = pivot_table[ordered_periods]

# Plot
pivot_table.plot(kind='bar', stacked=True, figsize=(9, 5), colormap='Blues', edgecolor='black', linewidth=1)

plt.title("Stacked Number of Clusters (excluding -1) per Date")
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[ ]:


#df_global_default.to_csv('output_orig_test_global_default.csv', index=False)
df_global_tuned.to_csv('output_orig_test_global_tuned.csv', index=False)

df_default.to_csv('output_orig_test_default.csv', index=False)
#df_tuned.to_csv('output_orig_test_tuned.csv', index=False)


# ##### Add in the Topics for each Day
# 
# - We group by day (DateTimeStr assumed string like '2023-05-10')
# - For each day, extract clean text and embeddings
# - Run BERTopic on that subset (using your precomputed embeddings)
# - Assign the topic IDs and keywords back to the original DataFrame
# - Store the daily model if you want to inspect or analyze topics later
# 
# We also output a Wordcloud for sample clusters, proving that our method is satisfactory. 

# ##### Cluster Similarity Checks
# 
# Check clustering to ensure they are not too similar. 
# 
# Results: 
# - Tweets are short text, these often result in lower scores even if semantically similar
# - Sensitive to exact word matches, therefore more likley to show low value
# - No word order or meaning, only overlap
# - Pre-processing will have improved accuracy

# In[481]:


# Run this to get the data for comparison

cluster_id_to_use = 'global_cluster_id'

df_matrix = df_default.copy()
df_matrix = df_tuned.copy()
df_matrix = df_matrix[df_matrix[cluster_id_to_use] != -1]
#df_matrix = df_matrix[df_matrix['DateTimeStr'] == '2017-09-20']

# Convert embeddings to a 2D numpy array
embedding_matrix = np.vstack(df_matrix['embeddings'].values)

# Compute cosine similarity
cosine_sim_matrix = cosine_similarity(embedding_matrix)

# Create a DataFrame with cluster IDs and embeddings
df_clusters_embeddings = df_matrix.groupby(cluster_id_to_use)['embeddings'].apply(
    lambda x: np.mean(np.vstack(x), axis=0)
).reset_index()

# Compute similarity between cluster-level embeddings
embedding_matrix = np.vstack(df_clusters_embeddings['embeddings'].values)
cosine_sim_matrix = cosine_similarity(embedding_matrix)

cluster_labels = df_clusters_embeddings[cluster_id_to_use].astype(str).tolist()
highlight_mask = cosine_sim_matrix > 0.9




# Create mask for upper triangle
mask = np.triu(np.ones_like(cosine_sim_matrix, dtype=bool))  # <-- changed to upper triangle mask

# Plot only lower triangle
plt.figure(figsize=(5, 4))
ax = sns.heatmap(
    cosine_sim_matrix,
    xticklabels=cluster_labels,
    yticklabels=cluster_labels,
    cmap='Blues',
    square=True,
    mask=mask,
    linewidths=0.5,
    #linecolor='gray',
    cbar=True
)
plt.title("Cosine Similarity Between Clusters")
plt.xlabel("Cluster")
plt.ylabel("Cluster")
plt.xticks(rotation=45)

# Add red border around lower-triangle cells with value > 0.9
for i in range(cosine_sim_matrix.shape[0]):
    for j in range(i):  # only lower triangle (j < i)
        if cosine_sim_matrix[i, j] > 0.90:
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', linewidth=1)
            )

plt.tight_layout()
plt.savefig("cluster_cosine_similarity_test_tuned.pdf", format='pdf')
plt.show()


# In[483]:


def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

df_matrix = df_default.copy()
df_matrix = df_tuned.copy()
df_matrix = df_matrix[df_matrix[cluster_id_to_use] != -1]
cluster_id_to_use = 'global_cluster_id'

# Combine all tweet texts in each cluster
df_cluster_texts = df_matrix.groupby(cluster_id_to_use)['tweet_text_clean'].apply(lambda texts: ' '.join(texts)).reset_index()

texts = df_cluster_texts['tweet_text_clean'].tolist()
n = len(texts)
jaccard_sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        jaccard_sim_matrix[i, j] = jaccard_similarity(texts[i], texts[j])

cluster_labels = df_cluster_texts[cluster_id_to_use].astype(str).tolist()
mask = np.triu(np.ones_like(jaccard_sim_matrix, dtype=bool))

plt.figure(figsize=(5, 4))
ax = sns.heatmap(
    jaccard_sim_matrix,
    mask=mask,
    cmap='Purples',
    xticklabels=cluster_labels,
    yticklabels=cluster_labels,
    square=True,
    linewidths=0.5,
    #linecolor='gray'
)
plt.title("Jaccard Similarity Between Clusters")
plt.xlabel("Cluster")
plt.ylabel("Cluster")
plt.xticks(rotation=45)

# Add red border around lower-triangle cells with value > 0.2
for i in range(jaccard_sim_matrix.shape[0]):
    for j in range(i):  # only lower triangle (j < i)
        if jaccard_sim_matrix[i, j] > 0.9:
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', linewidth=1)
            )

plt.tight_layout()
plt.savefig("cluster_jaccard_similarity_test_tuned.pdf", format='pdf')
plt.show()



# #### Topic Modelling
# 
# We use BERTopic to get the topics for two things: 
# 1) Topic for each cluster
# 2) Topic for each cluster and each day (useful for checking cluster coherance)

# In[260]:


# We have cluster labels, tweet text and embedding, need the topics for model that is run each day
df_topics = df_default.copy()
df_topics = df_tuned.copy()

# Prepare a dict to store daily topic models or topic info if needed
daily_topic_models = {}

for day, group in df_topics.groupby('DateTimeStr'):
    print(f"Processing day: {day} with {len(group)} tweets")

    if len(group) < 10:  # small threshold to skip tiny days
        continue

    # Extract text and embeddings for the day
    texts = group['tweet_text_clean'].tolist()
    embeddings_topic = list(group['embeddings'])
    # Convert embeddings list to NumPy array
    embeddings_array = np.vstack(embeddings_topic)

    # Use BERTopic to get topics
    # Add to the stopwords
    stopwords_additions = ['http','https','amp','com','rt','hurricane','peurto','rico','maria','puerto','hurricanemaria','need', ''] # Remove disaster words that do not add anything
    stopwords_custom = list(stopwords.words('english')) + stopwords_additions
    vectorizer_model = CountVectorizer(ngram_range=(1, 2)
                                       ,stop_words=stopwords_custom
                                       )
    ctfidf_model = ClassTfidfTransformer(
        # seed_words=domain_specific_terms,
        seed_multiplier=2
        #reduce_frequent_words=True
    )
    topic_model = BERTopic(embedding_model=None,
                           vectorizer_model=vectorizer_model,
                           ctfidf_model=ctfidf_model,
                           calculate_probabilities=False,
                           verbose=False)
    
    # Fit BERTopic on the text + embeddings
    topics, probs = topic_model.fit_transform(texts, embeddings_array)

    # Store the model for later if needed
    daily_topic_models[day] = topic_model

    # Add topics to the df_topics
    df_topics.loc[group.index, 'daily_topic'] = topics

    # Optionally, add topic keywords for each tweet
    topic_keywords_map = {}
    for t in set(topics):
        if t == -1:
            topic_keywords_map[t] = "Noise"
        else:
            words_scores = topic_model.get_topic(t)
            keywords = ", ".join([word for word, _ in words_scores[:5]])
            topic_keywords_map[t] = keywords

    df_topics.loc[group.index, 'daily_topic_keywords'] = df_topics.loc[group.index, 'daily_topic'].map(topic_keywords_map)


# In[245]:


df_tuned_bar = df_tuned[df_tuned["global_cluster_id"] != -1].copy()

# Make sure we have the colours selected to consistent across clusters
cluster_colors = {cid: scalar_map.to_rgba(cid) for cid in unique_cluster_ids}

top_words = ctfiidf_top_words(
    df_tuned_bar,
    group_col="global_cluster_id",
    text_col="tweet_text_clean",
    top_n=5,
    ngram_range=(1,2),   # Unigrams and bigrams
    min_df=2,            # Ignore rare terms across clusters
)

plot_cluster_bars(top_words, title_prefix="Cluster", cols=5, figsize=(12, 6))


# In[302]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Filter the DataFrame for cluster = 0
cluster_0_topics = df_topics[df_topics['global_cluster_id'] == 4]['daily_topic_keywords'].dropna().tolist()

# Combine all the text into one string
text = ' '.join(cluster_0_topics)

# Create the WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

# Plot it
plt.figure(figsize=(6, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Centroid Check Method\nCluster 0", fontsize=16)
plt.show()


# #### Data Cleanse
# 
# We now have a cluster for each event, which is unique for each new event, i.e. the cluster value is not reset each day. Now we clean the data and then use the clusters to identify events of interest. 
# 
# We rank tweets closet to the medoid, however we want some diversity as this will avoid narrow and redundant summaries that do not capture the full range of sub-topics in the cluster. This aligns with the task of best representing events in the data. Clusters contain internal variation, sub-topics, views and emotional differences. We do not want repeating tweets for summarisation, we want maximum coverage. 
# 
# Ideas:
# - Medoid + MMR
# - Agglomerative or k-means within a cluster
# 
# Dataframe is now called **df_clean**. 

# In[268]:


df_clean = df_topics[df_topics["global_cluster_id"] != -1].copy()
df_clean['cosine_distance_to_medoid'] = df_clean['cosine_distance_to_medoid'].fillna(1.0)
df_clean = df_clean.drop(columns=['class_label', 'cluster', 'daily_topic', 'DateTimeUnix', 'DateTime', 'DateTimeHour'])
#df_clean['embedding'] = df_clean['embedding'].apply(np.array)

# Set the rank to order by the best tweet
df_clean['rank'] = df_clean.groupby(['DateTimeDt','global_cluster_id'])['cosine_distance_to_medoid'].rank(method='dense', ascending=True)
df_clean = df_clean.sort_values(by=['DateTimeDt', 'global_cluster_id', 'rank'], ascending=[True, False, True])

# MMR variables
lambda_param = 0.5
top_k = 5

# MMR
mmr_selected_rows = []
for (dt, cluster_id), group in df_clean.groupby(['DateTimeDt', 'global_cluster_id']):
    if len(group) < 2:
        mmr_selected_rows.append(group)  # If MMR too small
        continue

    embeddings = np.stack(group['embeddings'].values)
    medoid_idx = group['cosine_distance_to_medoid'].idxmin()
    medoid_vector = df_clean.loc[medoid_idx, 'embeddings']

    # Select top_k using MMR
    selected_idxs = mmr(embeddings, medoid_vector, lambda_param=lambda_param, top_k=min(top_k, len(group)))

    # Map local MMR indices to global DataFrame indices
    selected_rows = group.iloc[selected_idxs]
    selected_rows = selected_rows.copy()
    selected_rows['rank_mmr'] = range(1, len(selected_rows) + 1)

    mmr_selected_rows.append(selected_rows)

# Combine MMR-selected tweets into final DataFrame
df_clean_mmr = pd.concat(mmr_selected_rows, ignore_index=True)

# View final MMR-ranked tweets
df_clean_mmr = df_clean_mmr.sort_values(by=['DateTimeDt', 'global_cluster_id', 'rank_mmr'])
df_clean_mmr.head(20)


# In[265]:


#df_clean.to_csv('output_orig_test.csv', index=False)
#df_clean = pd.read_csv('output_orig_test.csv')


# In[266]:


#df_clean.groupby(['DateTimeDt','global_cluster_id', 'cluster']).size().reset_index(name='count')
df_clean.groupby(['DateTimeDt','global_cluster_id']).size().reset_index(name='count')
#df_clean.groupby(['global_cluster_id','DateTimeDt']).size().reset_index(name='count')


# ##### Create a similarity dataframe to identify Drift Spikes

# In[269]:


# Medoid df

# Ensure datetime is parsed
df_clean['DateTimeDt'] = pd.to_datetime(df_clean['DateTimeDt'])
df_clean = df_clean.sort_values(['global_cluster_id', 'DateTimeDt'])

# Medoid per date and cluster
def compute_medoid(emb_list):
    X = np.vstack(emb_list)
    distance_matrix = pairwise_distances(X, metric='cosine')
    medoid_idx = distance_matrix.sum(axis=1).argmin()
    return X[medoid_idx]

ROLLING_PERIOD = 3

df_medoids = (
    df_clean.groupby(['global_cluster_id', 'DateTimeDt'])['embeddings']
    .apply(compute_medoid)
    .reset_index()
    .rename(columns={'embeddings': 'medoid'})
)

# Medoid cosine similarity to previous day
df_medoids['sim_to_prev_day'] = df_medoids.groupby('global_cluster_id')['medoid'].transform(
    lambda x: [np.nan] + [
        cosine_similarity([x.iloc[i]], [x.iloc[i - 1]])[0][0]
        for i in range(1, len(x))
    ]
)

# Rolling 2-day average medoid
def rolling_mean_similarity(x):
    result = [np.nan]
    for i in range(1, len(x)):
        emb_stack = np.vstack([x.iloc[i], x.iloc[i - 1]])
        avg_emb = emb_stack.mean(axis=0)
        sim = cosine_similarity([x.iloc[i]], [avg_emb])[0][0]
        result.append(sim)
    return result

df_medoids['sim_to_rolling_mean'] = df_medoids.groupby('global_cluster_id')['medoid'].transform(rolling_mean_similarity)

# Rolling mean
df_medoids['rolling_mean'] = (
    df_medoids.groupby('global_cluster_id')['sim_to_rolling_mean']
    .transform(lambda x: x.rolling(window=ROLLING_PERIOD, min_periods=ROLLING_PERIOD).mean())
)

# Rolling standard deviation
df_medoids['rolling_sd'] = (
    df_medoids.groupby('global_cluster_id')['sim_to_rolling_mean']
    .transform(lambda x: x.rolling(window=ROLLING_PERIOD, min_periods=ROLLING_PERIOD).std())
)

# Flag event spikes
df_medoids['exceeds_1_sd'] = (
    (df_medoids['sim_to_rolling_mean'] > df_medoids['rolling_mean'] + df_medoids['rolling_sd']) |
    (df_medoids['sim_to_rolling_mean'] < df_medoids['rolling_mean'] - df_medoids['rolling_sd'])
)

# Done
df_drift_spike_all = df_medoids.drop(columns='medoid')

df_drift_spike_all


# In[270]:


# Set your cluster of interest
cluster_id = 0

# Filter DataFrame
sim_df = df_drift_spike_all[df_drift_spike_all['global_cluster_id'] == cluster_id].copy()
if sim_df.empty:
    raise ValueError(f"No data found for cluster {cluster_id}")

# Ensure datetime index
sim_df['DateTimeDt'] = pd.to_datetime(sim_df['DateTimeDt'])
sim_df.sort_values('DateTimeDt', inplace=True)
sim_df.set_index('DateTimeDt', inplace=True)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(sim_df.index, sim_df['sim_to_rolling_mean'], marker='o', label='Similarity to Rolling Mean')
plt.plot(sim_df.index, sim_df['rolling_mean'], linestyle='--', color='gray', label='Rolling Mean')
plt.fill_between(sim_df.index,
                 sim_df['rolling_mean'] - sim_df['rolling_sd'],
                 sim_df['rolling_mean'] + sim_df['rolling_sd'],
                 color='orange', alpha=0.2, label='±1 Std Dev')

# Highlight spikes
spikes = sim_df['exceeds_1_sd']
plt.scatter(sim_df.index[spikes], sim_df['sim_to_rolling_mean'][spikes], color='red', label='Drift Spike', zorder=5)

plt.title(f"Topic Spike – Cluster {cluster_id}")
plt.xlabel("Date")
plt.ylabel("Cosine Similarity to Rolling Mean")
plt.xticks(rotation=45)
plt.ylim([0.82,0.98])
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.savefig('semantic-drift-cluster-1.pdf', format='pdf')
plt.show()


# #### Get Events to Summariase
# 
# We will identify two types of events: 
# 1) Emerging events, this is the earliest date for each cluster, the first event of a new topic
# 2) Event drift, this is events that are more than one standard deviation different from previous days event, this is where the topic has changed enough to be of interest, and therefore a new event
# 
# Dataframes we use: 
# - **df_clean_merged_min** - new events, this is the minimum date in a cluster
# - **df_clean_merged_spike** - spike events, this is where the topic has spiked
# - **df_events_all** - contains both event types above

# ##### Earliest Event per Cluster

# In[790]:


df_clean_merged_min_test = df_clean_merged_min[df_clean_merged_min['global_cluster_id'] == 0].copy()
df_clean_merged_min_test



# In[271]:


n = 5  # number of tweets to select per cluster

# Get the minimum date per cluster
df_min_date_cluster = df_clean_mmr.groupby('global_cluster_id')['DateTimeDt'].min().reset_index()
df_min_date_cluster.rename(columns={'DateTimeDt': 'min_dt'}, inplace=True)

# Merge to keep only rows on the min date
df_clean_merged = pd.merge(df_clean_mmr, df_min_date_cluster, on='global_cluster_id')
df_clean_merged_min = df_clean_merged[df_clean_merged['DateTimeDt'] == df_clean_merged['min_dt']].copy()
df_clean_merged_min = df_clean_merged_min.sort_values(['DateTimeDt', 'global_cluster_id', 'rank_mmr'], ascending=[True, False, True])

# n tweets per cluster, this should already be reduced anyhow
df_clean_merged_min = df_clean_merged_min.groupby(['global_cluster_id', 'DateTimeDt']).head(n)

# Clean data
df_clean_merged_min = df_clean_merged_min.drop(columns=['embeddings', 'min_dt'])
df_clean_merged_min['event_type'] = 'new_event'

df_clean_merged_min = df_clean_merged_min.sort_values(by=['DateTimeDt', 'global_cluster_id', 'rank_mmr'], ascending=[True, False, True])

df_clean_merged_min.head(10)


# ##### Drift Spike Events per Cluster

# In[358]:


df_drift_spikes = df_drift_spike_all[df_drift_spike_all['exceeds_1_sd'] == True]
df_drift_spikes


# In[273]:


# Only where a drift spike
df_drift_spikes = df_drift_spike_all[df_drift_spike_all['exceeds_1_sd'] == True]
# Change to date
df_drift_spikes['DateTimeDt'] = pd.to_datetime(df_drift_spikes['DateTimeDt']).dt.date
df_clean_mmr['DateTimeDt'] = pd.to_datetime(df_clean_mmr['DateTimeDt']).dt.date

df_clean_merged_spike = pd.merge(df_clean_mmr, df_drift_spikes, left_on=['global_cluster_id', 'DateTimeDt'],
                               right_on=['global_cluster_id', 'DateTimeDt'],
                               how='inner')
df_clean_merged_spike = df_clean_merged_spike.sort_values(by=['DateTimeDt', 'global_cluster_id', 'rank_mmr'], ascending=[True, False, True])

# Tweets per cluster, should already be only 3
df_clean_merged_spike = df_clean_merged_spike.groupby(['DateTimeDt', 'global_cluster_id']).head(n)

# Clean data
df_clean_merged_spike = df_clean_merged_spike.drop(columns=['embeddings'])
df_clean_merged_spike['event_type'] = 'event_spike'

df_clean_merged_spike = df_clean_merged_spike.sort_values(by=['DateTimeDt', 'global_cluster_id', 'rank_mmr'], ascending=[True, False, True])

df_clean_merged_spike.head(10)


# ##### Tables Review
# 
# We now have 2 tables, a new event table and a event spike table, need to check there is no overlap, should not be but just in case. 

# In[274]:


set_min = set(df_clean_merged_min[['DateTimeDt', 'global_cluster_id']].itertuples(index=False, name=None))
set_spike = set(df_clean_merged_spike[['DateTimeDt', 'global_cluster_id']].itertuples(index=False, name=None))

# Find intersection
common_pairs = set_min & set_spike

common_pairs


# In[275]:


# Combine event data
df_events_all = pd.concat([df_clean_merged_min, df_clean_merged_spike], ignore_index=True)
df_events_all.head()


# In[459]:


# Rank versus rank MMR analysis
# Here we compare performance to show whether MMR is better at reducing redundant tweets and improving diversity

cluster_id_analysis = 6
date_analysis = '2017-09-25'

# Using Rank
df_clean_rank_check = df_clean.copy()
df_clean_rank_check = df_clean_rank_check[['DateTimeStr','global_cluster_id','tweet_text_clean','cosine_distance_to_medoid','rank']]
df_clean_rank_check = df_clean_rank_check[df_clean_rank_check['global_cluster_id'] == cluster_id_analysis]
df_clean_rank_check = df_clean_rank_check[df_clean_rank_check['DateTimeStr'] == date_analysis]
df_clean_rank_check = df_clean_rank_check[df_clean_rank_check['rank'] <= 10]

# Using Rank MMR
df_clean_mmr_rank_check = df_clean_mmr2.copy()
df_clean_mmr_rank_check = df_clean_mmr_rank_check[['DateTimeStr','global_cluster_id','tweet_text_clean','cosine_distance_to_medoid','rank','rank_mmr']]
df_clean_mmr_rank_check = df_clean_mmr_rank_check[df_clean_mmr_rank_check['global_cluster_id'] == cluster_id_analysis]
df_clean_mmr_rank_check = df_clean_mmr_rank_check[df_clean_mmr_rank_check['DateTimeStr'] == date_analysis]
df_clean_mmr_rank_check = df_clean_mmr_rank_check[df_clean_mmr_rank_check['rank_mmr'] <= 10]

df_clean_rank_check.to_csv('df_clean_rank_check.csv', index=False)
df_clean_mmr_rank_check.to_csv('df_clean_mmr_rank_check.csv', index=False)

df_clean_rank_check.head(10)
df_clean_mmr_rank_check.head(10)


# In[461]:


df_clean_mmr_rank_check.head(10)


# In[362]:


counts = df_events_all.groupby(['DateTimeStr', 'global_cluster_id', 'event_type']).size().reset_index(name='count')

counts.sort_values(by=['DateTimeStr', 'global_cluster_id', 'event_type'],ascending=[False, True, True])
#counts = counts.sort_values(by=['global_cluster_id', 'DateTimeStr', 'event_type'])

print(counts)


# In[276]:


df_events_all_clean = df_events_all.drop(columns=['DateTimeDayWeek',
                                                  'cosine_distance_to_medoid',
                                                  'daily_topic_keywords',
                                                  'rolling_mean',
                                                  'rolling_sd',
                                                  'exceeds_1_sd',
                                                  'sim_to_prev_day',
                                                  'sim_to_rolling_mean'
                                                 ]).copy()

df_events_all_clean['DateTimeDt'] = pd.to_datetime(df_events_all_clean['DateTimeDt']).dt.date

df_events_all_clean.head()


# In[364]:


# Backup to CSV
df_events_all_clean.to_csv('df_events_all_clean_2025-08-01 v2_.csv', index=False)


# #### Summarise the Tweets
# 
# In this section we summarise the selected tweets, we use different LLMs and different methods of prompting. 
# 
# Methodology: 
# 1) Group by date and cluster, concatenate tweet_text_clean and separate by a fullstop
# 
# LLMs
# - Google T5 Base
# - Google FLAN-T5
# - BART
# - PEGASUS
# 
# Methods
# - Base (no changes)
# - Trained using SFT
# - RLHF

# ##### Group and Aggregate Tweets
# 
# We group and aggregate the tweets so there are 3 tweets concatenated by a full stop, these will then be summarised. 

# In[279]:


# Tweets are ranked and filtered on the highest probability that the tweet belongs to a cluster for each date. We reduce to three tweets, meaning that
# each day will contain three tweets that are most closely associated to that cluster. These three tweets will then be summarised. 

# Columns to group by
cols_group = ['DateTimeDt', 'global_cluster_id', 'event_type']
# Columns to order by, need to add in others, such as text coherance, closest to cluster centre, etc
cols_order = ['DateTimeDt', 'global_cluster_id', 'rank_mmr']

df_events_all_clean = df_events_all_clean.sort_values(by = cols_order, ascending = [True, True, True])

# Rank the results by Date, Cluster and Similarity, the top N will then be taken for summarisation

# Normal order
df_forward = (
    df_events_all_clean.groupby(cols_group, as_index=False)
    .agg({'tweet_text_clean': lambda x: '. '.join(x)})
    .rename(columns={'tweet_text_clean': 'tweet_text_clean'})
)

# Reverse order
df_reverse = (
    df_events_all_clean.groupby(cols_group, as_index=False)
    .agg({'tweet_text_clean': lambda x: '. '.join(x[::-1])})
    .rename(columns={'tweet_text_clean': 'tweet_text_clean_reverse'})
)

# Merge both
df_events_all_group = pd.merge(df_forward, df_reverse, on=cols_group)

df_events_all_group.head()


# #### Summarise the Events
# 
# First do human evaluations, then pre-trained models, then fine-tuned base models. 

# In[280]:


# Check number of tokens to ensure we don't go over the limit for our models (512 tokens)

df_events_all_group_tokens = df_events_all_group.copy()

# Choose a model you're likely to use for summarisation
tokenizer_tokens = AutoTokenizer.from_pretrained("braindao/flan-t5-cnn")  # or "facebook/bart-base"

# Apply tokenizer to each tweet and count tokens
df_events_all_group_tokens['token_count'] = df_events_all_group_tokens['tweet_text_clean'].apply(
    lambda x: len(tokenizer_tokens.encode(x, truncation=True))
)

print(df_events_all_group_tokens['token_count'])


# In[387]:


# Use GPT3.5 Turbo to get golden standard text summaries

# Setup client
client = openai.OpenAI(api_key="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")

df_events_all_group['tweet_summary_gpt_human'] = df_events_all_group['tweet_text_clean'].apply(
    lambda x: open_ai_summarise_text(x)
)


# In[379]:


# Summarise each of the events

#summary_text = summarise_text("Your text here", model, tokenizer, model_type="t5")

# Backup table
#df_events_all_group_backup = df_events_all_group.copy()

# Model datasets
# Base
# CNN/Daily Mail
# SAMSUM
# XSum

# T5 Efficient Tiny, paper: arXiv:2109.10686
df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_t5-efficient-tiny',
                                         model_id = 'google/t5-efficient-tiny',
                                         model_family = 't5')

#T5 small
df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_t5-small',
                                         model_id = 'google-t5/t5-small',
                                         model_family = 't5')

# T5
df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_t5-base',
                                         model_id = 'google-t5/t5-base',
                                         model_family = 't5')

#df_events_all_group = summarise_each_llm(df_events_all_group,
#                                         text_input = 'tweet_text_clean',
#                                         text_output = 'tweet_summary_t5-base-cnn',
#                                         model_id = '',
#                                         model_family = 't5')

df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_t5-base-samsum',
                                         model_id = 'amagzari/t5-base-finetuned-samsum-v2',
                                         model_family = 't5')

df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_t5-base-xsum',
                                         model_id = 'PavanNeerudu/t5-base-finetuned-xsum',
                                         model_family = 't5')

# FLAN-T5
df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_flan-t5-base',
                                         model_id = 'google/flan-t5-base',
                                         model_family = 't5')

df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_flan-t5-base-cnn',
                                         model_id = 'braindao/flan-t5-cnn',
                                         model_family = 't5')

df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_flan-t5-base-samsum',
                                         model_id = 'philschmid/flan-t5-base-samsum',
                                         model_family = 't5')

df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_flan-t5-base-xsum',
                                         model_id = 'brutusxu/flan-t5-base-finetuned-xsum',
                                         model_family = 't5')

# Pegasus
df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_pegasus-base-cnn',
                                         model_id = 'google/pegasus-cnn_dailymail',
                                         model_family = 'pegasus')

df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_pegasus-base-cnn-samsum',
                                         model_id = 'Feluda/pegasus-samsum',
                                         model_family = 'pegasus')

df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_pegasus-base-xsum',
                                         model_id = 'google/pegasus-xsum',
                                         model_family = 'pegasus')

# BART base
df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_bart-base',
                                         model_id = 'facebook/bart-base',
                                         model_family = 'bart')

df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_bart-base-cnn',
                                         model_id = 'ainize/bart-base-cnn',
                                         model_family = 'bart')

df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_bart-base-saumsum',
                                         model_id = 'philschmid/bart-base-samsum',
                                         model_family = 'bart')

df_events_all_group = summarise_each_llm(df_events_all_group,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_bart-base-xsum',
                                         model_id = 'Prikshit7766/bart-base-xsum',
                                         model_family = 'bart')


# In[281]:


df_events_all_group.head(2)


# In[282]:


#df_events_all_group.to_csv('output_human_ranking_with_tuned_models_v3.csv', index=False)
df_events_all_group = pd.read_csv('output_human_ranking_with_tuned_models_v3.csv')
df_events_all_group.head()


# ##### Process for human rankings
# 
# 1) Output into CSV
# 2) Human to rank
# 3) Import from CSV

# In[ ]:


#summary_cols = ['tweet_summary_flan-t5-base', 'tweet_summary_flan-t5-base-cnn']

exclude_cols = ['DateTimeDt', 'global_cluster_id', 'event_type', 'tweet_text_clean', 'tweet_text_clean', 'tweet_text_clean_reverse']
summary_cols = [col for col in df_events_all_group.columns if col not in exclude_cols]

df_events_all_group_pivot = pd.melt(
    df_events_all_group,
    id_vars=['DateTimeDt', 'global_cluster_id', 'event_type', 'tweet_text_clean'], 
    value_vars=summary_cols,
    var_name='summary_model', 
    value_name='summary_text'
)
df_events_all_group_pivot = df_events_all_group_pivot.sort_values(by = ['DateTimeDt', 'global_cluster_id', 'event_type'],
                                                                  ascending = [True, True, True])
df_events_all_group_pivot['fluency'] = ''
df_events_all_group_pivot['faithfulness'] = ''
df_events_all_group_pivot['abstractiveness'] = ''

summary_dir = os.path.join(current_dir, r'Code\summaries')
file_address_human_rankings = os.path.join(summary_dir, r'output_human_ranking.csv')
df_events_all_group_pivot.to_csv(file_address_human_rankings, index=False)

df_events_all_group_pivot.head(10)


# In[63]:


# Import from CSV
ranking_dir = os.path.join(current_dir, r'Blu Shared Drive')
file_address_human_rankings = os.path.join(ranking_dir, r'Text Summary Ranking Final.csv')
df_human_rankings = pd.read_csv(file_address_human_rankings)
#df_human_rankings.dropna(how='any')
df_human_rankings = df_human_rankings.dropna(subset=['fluency','faithfulness','abstractiveness'])

df_human_rankings.loc[df_human_rankings['fluency'] == 'Very good', 'fluency'] = 5
df_human_rankings.loc[df_human_rankings['fluency'] == 'Good', 'fluency'] = 4
df_human_rankings.loc[df_human_rankings['fluency'] == 'Medium', 'fluency'] = 3
df_human_rankings.loc[df_human_rankings['fluency'] == 'Bad', 'fluency'] = 2
df_human_rankings.loc[df_human_rankings['fluency'] == 'Very bad', 'fluency'] = 1

df_human_rankings.loc[df_human_rankings['faithfulness'] == 'Very good', 'faithfulness'] = 5
df_human_rankings.loc[df_human_rankings['faithfulness'] == 'Good', 'faithfulness'] = 4
df_human_rankings.loc[df_human_rankings['faithfulness'] == 'Medium', 'faithfulness'] = 3
df_human_rankings.loc[df_human_rankings['faithfulness'] == 'Bad', 'faithfulness'] = 2
df_human_rankings.loc[df_human_rankings['faithfulness'] == 'Very bad', 'faithfulness'] = 1

df_human_rankings.loc[df_human_rankings['abstractiveness'] == 'Very good', 'abstractiveness'] = 5
df_human_rankings.loc[df_human_rankings['abstractiveness'] == 'Good', 'abstractiveness'] = 4
df_human_rankings.loc[df_human_rankings['abstractiveness'] == 'Medium', 'abstractiveness'] = 3
df_human_rankings.loc[df_human_rankings['abstractiveness'] == 'Bad', 'abstractiveness'] = 2
df_human_rankings.loc[df_human_rankings['abstractiveness'] == 'Very bad', 'abstractiveness'] = 1

df_human_rankings.head(2)


# In[64]:


df_summary_scores = df_human_rankings.groupby('summary_model')[['fluency', 'faithfulness', 'abstractiveness']].mean().reset_index()
df_summary_scores['mean_score'] = df_summary_scores[['fluency', 'faithfulness', 'abstractiveness']].mean(axis=1)
df_summary_scores


# In[711]:


df_scores


# ##### Multiple outputs using Search and Sampling strategies
# 
# We discover that outputs are not diverse, even with maximum diversity configurations, we re-order tweets instead, this is done in the RLHF section. 

# In[923]:


model_id = 'braindao/flan-t5-cnn'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, return_dict=True)

def summarise_text_multiple(text):
    inputs = tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
    outputs = model.generate(
        inputs['input_ids'],
        length_penalty=1.0,
        max_length=500,
        min_length=30,
        no_repeat_ngram_size=3,
        early_stopping=True,
        # The below encourages diverse outputs
        #num_return_sequences=3,
        # Choose either of the below for multiple outputs: 
        # 1) do_sample=True + top_p / temperature (stochastic sampling)
        #do_sample=True,
        #top_p=0.9,
        #top_k=2,
        #temperature=0.9,
        # 2) num_beams + num_return_sequences + diversity_penalty (diverse beam search)
        #num_beams=6,
        #num_beam_groups=3,
        #diversity_penalty=1.0,
    )
    summary = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return summary

summary_output = summarise_text_multiple("Check out pictures from NWS San Juan Office. Hurricane Maria eyewall shredded and destroyed their NWS Radar will have to be replaced.. Hurricane Maria seriously damaged the Doppler Radar in San Juan, PR. Itâ€™s down indefinitely. Picture Courtesy: NWS San Juan. Pictures of the damage to the Doppler radar in Puerto Rico after Hurricane Maria. Donate at  PuertoRico HurricaneMaria. Hereâ€™s Where You Can Donate To Relief Efforts In Puerto Rico  DonateLife donate HurricaneMaria help. Dear NOLA friends, If you want to help support those affected by Hurricane Maria in Puerto Rico")

for i, summary in enumerate(summary_output):
    print(f"Summary {i}: {summary}")


# In[ ]:


#df_events_all_group[[DateTimeDt	global_cluster_id	event_type	tweet_text_clean]]

#output_summaries(['tweet_summary_bart-large-xsum'],[1])
output_summaries(None,None)


# ### Fine Tune Model
# 
# ### Summarise Training Tweets using ChatGPT
# 
# ChatGPT is used to summarise the data, this will then be used to train the model for better abstract summarisation. We want as much diversification as possible because this will teach the model a more diverse range of text to summarise, it will generalise better to new text, and reduce potential bias. This is done by using *stratified sampling*, this divides the data according to important features. Each group is then sampled proportionally or equally. 
# 
# The stratum used will be: 
# 
# - Disaster type, identified using the file
# - Topic, identified using multi-stage clustering
# - Readability and quality, using index and bins
# - Variance, using cosine and bins
# 
# Pipeline, similar to the one used above: 
# 
# 1) Preprocess the tweets
# 2) Embed tweets using Sentence Transformer
# 3) Cluster
# 4) Measure the readability and quality of the tweets
# 5) Measure the diversity in tweet text
# 6) Selection of tweets:
# 7) 1) Disaster type
#    2) At least 3 words
#    3) Between 2 and 6 tweets each
#    4) Maximum variance (cosine_distances(tweet_embeddings), Greedy Max-Min Selection)
#    5) Aim for a mixture of tweets within: same day, 2, 3, 4, 5, 7, 10, 14, 31, any days
# 

# In[ ]:


# Set the root directory you want to start from
# Use the 47k dataset as this is widely available from website, the 29k dataset requires a form to be filled out to access
# https://crisisnlp.qcri.org/humaid_dataset
root_dirs = [
    r'C:\Users\mail\OneDrive\University\04 Bristol\07 Project\Code\Data\HumAID_data_events_set1_47K',
    r'C:\Users\mail\OneDrive\University\04 Bristol\07 Project\Code\Data\HumAID_data_events_set2_29K'
]

# Create an empty list to store the DataFrames
dfs = []

# Walk through all folders and subfolders
for root_dir in root_dirs:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.tsv') and ('_dev' in file or '_train' in file or '_test' in file):
                file_path = os.path.join(dirpath, file)
                print(f"Loading {file_path}...")
                df_file = pd.read_csv(file_path, sep='\t')
                df_file['tweet_file'] = file
                dfs.append(df_file)

# Combine all dfs
df_training = pd.concat(dfs, ignore_index=True)

# Add Tweet DateTime in Unix epoch time format
#df_training['DateTimeUnix'] = df_training.apply(lambda row: ta.find_tweet_timestamp(row['tweet_id']), axis=1)

# Add Tweet DateTime in string format
df_training['tweet_date'] = df_training.apply(lambda row: timestamp_to_str(ta.find_tweet_timestamp(row['tweet_id'])), axis=1) # DateTimeStr
# Pre-process text
df_training['tweet_text_clean'] = df_training.apply(lambda row: custom_preprocessing(row['tweet_text']), axis=1)
# Clean up df
df_training = df_training[['tweet_id','tweet_text','tweet_text_clean','tweet_date','tweet_file']]

print("All _train CSV files loaded and combined!")


# In[754]:


# Add and update column names of the disaster and file name

# Label disaster type
df_training['disaster'] = df_training['tweet_file']
df_training['disaster'] = df_training['disaster'].str.replace(r'(_dev|_train|_test)\.tsv$', '', regex=True)

# Label dev/train/test
df_training['data_split'] = df_training['tweet_file']
df_training['data_split'] = df_training['data_split'].str.extract(r'_(train|dev|test)\.tsv$', expand=False)

df_training.head()


# In[824]:


df_training.shape


# In[807]:


df_training.groupby(['disaster']).size().reset_index(name='tweet_count')


# In[514]:


# Test pre-processing
def a_test(tweet_id, df):
    df_data_filtered = df[df['tweet_id'] == tweet_id]
    
    for tweet_date, text_original, text_clean in zip(
        df_data_filtered["tweet_date"],
        df_data_filtered["tweet_text"],
        df_data_filtered["tweet_text_clean"]):

        print("Tweet Date: ", tweet_date)
        print("-" * 10)  # separator line
        print("Original Tweet: ", text_original)
        print("-" * 10)  # separator line
        print("Clean Tweet: ", text_clean)
        print("-" * 80)  # separator line

a_test(910571399399559168, df_training)
a_test(910535316544610305, df_training)
a_test(910570785047228416, df_training)
a_test(910614483717914624, df_training)
a_test(910712228680155136, df_training)




# ##### Cluster the Training Data

# In[756]:


#df_training_clusters = df_training.head(1000).copy()
df_training_clusters = df_training.copy()

# Filter on Hurrucane Maria, only do this for RLHF, for SFT include all disasters
#df_training = df_training[df_training['disaster'] == 'hurricane_maria_2017']

# Define the Tweet text to use
#tweet_text = df_training_clusters['tweet_text_clean']
# Define the Tweet date to use
#tweet_date = df_training_clusters['tweet_date']

# Embed the tweets
embedding_model_id = 'all-MiniLM-L6-v2' # Use medium size sentence transformer

# Run the clustering for each file, we do this separately for each file as we will summarise the text separately for each file
n_neighbors = 10
min_dist = 0.1
min_cluster_size = 30
min_samples = 5
n_components = 45

all_results = []

for disaster_name in df_training_clusters['disaster'].unique():
    subset = df_training_clusters[df_training_clusters['disaster'] == disaster_name]
    print(f"Disaster {disaster_name} starting!")
    #tweet_text = subset['tweet_text_clean']
    embeddings_subset = embed_tweets(embedding_model_id,subset['tweet_text_clean'].tolist())
    print("Embeddings done!")

    # Get the clusters, and also the UMAP model, which we use for visualisations, for each file
    umap_model_first_reduction, X_umap_first_reduction, hdbscan_clusterer, clusters, dbcv_score = cluster_embeddings(embeddings_subset,
                                                                                                         n_neighbors,
                                                                                                         min_dist,
                                                                                                         min_cluster_size,
                                                                                                         min_samples,
                                                                                                         n_components
                                                                                                        )
    
    # Save results
    temp_result = subset.copy()
    temp_result['embeddings'] = list(embeddings_subset)
    temp_result['cluster'] = clusters
    all_results.append(temp_result)
    print(f"The disaster {disaster_name} is finished!")

# Combine all back together
df_training_clusters = pd.concat(all_results, ignore_index=True)


# In[759]:


df_training_clusters.groupby(['disaster','cluster']).size().reset_index(name='count')


# #### Visualise the Clusters
# 
# Visualise the clusters for each disaster, remember that clustering is just a way to introduce some diversity to the tweets when they are selected for summarisation. 

# In[86]:


df_training_clusters_test = df_training_clusters.head(1000).copy() # Reduce the number so runs quicker

embeddings_filtered = np.vstack(df_training_clusters_test['embeddings'].values)
clusters = df_training_clusters_test['cluster'].values

scatter_vis(embeddings_filtered, clusters, cluster_colors, title='Scatter Graph of Clusters', save_name='test', dr_model='tsne')


# In[92]:


# Visualise the clusters for each disaster
for disaster_name in df_training_clusters['disaster'].unique():
    subset = df_training_clusters[df_training_clusters['disaster'] == disaster_name]
    subset = subset.head(1000) # Reduce the number so runs quicker

    embeddings = np.vstack(subset['embeddings'].values)
    clusters = subset['cluster'].values

    scatter_vis(embeddings, clusters, cluster_colors, title=disaster_name, save_name='test', dr_model='umap')
    #scatter_vis(embeddings, clusters, cluster_colors, title='test', save_name='test', dr_model='tsne')
    #groupby_vis(embeddings, clusters)


# ##### Add Metrics for Measuring Tweet Quality
# 
# Add: 
# - ARI
# - Word count
# 
# Other metric options: 
# - Flesch Reading Ease	0–100	> 60	Higher = easier
# - Flesch-Kincaid Grade	School grade	< 8	Lower = easier
# - Gunning Fog Index	Years education	< 12	Good for spotting complex text
# - SMOG Index	School grade	< 8	Good for healthcare, education
# - ARI	School grade	< 8	Quick to compute
# - Dale-Chall	0–10 scale	< 6	Sensitive to hard words

# In[760]:


"""
print(textstat.flesch_reading_ease(text))
print(textstat.flesch_kincaid_grade(text))
print(textstat.gunning_fog(text))
print(textstat.smog_index(text))
print(textstat.automated_readability_index(text))
print(textstat.dale_chall_readability_score(text))
"""

df_training_clusters['ari'] = df_training_clusters['tweet_text_clean'].apply(textstat.automated_readability_index)
df_training_clusters['word_count'] = df_training_clusters['tweet_text_clean'].apply(lambda x: len(x.split()))

# Bin word counts
word_bins = [0, 5, 10, 20, 30, 40, np.inf]
word_labels = ['words=0-4', 'words=5-9', 'words=10-19', 'words=20-29', 'words=30-39', 'words=40+']
df_training_clusters['word_count_strata'] = pd.cut(df_training_clusters['word_count'], bins=word_bins, labels=word_labels)

# Bin ARI
ari_bins = [0, 10, 20, 30, np.inf]
ari_labels = ['ARI=0-9', 'ARI=10-19', 'ARI=20-29', 'ARI=30+']
df_training_clusters['ARI_strata'] = pd.cut(df_training_clusters['ari'], bins=ari_bins, labels=ari_labels)

# Combine into a super stratum
df_training_clusters['combined_strata'] = df_training_clusters['disaster'].astype(str) + '_' + df_training_clusters['data_split'].astype(str) + '_' + df_training_clusters['cluster'].astype(str) + '_' + df_training_clusters['word_count_strata'].astype(str) + '_' + df_training_clusters['ARI_strata'].astype(str)

df_training_clusters.head(5)


# ##### Remove Bad Quality Tweets

# In[761]:


df_training_clusters = df_training_clusters[df_training_clusters['word_count'] >= 3]
df_training_clusters = df_training_clusters[df_training_clusters['ari'] >= 5]
df_training_clusters = df_training_clusters[df_training_clusters['tweet_id'] != 1167540304108539904] # Crashes OpenAI


# #### Tweet Quality Visualisations
# 
# We use histograms and box plots to understand what bin sizes to use for each metric. 

# In[765]:


df_training_histogram = df_training_clusters[df_training_clusters['cluster'] != -1].copy()

clusters = df_training_histogram['cluster'].unique()

plt.figure(figsize=(4, 3))

for cluster in clusters:
    subset = df_training_histogram[df_training_histogram['cluster'] == cluster]
    plt.hist(subset['word_count'], bins=50, alpha=0.5, label=f'Cluster {cluster}')

plt.title('Histogram of Word Count by Cluster', fontsize=16)
plt.xlabel('Word Count', fontsize=14)
plt.ylabel('Number of Tweets', fontsize=14)
#plt.legend(title='Cluster')
plt.xlim(-10, 50)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[766]:


df_training_histogram = df_training_clusters[df_training_clusters['cluster'] != -1].copy()

clusters = df_training_histogram['cluster'].unique()

plt.figure(figsize=(4, 3))

for cluster in clusters:
    subset = df_training_histogram[df_training_histogram['cluster'] == cluster]
    plt.hist(subset['ari'], bins=50, alpha=0.5, label=f'Cluster {cluster}')

plt.title('Histogram of ARI Scores by Cluster', fontsize=16)
plt.xlabel('Automated Readability Index (ARI)', fontsize=14)
plt.ylabel('Number of Tweets', fontsize=14)
plt.legend(title='Cluster')
plt.xlim(-10, 50)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[766]:


# Group ARI scores by cluster
clustered_wc = [df_training_clusters[df_training_clusters['cluster'] == c]['word_count'].values for c in sorted(df_training_clusters['cluster'].unique())]

# Create boxplot
plt.figure(figsize=(5, 4))
plt.boxplot(clustered_wc, tick_labels=sorted(df_training_clusters['cluster'].unique()), patch_artist=True)

plt.title('Boxplot of Word Count per Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Word Count', fontsize=14)
plt.ylim(0, 30)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[764]:


# Group ARI scores by cluster
clustered_ari = [df_training_clusters[df_training_clusters['cluster'] == c]['ari'].values for c in sorted(df_training_clusters['cluster'].unique())]

# Create boxplot
plt.figure(figsize=(5, 4))
plt.boxplot(clustered_ari, tick_labels=sorted(df_training_clusters['cluster'].unique()), patch_artist=True)

plt.title('Boxplot of ARI Scores per Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Automated Readability Index (ARI)', fontsize=14)
plt.ylim(0, 40)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# #### Start the Stratified Sampling
# 
# We sample data and choose tweets, the options are to use the same distribution of the data or a uniform selection. The option selected is uniform method. 

# ##### Stratified Sampling
# 
# First we sample the same number from each stratum, these will be checked to ensure they are good quality tweets that can be summarised. We selected the uniform method, so there will be the same number of tweets in each stratum, the more the better, there will be a maximum amount for some stratum but it doesn't matter if we are slightly less. 

# In[836]:


# Sample DataFrame
df_stratified_sampling = df_training_clusters.copy()

# Number of rows per stratum
samples_per_stratum = 76780 # 76,780 is the total available

samples = []
for stratum, group_df in df_stratified_sampling.groupby('combined_strata', group_keys=False): # combined_strata = file, cluster, word count, ARI
    if len(group_df) >= samples_per_stratum:
        samples.append(group_df.sample(n=samples_per_stratum, random_state=42))
    else:
        # If not enough rows either skip or take all available
        samples.append(group_df)

# Combine
df_clusters_sample = pd.concat(samples)

print("\nFinal Uniform Sampled Data:")
print(df_clusters_sample.shape)
print(df_clusters_sample['combined_strata'].value_counts())

# Print low amount stratum
strata_counts = df_clusters_sample['combined_strata'].value_counts()

# Filter for those less than the required amount
under_sampled = strata_counts[strata_counts < samples_per_stratum]

# Print result
print("Strata with fewer than 2000 samples:")
print(under_sampled)
print(len(under_sampled))


# In[846]:


# Print low amount stratum
strata_counts = df_clusters_sample['combined_strata'].value_counts()

# Filter for those less than the required amount
under_sampled = strata_counts[strata_counts >= 1000]

# Print result
print(under_sampled)
print(len(under_sampled))


# ##### Randomly Select from the Stratum
# 
# Each group of tweets will come from the same file, then for each file we randomly select between 2 and 5 tweets, which can be from any of the stratum. 
# 
# How this works: 
# 1) Selcted number of groups, this is the final summaries to train our LLM on
# 2) For each group:
# 3) 1) Randomly pick a file
#    2) For that file, group by the strata
#    3) Check there is at least 3 or more
#    4) Randomly select between 2 and 5 tweets

# In[850]:


df = df_clusters_sample.copy()

# 1. Group tweets by file first
file_groups = {
    file: group for file, group in df.groupby('tweet_file')
}

# 2. Make nested groups (with 'tweets' list instead of nested strata keys)
groups = {}
group_selected = 60000 # Total number of summaries that will be output

for i in tqdm(range(1, group_selected + 1), 'Selecting groups'):
    selected_file = random.choice(list(file_groups.keys()))  # Randomly pick one file
    file_df = file_groups[selected_file]

    # Group by stratum within that file
    strata_in_file = {
        stratum: group[['tweet_id', 'tweet_text_clean', 'tweet_date', 'tweet_file']].to_dict('records')
        for stratum, group in file_df.groupby('combined_strata')
    }

    # Check if there are enough strata (at least 3)
    if len(strata_in_file) < 3:
        print(f"Skipping file {selected_file} because it doesn't have enough strata.")
        continue  # Skip if not enough different strata

    # Range of tweets to select in each group for summarisation
    tweet_low = 2
    tweet_high = 5
    tweets_to_select = random.randint(tweet_low, tweet_high)
    selected_strata = random.sample(list(strata_in_file.keys()), tweets_to_select)

    tweets_list = []
    for stratum in selected_strata:
        random_tweet = random.choice(strata_in_file[stratum])
        tweet_entry = {
            'tweet_id': random_tweet['tweet_id'],
            'tweet_text_clean': random_tweet['tweet_text_clean'],
            'tweet_file': random_tweet['tweet_file'],
            #'stratum': stratum  # Optionally keep track of the stratum
        }
        tweets_list.append(tweet_entry)

    groups[f'group_{i}'] = {
        'tweets': tweets_list,
        'tweet_summary': ''
    }

# Optional: print a few groups
for group_id, group in list(groups.items())[:3]:
    print(f"\n{group_id}:")
    for tweet in group['tweets']:
        print(f"  Tweet ID: {tweet['tweet_id']}, Text: {tweet['tweet_text_clean']}, File: {tweet['tweet_file']}")
    print(f"  Summary: {group['tweet_summary']}")


# In[853]:


# Print groups with tweets that have more than 1000 words
for group_id, group in list(groups.items())[:300]:
    #print(f"\n{group_id}:")

    for tweet in group['tweets']:
        word_count = len(tweet['tweet_text_clean'].split())
        if word_count > 330:
            print(f"  Tweet ID: {tweet['tweet_id']}, Word Count: {word_count}")
            print(f"  Text: {tweet['tweet_text_clean']}")
            print(f"  File: {tweet['tweet_file']}")
    
    #print(f"  Summary: {group['tweet_summary']}")


# ##### Summarise the Tweet Groups
# 
# Distillation stage 1: 
# - Use ChatGPT to generate 100, 1,000, and 5,000 summary outputs, these are used to train T5-base, FLAN-T5-base, BART-base, T5-base-efficient-tiny
# - Use OpenAI to summarise each group of tweets, the prompt is **Summarize the following in the third person like a news report**
# 
# Distillation stage 2 (skip to section ahead):
# - Use best performing T5 model (use T5 as we will be training T5-base-efficient-tiny and want to keep things consistent) to generate 10,000 summary outputs, this is used to train T5-base-efficient-tiny
# 
# 

# In[ ]:


# Setup client
client = openai.OpenAI(api_key="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")

def open_ai_summarise_text(text_to_summarize):
        # Call OpenAI to summarize
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" if preferred
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize the following in the third person like a news report:\n{text_to_summarize}"} # Can be anything, e.g. "content": "Say hello!"
        ],
        temperature=0.3,
        max_tokens=4000
    )

    # Extract the summary
    summary = response.choices[0].message.content

    return summary

# Function to summarize the tweets in a group using a language model
def summarize_group_tweets(group):
    # Concatenate the tweet texts
    tweet_texts = ". ".join([tweet["tweet_text_clean"] for tweet in group.values()])
    
    # Generate the summary (adjust max_length and min_length as needed)
    summary = open_ai_summarise_text(tweet_texts)
    
    return summary

# Summarise tweets for each group
for group_id, group in groups.items():
    # Concatenate the tweet texts (extract from the list under 'tweets')
    tweet_texts = ". ".join([tweet['tweet_text_clean'] for tweet in group['tweets']])
    
    # Generate the summary based on the tweets
    groups[group_id]['tweet_summary'] = open_ai_summarise_text(tweet_texts)  # Your summarization function
    print(f'Summarized: {group_id}')

# Sample first 3 summaries
for group_id, group in list(groups.items())[:3]:
    print(f"\n\n{group_id}:")
    for tweet in group['tweets']:
        print(f"\n  Tweet ID: {tweet['tweet_id']}, Tweet: {tweet['tweet_text_clean']}, File: {tweet['tweet_file']}")
    print(f"\n  Summary: {group['tweet_summary']}")


# ##### Cell below for RLHF only
# 
# This creates the summaries in forward and reverse direction. 

# In[575]:


# Put into a df

rows = []
for group_id, group in list(groups.items()):
    for tweet in group['tweets']:
        tweet['group_id'] = group_id
        tweet['tweet_summary'] = group['tweet_summary']
        rows.append(tweet)

df_rlhf = pd.DataFrame(rows)

# Columns to group by
cols_group = ['DateTimeDt', 'global_cluster_id', 'event_type']
# Columns to order by, need to add in others, such as text coherance, closest to cluster centre, etc
cols_order = ['DateTimeDt', 'global_cluster_id', 'rank_mmr']
df_events_all_clean = df_events_all_clean.sort_values(by = cols_order, ascending = [True, True, True])

# Normal order
df_rlhf_forward = df_rlhf.groupby('group_id').agg({
    'tweet_id': lambda tweet_ids: '|'.join(str(tweet_id) for tweet_id in tweet_ids),  # convert each ID to string
    'tweet_text_clean': lambda summaries: '. '.join(summaries),  # join all summaries together
    'tweet_summary': 'first', # Use first group name
    'tweet_file': 'first', # Use first as it will always be the same for all tweets in the group
}).reset_index()

# Reverse order
df_rlhf_reverse = df_rlhf.groupby('group_id').agg({
    'tweet_id': lambda tweet_ids: '|'.join(str(tweet_id) for tweet_id in tweet_ids),  # convert each ID to string
    'tweet_text_clean': lambda summaries: '. '.join(summaries[::-1]),  # join all summaries together
    'tweet_summary': 'first', # Use first group name
    'tweet_file': 'first', # Use first as it will always be the same for all tweets in the group
}).reset_index()

# Merge both
df_rlhf_forward = df_rlhf_forward[['group_id','tweet_id','tweet_text_clean']]
df_rlhf_reverse = df_rlhf_reverse[['group_id','tweet_text_clean']]
df_rlhf_reverse = df_rlhf_reverse.rename(columns={'tweet_text_clean': 'tweet_text_clean_reverse'})
df_rlhf_group = pd.merge(df_rlhf_forward, df_rlhf_reverse, on=['group_id'])

df_rlhf_group.head()

# Backup in flat file CSV format
df_rlhf_group.to_csv('df_rlhf_forward_reverse_100_2_5.csv', index=False)


# ##### Backup into a Flat File CSV Format
# 
# This will be used for human checking, and human summaries, backups, if needed. 

# In[854]:


rows = []
for group_id, group_data in groups.items():
    for tweet in group_data['tweets']:
        row = {
            'group_id': group_id,
            'tweet_id': tweet['tweet_id'],
            'tweet_text_clean': tweet['tweet_text_clean'],
            'tweet_summary': group_data['tweet_summary'],
            'tweet_file': tweet['tweet_file']
        }
        rows.append(row)

# Create df
df = pd.DataFrame(rows)

# Group by tweet_text_clean and concatenate the tweet_summary
df_grouped = df.groupby('group_id').agg({
    'tweet_id': lambda tweet_ids: '|'.join(str(tweet_id) for tweet_id in tweet_ids),  # convert each ID to string
    'tweet_text_clean': lambda summaries: '. '.join(summaries),  # join all summaries together
    'tweet_summary': 'first', # Use first group name
    'tweet_file': 'first', # Use first as it will always be the same for all tweets in the group
}).reset_index()

df_grouped = df_grouped[['group_id','tweet_id','tweet_text_clean','tweet_summary','tweet_file']]
df_grouped = df_grouped.sort_values(by='group_id')
df_grouped = df_grouped.rename(columns={
    "tweet_text_clean": "input_text",
    "tweet_summary": "target_text"
})
df_grouped.head(2)
df_grouped.to_csv('df_gpt_summarised_50000_2_5.csv', index=False)


# #### Import GPT Summarised Data

# In[90]:


# Now we can put data in the correct format and train each model

# Read CSV

#file_address_summaries = 'df_gpt_summarised_100_2_5.csv'
#file_address_summaries = 'df_gpt_summarised_1000_2_5.csv'
file_address_summaries = 'df_gpt_summarised_5000_2_5.csv'

df_gpt_summarised = pd.read_csv(file_address_summaries, sep=',')

# Label disaster type
df_gpt_summarised['disaster'] = df_gpt_summarised['tweet_file']
df_gpt_summarised['disaster'] = df_gpt_summarised['disaster'].str.replace(r'(_dev|_train|_test)\.tsv$', '', regex=True)

# Label dev/train/test
df_gpt_summarised['data_split'] = df_gpt_summarised['tweet_file']
df_gpt_summarised['data_split'] = df_gpt_summarised['data_split'].str.extract(r'_(train|dev|test)\.tsv$', expand=False)

# Convert to Hugging Face Dataset
df_gpt_summarised_dataset_train = df_gpt_summarised[df_gpt_summarised['data_split'] == 'train'].copy()
df_gpt_summarised_dataset_dev = df_gpt_summarised[df_gpt_summarised['data_split'] == 'dev'].copy()
df_gpt_summarised_dataset_test = df_gpt_summarised[df_gpt_summarised['data_split'] == 'test'].copy()

df_gpt_summarised_dataset_train_dataset = Dataset.from_pandas(df_gpt_summarised_dataset_train.reset_index(drop=True))
df_gpt_summarised_dataset_dev_dataset   = Dataset.from_pandas(df_gpt_summarised_dataset_dev.reset_index(drop=True))
df_gpt_summarised_dataset_test_dataset  = Dataset.from_pandas(df_gpt_summarised_dataset_test.reset_index(drop=True))

# Put in a DatasetDict mainly for the Seq2SeqTrainer
df_gpt_summarised_dataset = DatasetDict({
    "train": df_gpt_summarised_dataset_train_dataset,
    "dev": df_gpt_summarised_dataset_dev_dataset,
    "test": df_gpt_summarised_dataset_test_dataset,
})


# ##### Upload to HuggingFace

# In[ ]:


# Use your write-access token
login(token="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")

df_gpt_summarised_dataset = df_gpt_summarised_dataset.remove_columns(['input_text','data_split'])
df_gpt_summarised_dataset.push_to_hub("bluparsons/HumAIDSum100")
df_gpt_summarised_dataset.push_to_hub("bluparsons/HumAIDSum1000")


# #### Use SFT to train Models using ChatGPT Summaries
# 
# ##### First we train each model and save locally

# In[781]:


model_id = 'google/t5-efficient-tiny' # %5 efficient tiny
#model_id = "google-t5/t5-base" # T5
#model_id = 'google/flan-t5-base' # T5 FLAN
#model_id = 'google/pegasus-base' # Pegasus, Does not exist
#model_id = 'facebook/bart-base' # BART base
#model_id = 'facebook/bart-large' # BART large

#model_family = None
model_family = 'T5'

# T5 models only
if model_family == 'T5':
    print('T5 model family in use')
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
else:
    print('Non-T5 model family in use')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, return_dict=True)

dataset = df_gpt_summarised_dataset

# Run the preprocess function
tokenized_dataset = dataset.map(preprocess, fn_kwargs={"model_family": model_family})

print('Tokenisation done!')

#output_dir_name = './t5-base-finetuned-gpt_100_2_5' # 00:23:00
#output_dir_name = './t5-base-finetuned-gpt_1000_2_5' # 02:56:00 + 30
#output_dir_name = './flan-t5-base-finetuned-gpt_100_2_5' # Done
#output_dir_name = './flan-t5-base-finetuned-gpt_1000_2_5' # Done
#output_dir_name = './bart-base-finetuned-gpt_100_2_5' # Done
#output_dir_name = './bart-base-finetuned-gpt_1000_2_5' # Done
#output_dir_name = './pegasus-base-finetuned-gpt_100_2_5' # No base model
#output_dir_name = './t5-base-efficient-tiny-finetuned-gpt_1000_2_5' # 20 minutes
output_dir_name = './t5-base-efficient-tiny-finetuned-gpt_5000_2_5' # 101 minutes

#output_dir_name = './t5-base-finetuned-gpt_100_6_10' # Done
#output_dir_name = './t5-base-finetuned-gpt_1000_6_10' # 3.5 hours
#output_dir_name = './flan-t5-base-finetuned-gpt_100_6_10' # Done
#output_dir_name = './flan-t5-base-finetuned-gpt_1000_6_10'
#output_dir_name = './bart-base-finetuned-gpt_100_6_10' # Done
#output_dir_name = './bart-base-finetuned-gpt_1000_6_10'

start = time.time()

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir_name,
    #evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=5,
    logging_dir="./logs",
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"], # Training data
    eval_dataset=tokenized_dataset["dev"], # validation, need to change to validation as this is convention in HuggingFace
    tokenizer=tokenizer,
)

trainer.train()
print('Training done!')
trainer.evaluate()
print('Evaluating done!')

end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")

trainer.save_model(output_dir_name)
tokenizer.save_pretrained(output_dir_name)


# In[ ]:


logs = trainer.state.log_history
df_logs = pd.DataFrame(logs)
print(df_logs)

import matplotlib.pyplot as plt

plt.plot(df_logs['step'], df_logs['train_loss'], label='Training loss')
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.show()

df_logs.head()


# ##### Now summarise final text outputs using each model

# In[153]:


# Get a clean version of the df
df_lm_test = df_gpt_summarised_dataset_test.copy()
df_lm_test = df_lm_test[['group_id','input_text','target_text']]
df_lm_test = df_lm_test.rename(columns={'target_text': 'tweet_summary_gpt'})

df_lm_test.head(2)


# In[158]:


df_lm_test.head(5)


# In[159]:


# Test data
# Summarise using the trained LLMs and update final df

"""
# This is the efficient way to run, but having problems, it is only summarising first 2 rows

model_id = "./t5-base-finetuned-gpt_100_2_5" # The model we will use to summariase the text
tok, mdl, device, family = load_summarizer(model_id, model_family="t5")

df_lm_test = summarise_dataframe(
    df_lm_test,
    input_col="input_text",
    output_col='tweet_summary_t5-base-finetuned-gpt-100',
    tok=tok,
    model=mdl,
    device=device,
    model_family=family,
    batch_size=16,          # tune based on VRAM
    max_input_length=512,
    max_new_tokens=128,
    num_beams=4,
    do_sample=False
)
print(f"{model_id} done!")
"""

#df_lm_test = df_lm_test.head(200)

# Summarise the test data, we will then output the metrics and human evaluation scores
"""
# T5 tiny
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-efficient-tiny',
                                         model_id = 'google/t5-efficient-tiny',
                                         model_family = 't5')
print('Done!')

#T5 small
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-small',
                                         model_id = 'google-t5/t5-small',
                                         model_family = 't5')
print('Done!')

# T5
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-base',
                                         model_id = 'google-t5/t5-base',
                                         model_family = 't5')
print('Done!')

#df_events_all_group = summarise_each_llm(df_events_all_group,
#                                         text_input = 'tweet_text_clean',
#                                         text_output = 'tweet_summary_t5-base-cnn',
#                                         model_id = '',
#                                         model_family = 't5')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-base-samsum',
                                         model_id = 'amagzari/t5-base-finetuned-samsum-v2',
                                         model_family = 't5')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-base-xsum',
                                         model_id = 'PavanNeerudu/t5-base-finetuned-xsum',
                                         model_family = 't5')
print('Done!')

# FLAN-T5
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_flan-t5-base',
                                         model_id = 'google/flan-t5-base',
                                         model_family = 't5')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_flan-t5-base-cnn',
                                         model_id = 'braindao/flan-t5-cnn',
                                         model_family = 't5')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_flan-t5-base-samsum',
                                         model_id = 'philschmid/flan-t5-base-samsum',
                                         model_family = 't5')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_flan-t5-base-xsum',
                                         model_id = 'brutusxu/flan-t5-base-finetuned-xsum',
                                         model_family = 't5')
print('Done!')

# Pegasus
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_pegasus-base-cnn',
                                         model_id = 'google/pegasus-cnn_dailymail',
                                         model_family = 'pegasus')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_pegasus-base-cnn-samsum',
                                         model_id = 'Feluda/pegasus-samsum',
                                         model_family = 'pegasus')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_pegasus-base-xsum',
                                         model_id = 'google/pegasus-xsum',
                                         model_family = 'pegasus')
print('Done!')
"""
# BART base
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_bart-base',
                                         model_id = 'facebook/bart-base',
                                         model_family = 'bart')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_bart-base-cnn',
                                         model_id = 'ainize/bart-base-cnn',
                                         model_family = 'bart')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_bart-base-saumsum',
                                         model_id = 'philschmid/bart-base-samsum',
                                         model_family = 'bart')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_bart-base-xsum',
                                         model_id = 'Prikshit7766/bart-base-xsum',
                                         model_family = 'bart')
print('Done!')
"""
# Now fine-tuned models

# T5 Efficient Tiny
df_lm_test = summarise_each_llm(df_lm_test,
                                 text_input = 'input_text',
                                 text_output = 'tweet_summary_t5-base-efficient-tiny-finetuned-1k',
                                 model_id = './t5-base-efficient-tiny-finetuned-gpt_1000_2_5',
                                 model_family = 't5')
print('Done!')

# T5
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-base-finetuned-gpt-100',
                                         model_id = './t5-base-finetuned-gpt_100_2_5',
                                         model_family = 't5')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-base-finetuned-gpt-1k',
                                         model_id = './t5-base-finetuned-gpt_1000_2_5',
                                         model_family = 't5')
print('Done!')

# FLAN-T5
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_flan-t5-base-finetuned-gpt-100',
                                         model_id = './flan-t5-base-finetuned-gpt_100_2_5',
                                         model_family = 't5')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_flan-t5-base-finetuned-gpt-1k',
                                         model_id = './flan-t5-base-finetuned-gpt_1000_2_5',
                                         model_family = 't5')
print('Done!')

# BART
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_bart-base-finetuned-gpt-100',
                                         model_id = './bart-base-finetuned-gpt_100_2_5',
                                         model_family = 'bart')
print('Done!')

df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_bart-base-finetuned-gpt-1k',
                                         model_id = './bart-base-finetuned-gpt_1000_2_5',
                                         model_family = 'bart')
print('Done!')
"""


# In[785]:


# Testing

#tokenizer = BartTokenizer.from_pretrained(model_id)
#model = BartForConditionalGeneration.from_pretrained(model_id)
model_id = './t5-base-efficient-tiny-finetuned-gpt_5000_2_5'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, return_dict=True)
summarise_text('Our hearts and prayers go out to all those who affected by HurricaneMaria and are now moving towards recovery. There will be a HurricaneMaria fundraiser at Isla Verde,  N. American Street in Philly.',
               model,
               tokenizer,
               model_type="t5")


# In[ ]:


output_summaries(['tweet_summary_bart-base-finetuned-gpt'],[1])


# #### RLHF
# 
# ##### We can select any 3 tweets and summarise backwards (forwards is alread done above), this is still diverse and this is acceptable as RLHF is about optimising generation quality via preference feedback. 'For RLHF rather than apply the same sampling as SFT, we are more flexible in this approach without strict stratification criteria, this allows for diverse semantic combinations that reflect real-world summarisation'. SFT structured, RLHF event-based
# 
# We now generate summaries using the reverse order text, this goes into a df then we pivot and output to CSV for human evaluation. Data has come from GPT summarised data, this was done in previous code above and does not use our event data, i.e. there is no data leakage. 
# 
# We use Direct Preference Optimization (DPO) as it is simple and stable, compared to PPO, DPO is also specifically designed for RLHF with ranked pairs, it also works with encoder-decoder models. 

# In[ ]:


# Load from CSV

file_address_summaries = 'df_rlhf_forward_reverse_100_2_5.csv'

df_rlhf_group = pd.read_csv(file_address_summaries, sep=',')

df_rlhf_group_dataset = DatasetDict({
    "train": df_rlhf_summarised,
    "dev": df_rlhf_summarised,
    "test": df_rlhf_summarised,
})

df_rlhf_group.head()


# In[577]:


df_events_all_group_rlhf = df_rlhf_group.copy()


# In[582]:


# Generate summaries for all the fine-tuned base models in both normal and reverse direction
# This is not the event data, it is test data summarised using GPT

# T5 normal
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_t5-base-finetuned-gpt_100_2_5',
                                         model_id = './t5-base-finetuned-gpt_100_2_5',
                                         model_family = 't5')
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_t5-base-finetuned-gpt_1000_2_5',
                                         model_id = './t5-base-finetuned-gpt_1000_2_5',
                                         model_family = 't5')
# T5 Reverse
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean_reverse',
                                         text_output = 'tweet_summary_t5-base_reverse-finetuned-gpt_100_2_5',
                                         model_id = './t5-base-finetuned-gpt_100_2_5',
                                         model_family = 't5')
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean_reverse',
                                         text_output = 'tweet_summary_t5-base_reverse-finetuned-gpt_1000_2_5',
                                         model_id = './t5-base-finetuned-gpt_1000_2_5',
                                         model_family = 't5')
# FLAN-T5 normal
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_flan-t5-base-finetuned-gpt_100_2_5',
                                         model_id = './flan-t5-base-finetuned-gpt_100_2_5',
                                         model_family = 't5')
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_flan-t5-base-finetuned-gpt_1000_2_5',
                                         model_id = './flan-t5-base-finetuned-gpt_1000_2_5',
                                         model_family = 't5')
# FLAN-T5 reverse
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean_reverse',
                                         text_output = 'tweet_summary_flan-t5-base_reverse-finetuned-gpt_100_2_5',
                                         model_id = './flan-t5-base-finetuned-gpt_100_2_5',
                                         model_family = 't5')
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean_reverse',
                                         text_output = 'tweet_summary_flan-t5-base_reverse-finetuned-gpt_1000_2_5',
                                         model_id = './flan-t5-base-finetuned-gpt_1000_2_5',
                                         model_family = 't5')
# BART normal
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_bart-base-finetuned-gpt_100_2_5',
                                         model_id = './bart-base-finetuned-gpt_100_2_5',
                                         model_family = 'bart')
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean',
                                         text_output = 'tweet_summary_bart-base-finetuned-gpt_1000_2_5',
                                         model_id = './bart-base-finetuned-gpt_1000_2_5',
                                         model_family = 'bart')
# BART reverse
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean_reverse',
                                         text_output = 'tweet_summary_bart_reverse-base-finetuned-gpt_100_2_5',
                                         model_id = './bart-base-finetuned-gpt_100_2_5',
                                         model_family = 'bart')
df_events_all_group_rlhf = summarise_each_llm(df_events_all_group_rlhf,
                                         text_input = 'tweet_text_clean_reverse',
                                         text_output = 'tweet_summary_bart-base_reverse-finetuned-gpt_1000_2_5',
                                         model_id = './bart-base-finetuned-gpt_1000_2_5',
                                         model_family = 'bart')


# In[583]:


df_events_all_group_rlhf


# In[588]:


# For each base model and the reverse data, run the process to train and save the new model

summary_col_pairs = [
    # Forward and reverse
    # T5
    ('tweet_summary_t5-base-finetuned-gpt_100_2_5', 'tweet_summary_t5-base_reverse-finetuned-gpt_100_2_5'),
    ('tweet_summary_t5-base-finetuned-gpt_1000_2_5', 'tweet_summary_t5-base_reverse-finetuned-gpt_1000_2_5'),
    # FLAN-T5
    ('tweet_summary_flan-t5-base-finetuned-gpt_100_2_5', 'tweet_summary_flan-t5-base_reverse-finetuned-gpt_100_2_5'),
    ('tweet_summary_flan-t5-base-finetuned-gpt_1000_2_5', 'tweet_summary_flan-t5-base_reverse-finetuned-gpt_1000_2_5'),
    # BART
    ('tweet_summary_bart-base-finetuned-gpt_100_2_5', 'tweet_summary_bart_reverse-base-finetuned-gpt_100_2_5'),
    ('tweet_summary_bart-base-finetuned-gpt_1000_2_5', 'tweet_summary_bart-base_reverse-finetuned-gpt_1000_2_5'),
]

summary_dir = os.path.join(current_dir, r'Code\summaries')
os.makedirs(summary_dir, exist_ok=True)

# Loop each text pairs
dfs = []
for forward_col, reverse_col in summary_col_pairs:
    forward_model_name = forward_col.replace('tweet_summary_', '')
    reverse_model_name = reverse_col.replace('tweet_summary_', '').replace('_reverse', '')

    # Forward
    df_events_all_group_rlhf_pivot_f = pd.melt(
        df_events_all_group_rlhf,
        id_vars=['group_id', 'tweet_text_clean'],
        value_vars=[forward_col],
        var_name='summary_model',
        value_name='summary_text'
    )

    # Reverse
    df_events_all_group_rlhf_pivot_r = pd.melt(
        df_events_all_group_rlhf,
        id_vars=['group_id', 'tweet_text_clean'],
        value_vars=[reverse_col],
        var_name='summary_model',
        value_name='summary_text_reverse'
    )
    df_events_all_group_rlhf_pivot_r['summary_model'] = df_events_all_group_rlhf_pivot_r['summary_model'].str.replace('_reverse', '', regex=False)

    # Merge
    df_events_all_group_rlhf_pivot = pd.merge(
        df_events_all_group_rlhf_pivot_f,
        df_events_all_group_rlhf_pivot_r,
        on=['group_id', 'tweet_text_clean', 'summary_model'],
        how='outer'
    )

    # Sort and merge
    df_events_all_group_rlhf_pivot = df_events_all_group_rlhf_pivot.sort_values(by='group_id')
    df_events_all_group_rlhf_pivot['reverse_better'] = 0
    dfs.append(df_events_all_group_rlhf_pivot)

df_events_all_group_rlhf_pivot = pd.concat(dfs, ignore_index=True).sort_values(by=['summary_model', 'group_id'])

# Export to CSV
file_address_human_rlhf = os.path.join(summary_dir, 'output_human_rlhf_all.csv')
df_events_all_group_rlhf_pivot.to_csv(file_address_human_rlhf, index=False)

#df_events_all_group_rlhf_pivot.head()


# In[589]:


# Import from CSV
summary_dir = os.path.join(current_dir, r'Code\summaries')
file_address_human_rlhf = os.path.join(summary_dir, r'output_human_rlhf_all.csv')
df_human_rlhf = pd.read_csv(file_address_human_rlhf)

df_human_rlhf = df_human_rlhf.drop('Unnamed: 7', axis=1)

df_human_rlhf.head()


# In[616]:


# This takes the df and transforms it into a dataset with prompt, chosen, and rejected fields

# T5
#df_human_rlhf_model = df_human_rlhf[df_human_rlhf['summary_model'] == 'tweet_summary_t5-base-finetuned-gpt_100_2_5'].copy() # Done
#df_human_rlhf_model = df_human_rlhf[df_human_rlhf['summary_model'] == 'tweet_summary_t5-base-finetuned-gpt_1000_2_5'].copy() # Done
# FLAN-T5
#df_human_rlhf_model = df_human_rlhf[df_human_rlhf['summary_model'] == 'tweet_summary_flan-t5-base-finetuned-gpt_100_2_5'].copy() # Done
#df_human_rlhf_model = df_human_rlhf[df_human_rlhf['summary_model'] == 'tweet_summary_flan-t5-base-finetuned-gpt_1000_2_5'].copy() # Done
# BART
#df_human_rlhf_model = df_human_rlhf[df_human_rlhf['summary_model'] == 'tweet_summary_bart-base-finetuned-gpt_100_2_5'].copy() # Done
df_human_rlhf_model = df_human_rlhf[df_human_rlhf['summary_model'] == 'tweet_summary_bart-base-finetuned-gpt_1000_2_5'].copy() # Done

print(df_human_rlhf_model.head(50))

dataset = build_dpo_dataset(df_human_rlhf_model)
dataset


# In[617]:


#model_id = './t5-base-finetuned-gpt_100_2_5' # Done
#model_id = './t5-base-finetuned-gpt_1000_2_5' # Done
#model_id = './flan-t5-base-finetuned-gpt_100_2_5' # Done
#model_id = './flan-t5-base-finetuned-gpt_1000_2_5' # Done
#model_id = './bart-base-finetuned-gpt_100_2_5' # Done
model_id = './bart-base-finetuned-gpt_1000_2_5' # Done

#output_dir_name = './t5-base-finetuned-gpt_100_2_5-rlhf' # 10 minutes
#output_dir_name = './t5-base-finetuned-gpt_1000_2_5-rlhf' # 10 minutes
#output_dir_name = './flan-t5-base-finetuned-gpt_100_2_5-rlhf' # 14 minutes
#output_dir_name = './flan-t5-base-finetuned-gpt_1000_2_5-rlhf' # 12 minutes
#output_dir_name = './bart-base-finetuned-gpt_100_2_5-rlhf' # 8 minutes
output_dir_name = './bart-base-finetuned-gpt_1000_2_5-rlhf' # 7 minutes

model_family = None
#model_family = 'T5'

# T5 models only
if model_family == 'T5':
    print('T5 model family in use')
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
else:
    print('Non-T5 model family in use')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, return_dict=True)

# Tokenisation not needed as this is a comparison method, not SFT

start = time.time()

# Below directly inherits from TrainingArguments but with beta added
#training_args = Seq2SeqTrainingArguments(
dpo_config = DPOConfig(
    output_dir=output_dir_name,
    #evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=5,
    logging_dir="./logs",
    push_to_hub=False,
    beta=0.1, # New
    fp16=False, # New
    bf16=False, # New
)

# Initialize DPO trainer
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    #tokenizer=tokenizer,
    train_dataset=dataset,      # Should be in the required format (with chosen & rejected responses)
    eval_dataset=None,          # Optional
)

# Train and save
trainer.train()
print('Training done!')

# Evaluate not needed

end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")

trainer.save_model(output_dir_name)
tokenizer.save_pretrained(output_dir_name)

#dpo_config = DPOConfig(output_dir=output_dir_name,fp16=False,bf16=False)
#trainer = DPOTrainer(model=model, args=dpo_config, processing_class=tokenizer, train_dataset=dataset)


# ##### Now we use the new RLHF models to output summaries

# In[623]:


df_events_all_group.head(2)


# In[141]:


# Use PCO to output summaries on the test data so we can measure performance with metrics and human scores

# T5
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-base-finetuned-gpt-100-rlhf',
                                         model_id = './t5-base-finetuned-gpt_100_2_5-rlhf',
                                         model_family = 't5')
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-base-finetuned-gpt-1k-rlhf',
                                         model_id = './t5-base-finetuned-gpt_1000_2_5-rlhf',
                                         model_family = 't5')

# FLAN-T5
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_flan-t5-base-finetuned-gpt-100-rlhf',
                                         model_id = './flan-t5-base-finetuned-gpt_100_2_5-rlhf',
                                         model_family = 't5')
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_flan-t5-base-finetuned-gpt-1k-rlhf',
                                         model_id = './flan-t5-base-finetuned-gpt_1000_2_5-rlhf',
                                         model_family = 't5')
# BART
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_bart-base-finetuned-gpt-100-rlhf',
                                         model_id = './bart-base-finetuned-gpt_100_2_5-rlhf',
                                         model_family = 'bart')
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_bart-base-finetuned-gpt-1k-rlhf',
                                         model_id = './bart-base-finetuned-gpt_1000_2_5-rlhf',
                                         model_family = 'bart')

df_lm_test.head(5)


# In[626]:


df_events_all_group.to_csv('output_human_ranking_with_tuned_models_v4.csv', index=False)


# #### Stage 2 Distillation
# 
# This is where we generate 5,000 summaries using the previously distillied T5-Base-HumAID_1000 model, and train the T5-efficient-tiny model. 

# In[27]:


# Load the data we have selected using stratified sampling

#file_address_summaries = 'df_gpt_summarised_60000_2_5.csv'
#file_address_summaries = 'df_gpt_summarised_60000_2_5_Part1Final.csv'
file_address_summaries = 'df_gpt_summarised_60000_2_5_Part2Final.csv'

df_grouped = pd.read_csv(file_address_summaries, sep=',')

df_grouped.head(2)


# In[29]:


# Summarise using a more efficient method than the summarise_each_event def

model_id = "./t5-base-finetuned-gpt_1000_2_5" # The model we will use to summariase the text
tok, mdl, device, family = load_summarizer(model_id, model_family="t5")

# Create summaries using the distilled model, this will then be used to train the T5-efficient-tiny model
df_grouped_training = df_grouped.copy()
#df_grouped_training = df_grouped_training.head(10)

df_grouped_training = summarise_dataframe(
    df_grouped_training,
    input_col="input_text",
    output_col='target_text',
    tok=tok, model=mdl, device=device,
    model_family=family,
    batch_size=16,          # tune based on VRAM
    max_input_length=512,
    max_new_tokens=128,
    num_beams=4,
    do_sample=False
)
print(f"{model_id} done!")

df_grouped_training.head(2)


# In[816]:


# Create summaries using the distilled model, this will then be used to train the T5-efficient-tiny model
df_grouped_training = df_grouped.copy()

df_grouped_training = df_grouped_training.head(10)

# T5 Efficient Tiny
# Time taken: 
df_grouped_training = summarise_each_llm(df_grouped_training,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-base-finetuned-t5-base_5000_2_5',
                                         model_id = './t5-base-finetuned-gpt_1000_2_5', # Use the T5-base_1000 model we trained in distillation 1
                                         model_family = 't5')

df_grouped_training.head()


# In[30]:


# Backup to CSV
df_grouped_training.to_csv('df_t5-base-1000_summarised_2000_2_5.csv', index=False)


# In[22]:


# Load from CSV
file_address_summaries = 'df_t5-base-1000_summarised_4000_2_5.csv'

df_gpt_summarised = pd.read_csv(file_address_summaries, sep=',')

# Label disaster type
df_gpt_summarised['disaster'] = df_gpt_summarised['tweet_file']
df_gpt_summarised['disaster'] = df_gpt_summarised['disaster'].str.replace(r'(_dev|_train|_test)\.tsv$', '', regex=True)

# Label dev/train/test
df_gpt_summarised['data_split'] = df_gpt_summarised['tweet_file']
df_gpt_summarised['data_split'] = df_gpt_summarised['data_split'].str.extract(r'_(train|dev|test)\.tsv$', expand=False)

# Convert to Hugging Face Dataset
df_gpt_summarised_dataset_train = df_gpt_summarised[df_gpt_summarised['data_split'] == 'train'].copy()
df_gpt_summarised_dataset_dev = df_gpt_summarised[df_gpt_summarised['data_split'] == 'dev'].copy()
df_gpt_summarised_dataset_test = df_gpt_summarised[df_gpt_summarised['data_split'] == 'test'].copy()

df_gpt_summarised_dataset_train_dataset = Dataset.from_pandas(df_gpt_summarised_dataset_train.reset_index(drop=True))
df_gpt_summarised_dataset_dev_dataset   = Dataset.from_pandas(df_gpt_summarised_dataset_dev.reset_index(drop=True))
df_gpt_summarised_dataset_test_dataset  = Dataset.from_pandas(df_gpt_summarised_dataset_test.reset_index(drop=True))

df_gpt_summarised_dataset = DatasetDict({
    "train": df_gpt_summarised_dataset_train_dataset,
    "dev": df_gpt_summarised_dataset_dev_dataset,
    "test": df_gpt_summarised_dataset_test_dataset,
})

df_gpt_summarised_dataset


# In[23]:


# Now train the model

#model_id = 'google/t5-efficient-tiny' # The base model we will be training and trying to improve
model_id = 'google-t5/t5-small' # The base model we will be training and trying to improve

#model_family = None
model_family = 'T5'

# T5 models only
if model_family == 'T5':
    print('T5 model family in use')
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
else:
    print('Non-T5 model family in use')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, return_dict=True)

dataset = df_gpt_summarised_dataset

# Run the preprocess function
tokenized_dataset = dataset.map(preprocess, fn_kwargs={"model_family": model_family})

print('Tokenisation done!')

output_dir_name = './t5-base-efficient-tiny-finetuned-gpt_5000_2_5'
output_dir_name = './t5-small-finetuned-gpt_1k_4k'

start = time.time()

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir_name,
    #evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=5,
    logging_dir="./logs",
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"], # Training data
    eval_dataset=tokenized_dataset["dev"], # validation, need to change to validation as this is convention in HuggingFace
    tokenizer=tokenizer,
)

trainer.train()
print('Training done!')
trainer.evaluate()
print('Evaluating done!')

end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")

trainer.save_model(output_dir_name)
tokenizer.save_pretrained(output_dir_name)



# In[144]:


# Summarise the test data

# T5
# T5 tiny
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-base-efficient-tiny-finetuned-gpt-1k-4k',
                                         model_id = './t5-base-efficient-tiny-finetuned-gpt_5000_2_5',
                                         model_family = 't5')

# T5 small
df_lm_test = summarise_each_llm(df_lm_test,
                                         text_input = 'input_text',
                                         text_output = 'tweet_summary_t5-small-finetuned-gpt-1k-4k',
                                         model_id = './t5-small-finetuned-gpt_1k_4k',
                                         model_family = 't5')
 


# ##### Output into CSV the summarisation of test data

# In[147]:


df_lm_test.to_csv('df_lm_test.csv', index=False)


# In[ ]:


# Some testing of summarisation

#model_id = './bart-base-finetuned-gpt_1000_2_5'
#tokenizer = BartTokenizer.from_pretrained(model_id)
#model = BartForConditionalGeneration.from_pretrained(model_id)

#model_id = './t5-base-efficient-tiny-finetuned-gpt_5000_2_5'
model_id = './t5-base-finetuned-gpt_1000_2_5'
#model_id = './t5-small-finetuned-gpt_1k_4k'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, return_dict=True)
summarise_text('Bangladesh has decided to provide a cash assistance of US $ ,  to support the victims of recent flood and landslides in. Heartbreaking numbers from the situation in Sri Lanka. Lord, give wisdom to those working to aid and bring comfort to all.. Pakistan Navy conducts relief operations in flood-battered Sri Lanka  news. Three Chinese naval ships arrive in Sri Lanka to assist in flood relief -. In Gampaha our volunteers conducted a firstaid camp for flood affected in Dompe and Biyagama, supported by  LKA FloodSL',
               model,
               tokenizer,
               model_type='bart',
              )

#Bangladesh has decided to provide a cash assistance of US $ ,  to support the victims of recent flood and landslides in. 
#Heartbreaking numbers from the situation in Sri Lanka. Lord, give wisdom to those working to aid and bring comfort to all.. 
#Pakistan Navy conducts relief operations in flood-battered Sri Lanka  news. Three Chinese naval ships arrive in Sri Lanka to 
#assist in flood relief -. In Gampaha our volunteers conducted a firstaid camp for flood affected in Dompe and Biyagama, 
#supported by  LKA FloodSL


# #### Summarisation Metrics

# In[145]:


df_lm_test


# In[146]:


# Get ROUGE and BLEU scores for the generated summaries, use the human summary as a reference point

# Only run scoring on columns with summarised text in, leave the human summary as a check, all values should be 1
exclude_cols = ['group_id', 'input_text']
selected_cols = [col for col in df_lm_test.columns if col not in exclude_cols]
references = df_lm_test['tweet_summary_gpt'].tolist()

results = []

#Run for each summary column
for col in selected_cols:
    candidates = df_lm_test[col].tolist()
    bleu_results, rouge_results = rouge_bleu_scores(references, candidates)

    results.append({
        "model": col,
        "ROUGE-1": round(rouge_results['rouge1'], 2),
        "ROUGE-2": round(rouge_results['rouge2'], 2),
        "ROUGE-L": round(rouge_results['rougeL'], 2),
        "BLEU": round(bleu_results['bleu'] * 100, 2),
    })

df_scores = pd.DataFrame(results)
df_scores = df_scores.sort_values(by="model", ascending=True)

"""
#P, R, F1 = score(candidates, references, lang="en", verbose=True)

#print(f"BERTScore Precision: {P.mean().item():.4f}")
#print(f"BERTScore Recall: {R.mean().item():.4f}")
#print(f"BERTScore F1: {F1.mean().item():.4f}")
"""

df_scores


# In[ ]:




