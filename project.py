# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:36:05 2024

@author: Blu Parsons - CS23355
"""

# Import libraries
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime

# System libraries
import sys
import os

sys.path.append(r'C:\Users\mail\OneDrive\University\04 Bristol\07 Project') # To import procedures from other python files

# Import Github packages
# https://github.com/oduwsdl/tweetedat
import Code.source_third_party.tweetedat.script.TimestampEstimator as ta
# https://github.com/guyfe/Tweetsumm
import Code.source_third_party.Tweetsumm.tweet_sum_processor as tsp

# Import transformer models
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import umap
import hdbscan
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
#from clustering_metrics import S_Dbw
from sklearn.model_selection import ParameterGrid

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')
from bertopic.vectorizers import ClassTfidfTransformer

# T5 Fine Tuning
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments

# Transformers
from transformers import AutoModelWithLMHead
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

os.chdir(r'C:\Users\mail\OneDrive\University\04 Bristol\07 Project')
current_dir = os.path.dirname(os.path.abspath('project.py'))
module_dir = os.path.join(current_dir, 'modules')
data_dir = os.path.join(current_dir, r'Code\data')
visualisations_dir = os.path.join(current_dir, r'Code\visualisations')
models_dir = os.path.join(current_dir, r'Code\models')

# current_path = os.getcwd()  # Outputs the current working directory

####################################################
# Import Data
####################################################
# Import the data, this is in TSV (Tab Separated Values) format
file_address = os.path.join(data_dir, r'HumAID_data_events_set1_47K\events_set1\canada_wildfires_2016\canada_wildfires_2016_train.tsv')

df_data = pd.read_csv(file_address, sep='\t')

####################################################
# Data Cleanse

# Get Date Time from Snowflafe ID
# Preprocess Tweets
####################################################
# Get Unix timestamp by inputting a twitter snowflake ID
# tstamp = ta.find_tweet_timestamp(721630546711986178)

# Get a date format datetime by inputting a Unix timestamp
# utcdttime = datetime.utcfromtimestamp(tstamp / 1000)


# Change date format from Unix timestamp to datetime as a string
def timestamp_to_str(tstamp):
    utcdttime = datetime.utcfromtimestamp(tstamp / 1000)
    utcdttime = utcdttime.strftime('%Y-%m-%d')
    return utcdttime


def date_time_stamp_to_str(tstamp):
    utcdttime = datetime.utcfromtimestamp(tstamp / 1000)
    utcdttime = utcdttime.strftime('%Y-%m-%d, %H:%M:%S')  # H is in 24 hour clock
    return utcdttime


# Add Tweet DateTime in Unix epoch time format
df_data['DateTimeUnix'] = df_data.apply(lambda row: ta.find_tweet_timestamp(row['tweet_id']), axis=1)

# Add Tweet DateTime in string format
df_data['DateTimeStr'] = df_data.apply(lambda row: timestamp_to_str(ta.find_tweet_timestamp(row['tweet_id'])), axis=1)

# Data checks
# df_data['DateTimeStr'].unique()
# df_data.dtypes
# df_data['DateTimeStr'].value_counts()

# We could do some analysis around things like word frequency

####################################################
# Preprocessing and Normalising of Tweets

# Common preprocessing methods for social media text
# Lowercase
# Remove punctuation
# Remove stopwords
# Remove URLs or replace with standard token, this is a form of normalisation, e.g. <url>
# Remove HTML tags
# Known social media features, e.g. @user, RT, #hashtags
# Consider #hashtag words and not just replace them, e.g. #whataday to what a day
# Change emojis to words
# Change emoticons to words
# Lemmatisation
# Stemming
# Remove numbers
# Tokenisation

# References
# https://developers.google.com/edu/python/regular-expressions
# https://github.com/DavidBert/Tweet_normalizer/blob/master/normalize_tweets.py
# https://github.com/XuanyiZ/Text-Normalization
####################################################


def preprocessing_remove_urls(text):
    # Define a RE pattern for URLs
    url_pattern = re.compile(r'https?://\S+')
    url_removed = url_pattern.sub('', text)
    return url_removed


def preprocessing_remove_string(text, strToRemove):
    string_pattern = strToRemove
    string_removed = re.sub(string_pattern, '', text)
    return string_removed


def preprocessing_remove_digits(text):
    # Define a RE pattern for digits
    url_pattern = re.compile(r'\d')
    url_removed = url_pattern.sub('', text)
    return url_removed


def preprocessing_replace_emoji(text):
    # Define a RE pattern for emoji
    emoji_pattern = re.compile(r'\:\-\)')
    emoji_replaced = emoji_pattern.sub('smiling_face', text)

    # emoji_pattern = re.compile(r'\:\-\(')
    # emoji_replaced = emoji_pattern.sub('sad_face', text)

    # emoji_pattern = re.compile(r'\:\-\/')
    # emoji_replaced = emoji_pattern.sub('angry_face', text)

    # emoji_pattern = re.compile(r'\:\-\\')
    # emoji_replaced = emoji_pattern.sub('angry_face', text)
    return emoji_replaced


def preprocessing_remove_retweet_username(text):
    # Define a RE pattern for usernames
    # RT<space>@username:<space>
    url_pattern = re.compile(r'RT @[\w.-]+: ')
    url_removed = url_pattern.sub('', text)
    return url_removed


def preprocessing_remove_username(text):
    # Define a RE pattern for usernames
    # RT<space>@username:<space>
    url_pattern = re.compile(r'@[\w.-]+')
    url_removed = url_pattern.sub('', text)
    return url_removed


# Run the pre-processing functions
def custom_preprocessing(text):
    # Remove URLs
    text = preprocessing_remove_urls(text)

    # Remove all usernames
    text = preprocessing_remove_retweet_username(text)

    # Remove all usernames
    text = preprocessing_remove_username(text)

    # Remove the symbol #
    text = preprocessing_remove_string(text, '#')

    # Remove numerical digits
    text = preprocessing_remove_digits(text)

    # remove special chars
    # text = re.sub("\\W"," ",text)

    # Change icon emojis into describing word
    # text = emoji.demojize(text)

    # Change text emojis into work, e.g. :-)
    # text = preprocessing_replace_emoji(text)

    return text


df_data['tweet_text_clean'] = df_data.apply(
    lambda row: custom_preprocessing(row['tweet_text']), axis=1)

####################################################
# NLP Pipeline

# This is a sequence of interconnected steps to transform raw tweets into a
# desired output, this enables machines to better comprehend and understand
# human text.

# Preprocessing has already been done, e.g. removing punctuation

# BERTopic is a topic modelelling framework that allows users to create a
# customised topic model, it is flexible and modular, this means that models
# can be changed and updated easily.

# Tokenise tweets
# Embedding of tweets
# Reducing dimensionality of embeddings
# Clustering reduced embeddings into topics
# Tokenization of topics
# Weight tokens
# Represent topics with one or multiple representations (optional)
####################################################

# Create BERTopic model with customized UMAP and HDBSCAN parameters and return
# the topics and probabilities
def create_bertopic_model(umap_params, hdbscan_params):
    topic_model = BERTopic(embedding_model=embedding_model,
                           umap_model=umap.UMAP(**umap_params,
                                                random_state=42),
                           hdbscan_model=hdbscan.HDBSCAN(**hdbscan_params
                                                         ),
                           representation_model=representation_model,
                           vectorizer_model=vectorizer_model,
                           ctfidf_model=ctfidf_model,
                           top_n_words=5,
                           language='english',
                           # All probabilities, all topics across all documents
                           calculate_probabilities=True,
                           verbose=True)
    # Train the model and assign topic labels and cluster probabilities
    # topics = topic labels for each tweet
    # probs = probabilities for each topic that is assigned
    # gives the probabilities of the corresponding tweet of being part of
    # each topic
    topics, probs = topic_model.fit_transform(tweet_text, embeddings)
    return topics, probs, topic_model


# BERTopic error
# probs_final in topic_model.get_document_info contains the final probability
# and topic, this should match the topics with the highest probability but it
# does not.
# Impact: 45/1,156 = 3.9%

# Fine tuning for the BERTopic model
# This iterates through the UMAP and HDBSCAN parameters and calculates the
# score for each metric, the best score and BERTopic model is saved
def fine_tune_bertopic_model(umap_param_grid, hdbscan_param_grid):
    best_score = float('inf')
    best_umap_params = None
    best_hdbscan_params = None
    best_topic_model = None
    best_topics = None
    best_probs = None
    iteration_current = 0
    iteration_total = len(umap_param_grid) * len(hdbscan_param_grid)

    for umap_params in ParameterGrid(umap_param_grid):
        for hdbscan_params in ParameterGrid(hdbscan_param_grid):
            iteration_current += 1
            # Create BERTopic model for each UMAP and HDBSCAN parameters
            topics, probs, topic_model = create_bertopic_model(umap_params,
                                                               hdbscan_params)

            # Filter out noise (-1 topics) as these have no meaningful
            # distances to a cluster, filtering will ensure the score reflects
            # only valid clusters
            valid_indices = [i for i,
                             topic in enumerate(topics) if topic != -1]
            filtered_embeddings = embeddings[valid_indices]
            filtered_labels = [topics[i] for i in valid_indices]

            # Ensure at least two clusters for DBI calculation
            if len(set(filtered_labels)) > 1:
                # Silhouette score
                ss_index = silhouette_score(filtered_embeddings,
                                            filtered_labels)
                # Calinski-Harabasz Index
                ch_index = calinski_harabasz_score(filtered_embeddings,
                                                   filtered_labels)
                # Davies-Bouldin Index
                db_index = davies_bouldin_score(filtered_embeddings,
                                                filtered_labels)
                # S_Dbw
                #s_dbw_index = S_Dbw(filtered_embeddings,
                #                    filtered_labels,
                #                    method='Halkidi')  # Or use Kim

                print(f"Iteration: {iteration_current} of {iteration_total}")
                print(f"UMAP Params: {umap_params}")
                print(f"HDBSCAN Params: {hdbscan_params}")
                print(f"Silhouette Score: {ss_index}")
                print(f"Calinski-Harabasz Index: {ch_index}")
                print(f"Davies-Bouldin Index: {db_index}")
                #print(f"S_Dbw Index: {s_dbw_index}")

                if db_index < best_score:
                    best_score = db_index
                    best_umap_params = umap_params
                    best_hdbscan_params = hdbscan_params
                    best_topic_model = topic_model
                    best_topics = topics
                    best_probs = probs

    print("Best UMAP Parameters:", best_umap_params)
    print("Best HDBSCAN Parameters:", best_hdbscan_params)
    print("Best Davies-Bouldin Index:", best_score)

    return best_topics, best_probs, best_topic_model


# Define the Tweet text to use
tweet_text = df_data['tweet_text_clean']
# Define the Tweet date to use
tweet_date = df_data['DateTimeStr']

# Define embedding model for clustering
embedding_model_id = 'roberta-base-nli-stsb-mean-tokens'

# Define models and parameters for the NLP pipeline
# Add to the stopwords
stopwords_additions = ['http','https','amp','com','rt']
stopwords_custom = list(stopwords.words('english')) + stopwords_additions

# Load sentence transformer model for the embeddings
embedding_model = SentenceTransformer(embedding_model_id)

# Create tweet embeddings
embeddings = embedding_model.encode(tweet_text, show_progress_bar=False)

# Define the UMAP parameters for dimension reduction
umap_param_grid = {
    'n_neighbors': [20],
    'n_components': [2,3],  # n_components is number of dimensions
    'min_dist': [0.00],
    'metric': ['cosine'] # euclidean, manhattan
}

# Define HDBSCAN parameters for tweet clustering
hdbscan_param_grid = {
    'min_cluster_size': [50],
    'min_samples': [10],
    'metric': ['manhattan','euclidean'], # manhattan
    'cluster_selection_method': ['eom'],
    'prediction_data': [True]
}

vectorizer_model = CountVectorizer(ngram_range=(1, 2)
                                   ,stop_words='english'
                                   )

ctfidf_model = ClassTfidfTransformer(
    # seed_words=domain_specific_terms,
    seed_multiplier=2
    #reduce_frequent_words=True
)

# Create representation model (fine tune key words)
# This will improve the topic representation
# KeyBERTInspired - the standard model
# representation_model_pos - 
# MaximalMarginalRelevance - this will diversify topic representation, this
# limits the number of words that are duplicates/mean the same thing
representation_model_standard = KeyBERTInspired()
representation_model_pos = PartOfSpeech("en_core_web_sm") #top_n_words=30
representation_model_mmr = [KeyBERTInspired(top_n_words=30),
                            MaximalMarginalRelevance(diversity=0.5)]
representation_model_multi = [representation_model_pos,
                              representation_model_mmr]

representation_model = {
   "Main": representation_model_standard,
   "AspectPOS":  representation_model_pos,
   "AspectMMR":  representation_model_mmr,
   #"AspectMulti":  representation_model_multi
}

representation_model = representation_model_mmr

# Get best topic model for the parameters
best_topics, best_probs, best_topic_model = fine_tune_bertopic_model(umap_param_grid,
                                                                     hdbscan_param_grid)

# Train the model using the best BERTopic model
topic_model = best_topic_model
topics = best_topics
probs = best_probs

# Save the model
file_address = os.path.join(models_dir, r'fine-tuned-model-v1')
embedding_model_save = "fine-tuned-model"
topic_model.save(file_address, serialization="pytorch", save_ctfidf=True, save_embedding_model=embedding_model_save,embedd)

# Load the model
loaded_model = BERTopic.load(file_address,embedding_model=embedding_model_id)

####################################################
# Cluster and Topic visualisations
####################################################

# Use this to compare topic representation models, this outputs the topic
# words for each topic
df_topic_model = topic_model.get_topic_info()
# A vector representation derived from the embeddings of sentences that make up that topic
df_topic_model_2 = topic_model.topic_embeddings_  # Embedding vectors for topics

# Use this to see the topic words over time
topics_over_time = topic_model.topics_over_time(tweet_text,
                                                tweet_date,
                                                global_tuning=True,
                                                evolution_tuning=True,
                                                #nr_bins=10
                                                )  # Total dates
# Graph
# The vector model ised to generate the words is the same one used in the BERTopic model
fig = topic_model.visualize_topics_over_time(topics_over_time)
fig.write_html(os.path.join(visualisations_dir, r'topic_over_time_2.html'))

# Scatter graph
# https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html
fig = topic_model.visualize_documents(tweet_text, embeddings=embeddings)
fig.write_html(os.path.join(visualisations_dir, r'scatter.html'))  # Use reduced_embeddings=reduced_embeddings to visualise without dimenstion reduction

# Topic representation
# Bar chart
fig = topic_model.visualize_barchart()
fig.write_html(os.path.join(visualisations_dir, r'bar_1.html'))


# Graph
# Term rank
fig = topic_model.visualize_topics_over_time(topics_over_time)
fig.write_html(os.path.join(visualisations_dir, r'term_ranl.html'))




# Topic descriptions
topic_model.get_topic(0)

# Hierarchy clustering
fig = topic_model.visualize_hierarchy()
fig.write_html(os.path.join(visualisations_dir, r'hierarchy.html'))

# Hierarchy clustering with some more detail
hierarchical_topics = topic_model.hierarchical_topics(tweet_text)
fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(os.path.join(visualisations_dir, r'hierarchy_topics.html'))

# Hierarchy text based
tree = topic_model.get_topic_tree(hierarchical_topics)
print(tree)

# Scatter but with more details
hierarchical_topics = topic_model.hierarchical_topics(tweet_text)
fig = topic_model.visualize_hierarchical_documents(tweet_text, hierarchical_topics, embeddings=embeddings)
fig.write_html(os.path.join(visualisations_dir, r'scatter_2.html'))

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
fig = topic_model.visualize_hierarchical_documents(tweet_text, hierarchical_topics, reduced_embeddings=reduced_embeddings)
fig.write_html(os.path.join(visualisations_dir, r'scatter_2.html'))



topic_test = topic_model.get_topic_info().head(7).set_index('Topic')[['Count', 'Name', 'Representation']]
topic_test

topic_distr, _ = topic_model.approximate_distribution(tweet_text)
topic_model.visualize_distribution(topic_distr[1])

# Heatmap
topic_model.visualize_heatmap()

# Decline
topic_model.visualize_term_rank()

# Topics per class
classes = [data["target_names"][i] for i in data["target"]]
topics_per_class = topic_model.topics_per_class(tweet_text, classes=classes)
fig = topic_model.visualize_topics_per_class(topics_per_class)
fig.write_html(os.path.join(visualisations_dir, r'topic_per_class.html'))

# Token distribution
#pip install jinja2
# Calculate the topic distributions on a token-level
topic_distr, topic_token_distr = topic_model.approximate_distribution(tweet_text, calculate_tokens=True)

# Visualize the token-level distributions
df = topic_model.visualize_approximate_distribution(tweet_text[1], topic_token_distr[1])
print(df)

####################################################
# Assign Clusters and Topics to the Tweets Data

# The cluster that a tweet is most likelty to be in
# The sub-cluster (evolving topic) that the tweet is most likely to be in

# Useful code: 
# topic_model.probabilities_ # 1 = heart of the cluster, this is not the spatial centroid notion of core
# topic_model.get_document_info(tweet_text)['Topic'] another way to get topic

# Notes: 
# probs does not match probs_final (below), appears to be an error with BERTopic
# Confirm that topics and topic_model.topics_ are the same
# (unique, counts) = np.unique(topics, return_counts=True) # topic_model.topics_
# {x:y for x,y in zip(unique, counts)}
####################################################

# Bring in data that will be used to rank the tweets for selection: cluster, sub-cluster
# Probability for final topic for each tweet

# For each tweet, add the topic and topic probability
df_topics = pd.DataFrame(topics)
probs_final = topic_model.get_document_info(tweet_text)['Probability']
df_probs = pd.DataFrame(probs_final)
df_topics_probs = pd.concat([df_topics,df_probs],axis = 1)
df_topics_probs.columns = ['Topic', 'Probability']
df_data_full = pd.concat([df_data,df_topics_probs],axis = 1)

# For each tweet, add the sub-topic and sub-topic probability
# The tweet embeddings are: embeddings

topics_over_time = topic_model.topics_over_time(tweet_text,
                                                tweet_date,
                                                global_tuning=True,
                                                evolution_tuning=True,
                                                #nr_bins=10
                                                )  # Total dates

topics_over_time['keywords'] = topics_over_time['Words'].map(lambda x: x.replace(',',''))
#topics_over_time['topic_vector'] = topics_over_time['keywords'].apply(lambda x: embedding_model.encode(x, show_progress_bar=False))

# Ensure dates are in datetime format (optional, if dates are strings)
df_data_full['DateTimeStr'] = pd.to_datetime(df_data_full['DateTimeStr'])
topics_over_time['Timestamp'] = pd.to_datetime(topics_over_time['Timestamp'])

# Merge on two conditions: 'id' and 'date'
df_data_tweet_keywords = pd.merge(df_data_full,
                                  topics_over_time,
                                  left_on=['DateTimeStr', 'Topic'],
                                  right_on=['Timestamp', 'Topic'],
                                  how='left')

# Topic embeddings
topic_keywords = df_data_tweet_keywords['keywords']
topic_embeddings = embedding_model.encode(topic_keywords, convert_to_tensor=True)

# Sentence embeddings
original_tweets = df_data_tweet_keywords['tweet_text_clean']
embeddings.shape

# Cosine similarity
cosine_similarities = cosine_similarity(embeddings, topic_embeddings)
cosine_similarities.shape

diagonal_similarity = np.diagonal(cosine_similarities)
diagonal_similarity.shape
type(diagonal_similarity)

df_data_tweet_keywords['cosine_similarity'] = diagonal_similarity.tolist()

# Tidy up
df_data_tweet_keywords = df_data_tweet_keywords[['tweet_id','tweet_text_clean','DateTimeStr','Topic','Probability','keywords','Frequency','cosine_similarity']]
df_data_tweet_keywords = df_data_tweet_keywords[df_data_tweet_keywords['Topic'] >= 0]

# Get the top x tweets with the maximum topic probability and cosine similarity, 
# for each date and topic


# Drop outliers
df_data_full = df_data_full.drop(df_data_full[df_data_full['Topic'] == -1].index)




# We now have the embeddings for topics, this can be used to compare against tweets, joining also on cluster and date
# We could ignore the date and just match on embeddings, but makes sense to look at embeddings, date and topic
# Doing it this way is good because the ebedding model captures the semantic meaning of the topics on that date and for
# that topic, it also works regardless of the number of keywords or format, as this is a single representation of the keywords

# BERTopic
# Some interesting functions that may be useful in the future
#topic_model.topic_embeddings_
#topic_model.c_tf_idf_

# There are many different representations that can be obtained from BERTopic on the theme of topics, from 
# keywords, phrases, summaries and custome labels, see: https://maartengr.github.io/BERTopic/getting_started/multiaspect/multiaspect.html
# A SOTD process will be built to get more topic representation, this can be used to judge distance from original tweet to the topic keywords, 
# this may be better than just using the probability number, all representations get be obtained from: topic_model.get_topic_info() and topic_model.topic_aspects_

# BERTopic is able to easily identity the representative documents for each cluster/topic using topic_model.get_representative_docs(topic = 0)
# Not going to use as we want more than 3, also we would want representative topics for each date

####################################################
# Summarise the Text

# Tweets are ranked and filtered on the highest probability that the tweet
# belongs to a cluster for each date. We reduce to three tweets, meaning that
# each day will contain three tweets that are most closely associated to that
# cluster. These three tweets will then be summarised for each day.
####################################################

# Columns to group by
cols_group = ['DateTimeStr', 'Topic']
# Columns to order by, need to add in others, such as text coherance, closest to cluster centre, etc
cols_order = ['DateTimeStr', 'Topic', 'Probability', 'cosine_similarity']

# Order tweets
df_data_tweet_keywords = df_data_tweet_keywords.sort_values(by = cols_order, ascending = [True, True, False, False])
# Add tweet rank, 1 being the best tweet to include for summarising
#df_data_tweet_keywords['Rank'] = df_data_tweet_keywords.groupby(cols_group).cumcount() + 1  # + 1 to start at 1 not 0
df_data_tweet_keywords['Rank'] = df_data_tweet_keywords.groupby(cols_group)['Topic'].rank(ascending=False, method='first')
df_data_tweet_keywords['Rank'] = df_data_tweet_keywords.groupby(cols_group)['Probability'].rank(ascending=False, method='first')

# Keep top 3 tweets
df_data_tweet_keywords = df_data_tweet_keywords.drop(df_data_tweet_keywords[df_data_tweet_keywords['Rank'] > 3].index)

# Rank the results by Date, Cluster and Similarity, the top N will then be taken for summarisation
#https://stackoverflow.com/questions/44368537/pandas-groupby-with-delimiter-join
df_data_summary = df_data_tweet_keywords.groupby(['DateTimeStr', 'Topic'], as_index = False).agg({'tweet_text_clean': '. '.join})

#https://medium.com/artificialis/t5-for-text-summarization-in-7-lines-of-code-b665c9e40771
model_id = 't5-small'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelWithLMHead.from_pretrained(model_id, return_dict=True)


# Function that summarises text
def summarise_text(text):
    #text = ("It is my birthday next week, it will be a good day. Christmas it also soon. I hope that I get a good grade on my dissertation.")
    inputs = tokenizer.encode("sumarize" + text, 
                              return_tensors='pt', 
                              max_length=512, 
                              truncation=True)
    output = model.generate(inputs, 
                            min_length=10, #20
                            max_length=100, 
                            length_penalty=5, 
                            #num_beams=5 #https://www.reddit.com/r/LanguageTechnology/comments/igz9ul/what_are_beams_and_how_does_their_number_affect_a/
                            early_stopping=True
                            )
    summary = tokenizer.decode(output[0], 
                               skip_special_tokens=True)
    return summary

# Test
summarise_text("")

# Both of the below does the same, summarises the tweets
#df_data_full_text['TweetSummary'] = df_data_full_text['tweet_text_clean'].apply(summarise_text)
df_data_summary['tweet_summary'] = df_data_summary.apply(lambda row: summarise_text(row['tweet_text_clean']),axis=1)








####################################################
# Fine-Tuning the T5 Model
####################################################
"""
From: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter/data
twcs.csv
Content: 
The dataset is a CSV, where each row is a tweet. The different columns are described below. Every conversation included has at least one request 
from a consumer and at least one response from a company. Which user IDs are company user IDs can be calculated using the inbound field. 

tweet_id - A unique, anonymized ID for the Tweet. Referenced by response_tweet_id and in_response_to_tweet_id.
author_id - A unique, anonymized user ID. @s in the dataset have been replaced with their associated anonymized user ID.
inbound - Whether the tweet is "inbound" to a company doing customer support on Twitter. This feature is useful when re-organizing data for training conversational models.
created_at - Date and time when the tweet was sent.
text - Tweet content. Sensitive information like phone numbers and email addresses are replaced with mask values like __email__.
response_tweet_id - IDs of tweets that are responses to this tweet, comma-separated.
in_response_to_tweet_id - ID of the tweet this tweet is in response to, if any.
"""
TWCS_FILE_PATH = r'C:\Users\mail\OneDrive\University\04 Bristol\07 Project\Code\source_third_party\thoughtvector_customer-support-on-twitter\twcs\twcs.csv'
#TWCS_FILE_PATH = r'C:\Users\mail\OneDrive\University\04 Bristol\07 Project\Code\thoughtvector_customer-support-on-twitter\sample.csv'
TWEET_SUMM_FILE_PATH = r'C:\Users\mail\OneDrive\University\04 Bristol\07 Project\Code\source_third_party\Tweetsumm\tweet_sum_data_files\final_test_tweetsum.jsonl'

#final_test_tweetsum.jsonl
#final_train_tweetsum.jsonl
#final_valid_tweetsum.jsonl

"""
get_dialog_with_summaries
gets a list of TweetSumm entries (lines in the TweetSumm files) and returns a list of corresponding DialogWithSummaries objects. 
These objects allow to access to the readable conversation associated with their readable human generated summaries. 
"""
processor = tsp.TweetSumProcessor(TWCS_FILE_PATH)
json_format = []
string_format = []
dict_format = []
with open(TWEET_SUMM_FILE_PATH, encoding='utf-8') as f:
    dialog_with_summaries = processor.get_dialog_with_summaries(f.readlines())
    for dialog_with_summary in dialog_with_summaries:
        #json_format = dialog_with_summary.get_json()
        #string_format = str(dialog_with_summary)
        dict_format.append(json.loads(dialog_with_summary.get_json()))

"""
dict_format now contains conversation data between the agent and customer, via Tweets
This is in a certain format: 

dict_format
    dialog: dialog_id, turns
    turns: is_agent, sentences
        sentences: Customer sentence is here
    summaries: 
        extractive_summaries: is_agent, sentences
            sentences: Extracted sentence of importance is here
        abstractive_summaries: sentences
            sentences: Abstractive summaries is here

{
  "conversation_id": "bbde6d8ec7c39c4551da1ff6024f997b",
  "tweet_ids_sentence_offset": [
    {
      "tweet_id": 2263653,
      "sentence_offsets": [
        "[0, 80]",
        "[82, 95]"
      ]
    },
    {
      "tweet_id": 2263654,
      "sentence_offset": "[15, 114]"
    },
  ],
  "annotations": [
    {
      "extractive": [
        {
          "tweet_id": 2263653,
          "sentence_offset": "[0, 80]"
        },
        {
          "tweet_id": 2263654,
          "sentence_offset": "[15, 114]"
        },
      ],
      "abstractive": [
        "Customer is complaining that the watchlist is not updated with new episodes from past two days.",
        "Agent informed that the team is working hard to investigate to show new episodes on page."
      ]
    },
    Plus another two extractive and abstractive summaries, as the conversations are summarised three times
  ]
}
"""

# Look into the data to understand how it can be unpacked
dict_format[1]  # First dialog with all sentences, extractive summaries and abstractive summaries
dict_format[1]['dialog']  # Just the dialog between the agent and the customer
dict_format[1]['dialog']['turns']  # Same as above, but just without the dialog_id
dict_format[1]['dialog']['turns'][1]  # Sentences and is_agent
dict_format[1]['dialog']['turns'][1]['sentences']  # Sentences and without is_agent
dict_format[1]['summaries']  # Extractive and abstractive summaries
dict_format[1]['summaries']['extractive_summaries'][0][0]['sentences']
dict_format[1]['summaries']['abstractive_summaries'][0]

"""
We want the data in a format that is more suitable for training the model, at the very least we need
the customer and agent conversation, and also the abstractive conversation. 

Other columns can be included, this gives more options for training the model, e.g. results
may improve if we train on customer tweets and annotated customer tweets, as this could
be more reflective of the tweets that require summarisation, i.e. where someone is asking for help

Desired format:
1) Customer and agents conversation (essential)
2) Customer Conversation
3) Agent Conversation
4) Extractive customer and agent conversation
5) Extractive customer conversation
6) Extractive agent conversation
7) Abstractive customer and agent conversation (esssential)
8) Abstractive customer conversation
9) Abstractive agent conversation
"""

dict_format[1]['dialog']['turns'][1]['sentences'] # 1) Customer and agents conversation
dict_format[1]['summaries']['extractive_summaries'][0][0]['sentences'] # 4) Extractive customer and agent conversation
dict_format[1]['summaries']['abstractive_summaries'][0][0] # 7) Abstractive customer and agent conversation

"""
Notes on the below procedures: 
    Use the below on groupby so that the index is on dialog_id, this is needs as there are some blanks in the data
    and we need to ensure that the data is linked on dialog_id and not another index
    as_index = True
    
    Use the below to catch errors, there is at least one blank list, so this will be skipped over
    Try Except
"""
def unpack_conversation_sentences():
    list_sentences = []
    for x in range(len(dict_format)):
        try:
            data_id = str(dict_format[x]['dialog']['dialog_id'])
            data_sentences = dict_format[x]['dialog']['turns']
            df_data_sentences = pd.DataFrame(data_sentences)
            df_data_sentences['sentences'] = df_data_sentences['sentences'].str.join(' ') # Value in cell is a list so change to string
            df_data_sentences['dialog_id'] = data_id
            # All conversations
            df_data_sentences_all = df_data_sentences.groupby(['dialog_id'], as_index = True).agg({'sentences': '.'.join}) # We now have a single line for each conversation
            df_data_sentences_all = df_data_sentences_all.rename(columns={'sentences': 'conversation'}) # Rename column names
            # Not agent conversations
            df_data_sentences_not_agent = df_data_sentences.query("is_agent == False").groupby(['dialog_id'], as_index = True).agg({'sentences': '.'.join}) # We now have a single line for each conversation
            df_data_sentences_not_agent = df_data_sentences_not_agent.rename(columns={'sentences': 'conversation_customer'}) # Rename column names
            # Agent conversations
            df_data_sentences_agent = df_data_sentences.query("is_agent == True").groupby(['dialog_id'], as_index = True).agg({'sentences': '.'.join}) # We now have a single line for each conversation
            df_data_sentences_agent = df_data_sentences_agent.rename(columns={'sentences': 'conversation_agent'}) # Rename column names
    
            df_data_sentences_all_2 = pd.merge(df_data_sentences_all,df_data_sentences_not_agent, how='inner', left_index=True, right_index=True)
            df_data_sentences_all_2 = pd.merge(df_data_sentences_all_2,df_data_sentences_agent, how='inner', left_index=True, right_index=True)
            # Put in a list
            list_sentences.append(df_data_sentences_all_2)
        except IndexError:
                pass
                continue
    df_conversation_sentences = pd.concat(list_sentences)
    return df_conversation_sentences

def unpack_conversation_extractive():
    list_sentences = []
    for x in range(len(dict_format)):
        try:
            data_id = str(dict_format[x]['dialog']['dialog_id'])
            data_sentences = dict_format[x]['summaries']['extractive_summaries'][0]
            df_data_sentences = pd.DataFrame(data_sentences)
            df_data_sentences['sentences'] = df_data_sentences['sentences'].str.join(' ') # Value in cell is a list so change to string
            df_data_sentences['dialog_id'] = data_id
            df_data_sentences_all = df_data_sentences.groupby(['dialog_id'], as_index = True).agg({'sentences': '.'.join}) # We now have a single line for each conversation
            df_data_sentences_all = df_data_sentences_all.rename(columns={'sentences': 'conversation_extractive'}) # Rename column names
            # Put in a list
            list_sentences.append(df_data_sentences_all)
        except IndexError:
                pass
                continue
    df_conversation_extractive = pd.concat(list_sentences)
    return df_conversation_extractive

def unpack_conversation_abstractive():
    list_sentences = []
    for x in range(len(dict_format)):
        try:
            data_id = str(dict_format[x]['dialog']['dialog_id'])
            # Get conversation data
            data_sentences = dict_format[x]['summaries']['abstractive_summaries'][0]  # Customer and agent abstractive summary
            data_sentences_customer = dict_format[x]['summaries']['abstractive_summaries'][0][0]  # Customer abstractive summary
            data_sentences_agent = dict_format[x]['summaries']['abstractive_summaries'][0][1]  # Agent abstractive summary
            # Put data into dataframe
            df_data_sentences = pd.DataFrame(data_sentences)
            # Rename columns
            df_data_sentences = df_data_sentences.rename(columns={0: 'sentences'})
            df_data_sentences['dialog_id'] = data_id
            # Aggregate data so we have a single line for each conversation
            df_data_sentences_all = df_data_sentences.groupby(['dialog_id'], as_index = True).agg({'sentences': '.'.join})
            df_data_sentences_all['conversation_abstractive_customer'] = data_sentences_customer
            df_data_sentences_all['conversation_abstractive_agent'] = data_sentences_agent
            # Rename column names
            df_data_sentences_all = df_data_sentences_all.rename(columns={'sentences': 'conversation_abstractive'})
            # Put in a list
            list_sentences.append(df_data_sentences_all)
        except IndexError:
                pass
                continue
    df_conversation_abstractive = pd.concat(list_sentences)
    return df_conversation_abstractive


# Run the unpacking procedures
df_conversation_sentences = unpack_conversation_sentences()
df_conversation_extractive = unpack_conversation_extractive()
df_conversation_abstractive = unpack_conversation_abstractive()

# Final dataset with all columns togther in one DataFrame
df_conversation_final = pd.merge(df_conversation_sentences,
                                 df_conversation_extractive,
                                 how='inner',
                                 left_index=True,
                                 right_index=True
                                 )
df_conversation_final = pd.merge(df_conversation_final,
                                 df_conversation_abstractive,
                                 how='inner',
                                 left_index=True,
                                 right_index=True
                                 )

####################################################
# Fine-Tuning Data Analysis and Cleanse
####################################################
Longest
Shortest
Average words
Basically do the same as with the original Tweets we are dealing with

def print_person(is_agent, sentences):
    sentences_clean = ' '.join(sentences)
    print(f"The agent is {is_agent} and the sentence is {sentences_clean} end")

dict_test_1 = dict_format[1]['dialog']['turns'][0]
is_agent, sentences = dict_test_1.items()
sentence_clean = ' '.join(sentences[1])
print_person(**dict_test_1)

dict_test_1 = dict_format[1]['dialog']['turns']
df = pd.DataFrame(dict_test_1)

df_conversation_final2 = pd.concat([df_conversation_final['conversation'],df_conversation_final['conversation_abstractive']],axis = 1) # Get the columns needed for training, this is conversation and the abstractive conversation
train_data, val_data = train_test_split(df_conversation_final2, test_size=0.1) # Split into training and test data

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# Pre-trained Models
model_id = "t5-small"
model_id = 'philschmid/flan-t5-base-samsum'
model_id = 'KonradSzafer/flan-t5-small-samsum'
model_id = 'epinnock/flan-t5-small-samsum'


# AutoModelWithLMHead # Now depreciated as too generic. For autoregressive tasks where the model predicts the next token based on previous tokens (e.g., GPT, GPT-2).
# AutoModelForCausalLM for causal language modeling tasks (e.g., GPT-style models).
# AutoModelForSeq2SeqLM for sequence-to-sequence tasks (e.g., T5, BART). For tasks involving input-output mappings, like summarization, translation, or T5-style tasks. 
# Can be used to load any seq2seq (or encoder-decoder) model that has a language modeling (LM) head on top. So when you do AutoModelForSeq2SeqLM.from_pretrained(‘t5-base’), 
# it will actually load a T5ForConditionalGeneration for you behind the scenes. 
# T5ForConditionalGeneration tailored for tasks where an input sequence is mapped to an output sequence, making it ideal for tasks such as: translation, summarization, Q&A
# T5ForConditionalGeneration is a specific implementation of AutoModelForSeq2SeqLM, optimized for the T5 architecture.

tokenizer = AutoTokenizer.from_pretrained(model_id) # Tokeniser for the model
model = T5ForConditionalGeneration.from_pretrained(model_id) # Loads the pre-trained T5 model for conditional text generation

# Function to convert text data into model inputs and targets
def preprocess_data(examples):
    inputs = ["summarize: " + doc for doc in examples["conversation"]]
    model_inputs = tokenizer(inputs,
                             max_length=512,
                             truncation=True,
                             padding=True
                             )

    # Tokenize summaries (labels)
    with tokenizer.as_target_tokenizer():
        # targets = [summary for summary in examples['summaries'] # could put the targets here as above and then replace examples["conversation_abstractive"]
        labels = tokenizer(examples["conversation_abstractive"],
                           max_length=150,
                           truncation=True,
                           padding=True
                           )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply function to the whole dataset, the conversation and abstractive conversation will then appear as numbers and not words
tokenized_train = train_dataset.map(preprocess_data, batched=True)
tokenized_val = val_dataset.map(preprocess_data, batched=True)

# Step 4: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    report_to="none"
)

# Step 5: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer
)

# Step 6: Train the Model
trainer.train()

# Step 7: Save the Model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Step 8: Evaluate on a test sample
def summarize(text):
    input_ids = tokenizer("summarize: " + text,
                          return_tensors="pt",
                          truncation=True,
                          max_length=512,
                          padding='max_length'
                          ).input_ids
    output = model.generate(input_ids,
                            max_length=150,
                            num_beams=4,
                            early_stopping=True
                            )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example test
test_text = "Great tale from : Canada firefighters battled blaze as own homes burned down.SK Wildfire update: There are  fires currently burning. The fire risk remains severe..I dont understand this. , Canadians internally displaced and homeless, fires not yet under control. cdnpoli"
print(summarize(test_text))

df_data_summary['tweet_summary_tuned'] = df_data_summary.apply(lambda row: summarize(row['tweet_text_clean']),axis=1)










model_id = 'epinnock/flan-t5-small-samsum'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelWithLMHead.from_pretrained(model_id, return_dict=True)


# Function that summarises text
def summarise_text(text):
    #text = ("It is my birthday next week, it will be a good day. Christmas it also soon. I hope that I get a good grade on my dissertation.")
    inputs = tokenizer.encode("sumarize" + text, 
                              return_tensors='pt', 
                              max_length=512, 
                              truncation=True)
    output = model.generate(inputs, 
                            min_length=10, #20
                            max_length=100, 
                            length_penalty=5, 
                            #num_beams=5 #https://www.reddit.com/r/LanguageTechnology/comments/igz9ul/what_are_beams_and_how_does_their_number_affect_a/
                            early_stopping=True
                            )
    summary = tokenizer.decode(output[0], 
                               skip_special_tokens=True)
    return summary

# Test
summarise_text("Fort McMurray Canadian wildfire has grown to within a few kilometers of tar sands production areas..To get the latest information on the Fort McMurray wildfire situation and emergency response, follow  ym.Wow &gt;&gt;These Maps Show The Insane Size Of The Fort McMurray Wildfire")

# Both of the below does the same, summarises the tweets
#df_data_full_text['TweetSummary'] = df_data_full_text['tweet_text_clean'].apply(summarise_text)
df_data_full_text['tweet_summary_samsum'] = df_data_full_text.apply(lambda row: summarise_text(row['tweet_text_clean']),axis=1)






















###############################SAMSUM fine tuning#################################

#https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/flan-t5-samsum-summarization.ipynb

dataset_id = "samsum"
from datasets import load_dataset
dataset = load_dataset(dataset_id)
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

from random import randrange        

sample = dataset['train'][randrange(len(dataset["train"]))]
print(f"dialogue: \n{sample['dialogue']}\n---------------")
print(f"summary: \n{sample['summary']}\n---------------")

model_id="google/flan-t5-base"

# Load tokenizer of FLAN-t5-base
tokenizer = AutoTokenizer.from_pretrained(model_id)

from datasets import concatenate_datasets

# The maximum total input sequence length after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["dialogue"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

from transformers import AutoModelForSeq2SeqLM
# huggingface hub model id
model_id="google/flan-t5-base"

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

# Metric
metric = evaluate.load("rouge")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Hugging Face repository id
repository_id = f"{model_id.split('/')[1]}-{dataset_id}"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=5,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # metric_for_best_model="overall_f1",
    # push to hub parameters
    report_to="tensorboard",
    push_to_hub=False,
    hub_strategy="every_save",
    hub_model_id=repository_id,
    hub_token=HfFolder.get_token(),
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

# Plots
# https://www.williampnicholson.com/topic-modelling/
# https://medium.com/@haytham99cheikhrouhou/clustering-arxiv-ml-articles-using-kmeans-bertopic-to-generate-mind-maps-74beb27fc6a8

# Method 1
# Get topic probabilities into data
# https://stackoverflow.com/questions/73768683/how-to-get-topic-probs-matrix-in-bertopic-modeling

# Hypertuning of UMAP and HDBSCAN
# https://github.com/awslabs/amazon-denseclus/blob/main/notebooks/03_ValidationForUMAP.ipynb
