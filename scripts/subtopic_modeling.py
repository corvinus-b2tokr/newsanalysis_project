import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
from pathlib import Path
from collections import defaultdict, Counter
import math
import numpy as np
import spacy
from spacy.lang.hu.stop_words import STOP_WORDS
from spacy.lang.hu import Hungarian
nlp=Hungarian()

hu = spacy.load('hu_core_news_lg')
stopwords = hu.Defaults.stop_words

app_dir = (Path(__file__).parent).parent
f = open(app_dir / 'data/article_data.json', encoding="utf8")
article_data = json.load(f)

# Preprocessing function
def preprocess(word_list):
    result = []
    for token in word_list:
        token=token.lemma_
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stopwords:
            result.append(token)
    return result

# Step 1: Grouping tokenized texts by topic (tag)
topic_docs = defaultdict(list)
for doc in article_data:
    if doc['tags']:  # Ensuring tag exists
        text = preprocess(hu(doc['article_text']))
        for tag in doc['tags']:
            topic_docs[tag].append(text)

# Storing LDA models and dictionaries for classification
lda_models = {}
dictionaries = {}

# Step 2: Running LDA for each topic
for topic, docs in topic_docs.items():
    if len(docs) < 2:
        continue  # Not enough data for LDA

    # Preparing corpus
    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # Training LDA
    num_topics = 6
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
    
    # Storing model and dictionary for later classification
    lda_models[topic] = lda
    dictionaries[topic] = dictionary

# Step 3: Subtopic classification, with counting articles and aggregating facebook activity
subtopic_counts = defaultdict(lambda: Counter())
subtopic_fb_activity = defaultdict(lambda: defaultdict(int))

for article in article_data:
    if article['tags']:
        tokens = preprocess(hu(article['article_text']))
        for topic in article['tags']:  # Looping through all tags
            if topic in lda_models:
                dictionary = dictionaries[topic]
                bow = dictionary.doc2bow(tokens)
                subtopic, _ = max(lda_models[topic][bow], key=lambda x: x[1])
                subtopic_counts[topic][subtopic] += 1
                # Summing up facebook activity (safely handling non-int)
                try:
                    subtopic_fb_activity[topic][subtopic] += int(article['facebook_activity'])
                except ValueError:
                    pass

# Function to generate a fixed color function for a specific color
def fixed_color_func(color):
    def color_func(*args, **kwargs):
        return color
    return color_func

# Defining a list of distinct colors
colors = ['#FFA500', '#A9A9A9', '#FF6347', '#1E90FF', '#FF66CC', '#3CB371']

# Step 4: Displaying word clouds with article classification
for topic, lda in lda_models.items():
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"LDA Word Clouds for '{topic}'", fontsize=18, y=0.98)
    axes = axes.flatten()
    
    sorted_subtopics = sorted(range(6), key=lambda i: subtopic_fb_activity[topic][i], reverse=True)
    
    for idx, i in enumerate(sorted_subtopics):
        terms = lda.show_topic(i, topn=30)
        word_freq = {term: weight for term, weight in terms}
        color_func = fixed_color_func(colors[idx % len(colors)])
        wc = WordCloud(width=500, height=400, background_color='white', color_func=color_func).generate_from_frequencies(word_freq)
        
        axes[idx].imshow(wc, interpolation='bilinear')
        title_text = (rf"$\bf{{Subtopic\ {i+1}}}$" f"\n(Facebook activity: {subtopic_fb_activity[topic][i]})" f"\n({subtopic_counts[topic][i]} articles)")
        axes[idx].set_title(title_text, fontsize=12, fontweight='normal', multialignment='center')
        axes[idx].axis('off')
    
    plt.subplots_adjust(wspace=0.4, hspace=0.4, top=0.85)
    plt.show()