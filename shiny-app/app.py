# Import data from shared.py
from article_import import article_data

import pandas as pd
from shiny.express import input, render, ui
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
from collections import Counter
import gensim
import spacy
from spacy.lang.hu.stop_words import STOP_WORDS
from spacy.lang.hu import Hungarian
nlp=Hungarian()
#nlp.max_length = 2000000

hu = spacy.load('hu_core_news_lg')
stopwords = hu.Defaults.stop_words

article_data = pd.DataFrame(article_data)

all_tags = sorted(set(tag for tags in article_data["tags"] if isinstance(tags, list) for tag in tags))
all_authors = sorted(article_data["author"].dropna().unique().tolist())

ui.page_opts(title="Word Cloud for News Articles")

with ui.sidebar():
    ui.input_select("topic", "Choose Category", choices=list(all_tags), selected=list(all_tags)[0])
    ui.input_select("author", "Choose Author", choices=["-- All --"] + all_authors, selected="-- All --")
    ui.input_slider("max_words", "Maximum Words:", min=10, max=200, value=100)

# Preprocessing function
def preprocess(word_list):
    result = []
    for token in word_list:
        token=token.lemma_
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stopwords:
            result.append(token)
    return result

def get_filtered_text(topic, author):
    filtered_texts = article_data[article_data["tags"].apply(lambda x: topic in x if isinstance(x, list) else False)]
    
    if author != "-- All --":
        filtered_texts = filtered_texts[filtered_texts["author"] == author]

    filtered_texts_list = filtered_texts["article_text"].tolist()
    text = " ".join(filtered_texts_list)
    return preprocess(hu(text)), filtered_texts

@render.ui
def wordcloud():
    topic = input.topic()
    author = input.author()    
    text, filtered_texts = get_filtered_text(topic, author)
    
    if not text:
        return "No words to display."
    
    word_frequencies = Counter(text)
    wordcloud = WordCloud(width=1000, height=500, max_words=input.max_words(), background_color="white", colormap="viridis").generate_from_frequencies(word_frequencies)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
        
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()

    total_facebook_activity = filtered_texts["facebook_activity"].astype('int64').sum()
    article_count = len(filtered_texts)
        
    metrics_style = "font-size: 1.2em; color: #3B4CCA; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: bold; display: inline-block; margin-right: 20px;"

    return ui.div(
        ui.div(
            ui.tags.span(f"Total Facebook Activity: {total_facebook_activity}", style=metrics_style),
            ui.tags.span(f"Number of Articles: {article_count}", style=metrics_style),
            style="margin-bottom: 1em;"
        ),
        ui.tags.img(src="data:image/png;base64," + img_base64, style="width:100%")
    )
