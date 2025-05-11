import io
import math

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from spacy.lang.hu import Hungarian
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import gensim
import pandas as pd


# Load the article data
app_dir = (Path(__file__).parent).parent
print(app_dir)
with open(app_dir / 'data/article_data_dates.json', encoding="utf8") as f:
    article_data = json.load(f)

# Initialize Hungarian NLP
nlp = Hungarian()
hu = spacy.load('hu_core_news_lg')
stopwords = hu.Defaults.stop_words

def preprocess(word_list):
    result = []
    for token in word_list:
        token=token.lemma_
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stopwords:
            result.append(token)
    return result


def fixed_color_func(color):
    def color_func(*args, **kwargs):
        return color
    return color_func


# Streamlit UI
st.title("Telex Article Topic Modeling")

# Date range filter
min_date = datetime.strptime(min(article['date'] for article in article_data), "%Y-%m-%d").date()
max_date = datetime.strptime(max(article['date'] for article in article_data), "%Y-%m-%d").date()

# Date range filter
start_date, end_date = min_date, max_date


date_range = st.date_input("Select Date Range", [min_date, max_date])
if len(date_range) != 2:
    st.warning("Please select both a start and an end date.")
else:
    start_date, end_date = date_range


# Tag selection
all_tags = ["BELFÖLD", "KÜLFÖLD", "GAZDASÁG", "AFTER", "TECHTUD", "SPORT", "ZACC", "KULT", "ÉLETMÓD"]
select_all = st.checkbox("Select All Tags")

if select_all:
    selected_tags = all_tags  # Automatically select all tags
else:
    selected_tags = st.multiselect("Select Tags", all_tags)

# Filter articles based on date and tags
filtered_articles = [
    article for article in article_data
    if start_date <= datetime.strptime(article['date'], "%Y-%m-%d").date() <= end_date
       and any(tag in article.get('tags', []) for tag in selected_tags)
]

if len(filtered_articles) > 0:
    st.subheader(f"Number of articles selected: {len(filtered_articles)}")

# Run button
if st.button("Run Topic Modeling"):
    if len(filtered_articles) < 10:
        st.warning("At least 10 articles must be selected to run topic modeling.")
    else:
        progress_bar = st.progress(0)  # Initialize progress bar
        progress_step = 1 / (len(filtered_articles) + 3)  # Calculate step size

        text_placeholder = st.empty()
        text_placeholder.write("Preprocessing...")
        topic_docs = defaultdict(list)
        for idx, doc in enumerate(filtered_articles):
            if doc['tags']:  # Ensuring tag exists
                text = preprocess(hu(doc['article_text']))
                for tag in doc['tags']:
                    topic_docs[tag].append(text)
            progress_bar.progress(min((idx + 1) * progress_step, 1.0))  # Update progress

        # Run LDA for each topic
        lda_models = {}
        dictionaries = {}

        # Determine the number of subtopics
        num_articles = len(filtered_articles)
        num_subtopics = min(6, max(2, num_articles // 10))  # Min 2 subtopics, max 6 or 10% of articles

        for idx, (topic, docs) in enumerate(topic_docs.items()):
            if topic not in selected_tags:
                continue
            if len(docs) < 2:
                continue  # Not enough data for LDA

            # Preparing corpus
            dictionary = Dictionary(docs)
            corpus = [dictionary.doc2bow(doc) for doc in docs]

            # Training LDA
            lda = LdaModel(corpus, num_topics=num_subtopics, id2word=dictionary, passes=10, random_state=42)

            # Storing model and dictionary for later classification
            lda_models[topic] = lda
            dictionaries[topic] = dictionary
            progress_bar.progress(min((len(filtered_articles) + idx + 1) * progress_step, 1.0))  # Update progress

        progress_bar.empty()
        text_placeholder.empty()

        # Step 3: Subtopic classification, with counting articles and aggregating facebook activity
        progress_bar.progress(0)  # Initialize classification progress bar
        classification_step = 1 / len(filtered_articles)  # Calculate step size for classification
        text_placeholder.write("Subtopic classification...")

        subtopic_counts = defaultdict(lambda: Counter())
        subtopic_fb_activity = defaultdict(lambda: defaultdict(int))

        for idx, article in enumerate(filtered_articles):
            if article['tags']:
                tokens = preprocess(hu(article['article_text']))
                for topic in article['tags']:
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
            progress_bar.progress(min((idx + 1) * classification_step, 1.0))  # Update classification progress

        progress_bar.empty()
        text_placeholder.empty()
        st.success("Modeling completed!")

        # Defining a list of distinct colors
        colors = ['#FFA500', '#A9A9A9', '#FF6347', '#1E90FF', '#FF66CC', '#3CB371']

        # Step 4: Displaying word clouds with article classification
        for topic, lda in lda_models.items():
            cols = min(3, num_subtopics)
            rows = math.ceil(num_subtopics / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
            fig.suptitle(f"LDA Word Clouds for '{topic}'", fontsize=18, y=0.98)
            axes = axes.flatten()

            sorted_subtopics = sorted(range(num_subtopics), key=lambda i: subtopic_fb_activity[topic][i], reverse=True)

            for idx, i in enumerate(sorted_subtopics):
                terms = lda.show_topic(i, topn=30)
                word_freq = {term: weight for term, weight in terms}
                color_func = fixed_color_func(colors[idx % len(colors)])
                wc = WordCloud(width=500, height=400, background_color='white', color_func=color_func).generate_from_frequencies(word_freq)

                axes[idx].imshow(wc, interpolation='bilinear')
                title_text = (rf"$\bf{{Subtopic\ {i+1}}}$" f"\n(Facebook activity: {subtopic_fb_activity[topic][i]})" f"\n({subtopic_counts[topic][i]} articles)")
                axes[idx].set_title(title_text, fontsize=12, fontweight='normal', multialignment='center')
                axes[idx].axis('off')

            if f"{topic}_plot" not in st.session_state:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                st.session_state[f"{topic}_plot"] = buf

            # Provide a download button for the plot
            st.download_button(
                label=f"Download {topic}",
                data=st.session_state[f"{topic}_plot"],
                file_name=f"{topic}_wordcloud.png",
                mime="image/png"
            )


            # Display the combined image in Streamlit
            st.pyplot(fig)
            plt.close(fig)

# Count articles by date
filtered_dates = [
    datetime.strptime(article['date'], "%Y-%m-%d").date()
    for article in article_data
    if start_date <= datetime.strptime(article['date'], "%Y-%m-%d").date() <= end_date
       and any(tag in article.get('tags', []) for tag in selected_tags)
]


# Show visualizations about selected articles
if len(filtered_articles) > 0:
    date_counts = Counter(filtered_dates)

    # Convert to DataFrame for visualization
    date_df = pd.DataFrame.from_dict(date_counts, orient='index', columns=['Article Count'])
    date_df.index.name = 'Date'
    date_df = date_df.sort_index()

    # Display the chart
    st.subheader("Number of Articles by Date")
    st.bar_chart(date_df)

    tag_counts = Counter(tag for article in filtered_articles for tag in article.get('tags', []) if tag in selected_tags)
    # Convert to DataFrame and sort by count
    tag_df = pd.DataFrame.from_dict(tag_counts, orient='index', columns=['Article Count'])
    tag_df.index.name = 'Tag'
    tag_df = tag_df.sort_values(by='Article Count', ascending=False)

    # Display the bar chart in Streamlit
    st.subheader("Number of Articles per Category")
    st.bar_chart(tag_df)