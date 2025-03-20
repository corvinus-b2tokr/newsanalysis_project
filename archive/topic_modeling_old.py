import gensim
from gensim.corpora import Dictionary
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import spacy
from spacy.lang.hu.stop_words import STOP_WORDS
from spacy.lang.hu import Hungarian
nlp=Hungarian()

hu = spacy.load('hu_core_news_lg')
stopwords = hu.Defaults.stop_words

article_data_test = {'title': 'Változás az iskolákban: érkeznek a gerincvédő székek',
                'author': 'Nagy Bálint',
                'tags': ['BELFÖLD'],
                'facebook_activity': '600',
                'article_text': 'Változás lép életbe az iskolákban július 1-től: új iskolai szék vásárlásakor már csak gerincvédő székeket lehet beszerezni – írja a Magyar Nemzet az Országos Gerincgyógyászat Központ közleménye alapján. Az erről szóló tájékoztatást a Belügyminisztérium és az Oktatási Hivatal elküldte az iskolafenntartóknak és igazgatóknak is. Azt írták, hogy a tanulók egészségének védelme népegészségügyi, pedagógiai és össztársadalmi cél, a sok ülés miatt pedig fontos, hogy a szék segítse a helyes ülést.  A minisztérium ezzel az Országos Gerincgyógyászati Központ kezdeményezését támogatja. Ajánlásuk alapján azt kérik, hogy amikor tanulói székeket vásárolnak az iskoláknak, az Oktatási Hivatal honlapján az oktatási intézmények kötelező felszereléseiről szóló módosított jegyzékben közzétett új méreteknek megfelelő, gerincvédelmet biztosító tanulói székeket szerezzék be. A döntésről a Magyar Nemzetnek Somhegyi Annamária reumatológus azt mondta: ezzel a lépéssel közelebb kerültünk ahhoz, hogy a hazánkat is sújtó porckopásos gerincbetegségeket visszaszorítsuk. „Azért fontos, hogy a gyerekek gerincvédő székeken üljenek, mert így a derekukat a szék támlája meg tudja támasztani, ez pedig az ülés által a porckorongokra gyakorolt többletnyomást csökkenti. Ha mozgásszervi szakember vizsgálja meg az iskolások gerincét, akár 80 százalékuknál is eltérést talál, s ennek túlnyomó részét szintén a tartási rendellenességek képezik” – fogalmazott.'
}

article_data2 = [{'title': 'Változás az iskolákban: érkeznek a gerincvédő székek',
                'author': 'Nagy Bálint',
                'tags': ['BELFÖLD'],
                'facebook_activity': '600',
                'article_text': 'Változás lép életbe az iskolákban július 1-től: új iskolai szék vásárlásakor már csak gerincvédő székeket lehet beszerezni – írja a Magyar Nemzet az Országos Gerincgyógyászat Központ közleménye alapján. Az erről szóló tájékoztatást a Belügyminisztérium és az Oktatási Hivatal elküldte az iskolafenntartóknak és igazgatóknak is. Azt írták, hogy a tanulók egészségének védelme népegészségügyi, pedagógiai és össztársadalmi cél, a sok ülés miatt pedig fontos, hogy a szék segítse a helyes ülést.  A minisztérium ezzel az Országos Gerincgyógyászati Központ kezdeményezését támogatja. Ajánlásuk alapján azt kérik, hogy amikor tanulói székeket vásárolnak az iskoláknak, az Oktatási Hivatal honlapján az oktatási intézmények kötelező felszereléseiről szóló módosított jegyzékben közzétett új méreteknek megfelelő, gerincvédelmet biztosító tanulói székeket szerezzék be. A döntésről a Magyar Nemzetnek Somhegyi Annamária reumatológus azt mondta: ezzel a lépéssel közelebb kerültünk ahhoz, hogy a hazánkat is sújtó porckopásos gerincbetegségeket visszaszorítsuk. „Azért fontos, hogy a gyerekek gerincvédő székeken üljenek, mert így a derekukat a szék támlája meg tudja támasztani, ez pedig az ülés által a porckorongokra gyakorolt többletnyomást csökkenti. Ha mozgásszervi szakember vizsgálja meg az iskolások gerincét, akár 80 százalékuknál is eltérést talál, s ennek túlnyomó részét szintén a tartási rendellenességek képezik” – fogalmazott.'
                },
                {'title': 'Változás az iskolákban: érkeznek a gerincvédő székek',
                'author': 'Nagy Bálint',
                'tags': ['KÜLFÖLD'],
                'facebook_activity': '600',
                'article_text': 'Változás lép életbe az iskolákban július 1-től: új iskolai szék vásárlásakor már csak gerincvédő székeket lehet beszerezni – írja a Magyar Nemzet az Országos Gerincgyógyászat Központ közleménye alapján. Az erről szóló tájékoztatást a Belügyminisztérium és az Oktatási Hivatal elküldte az iskolafenntartóknak és igazgatóknak is. Azt írták, hogy a tanulók egészségének védelme népegészségügyi, pedagógiai és össztársadalmi cél, a sok ülés miatt pedig fontos, hogy a szék segítse a helyes ülést.  A minisztérium ezzel az Országos Gerincgyógyászati Központ kezdeményezését támogatja. Ajánlásuk alapján azt kérik, hogy amikor tanulói székeket vásárolnak az iskoláknak, az Oktatási Hivatal honlapján az oktatási intézmények kötelező felszereléseiről szóló módosított jegyzékben közzétett új méreteknek megfelelő, gerincvédelmet biztosító tanulói székeket szerezzék be. A döntésről a Magyar Nemzetnek Somhegyi Annamária reumatológus azt mondta: ezzel a lépéssel közelebb kerültünk ahhoz, hogy a hazánkat is sújtó porckopásos gerincbetegségeket visszaszorítsuk. „Azért fontos, hogy a gyerekek gerincvédő székeken üljenek, mert így a derekukat a szék támlája meg tudja támasztani, ez pedig az ülés által a porckorongokra gyakorolt többletnyomást csökkenti. Ha mozgásszervi szakember vizsgálja meg az iskolások gerincét, akár 80 százalékuknál is eltérést talál, s ennek túlnyomó részét szintén a tartási rendellenességek képezik” – fogalmazott.'
                },
                {'title': 'Változás az iskolákban: érkeznek a gerincvédő székek',
                'author': 'Nagy Bálint',
                'tags': ['BELFÖLD'],
                'facebook_activity': '600',
                'article_text': 'Változás lép életbe az iskolákban július 1-től: új iskolai szék vásárlásakor már csak gerincvédő székeket lehet beszerezni – írja a Magyar Nemzet az Országos Gerincgyógyászat Központ közleménye alapján. Az erről szóló tájékoztatást a Belügyminisztérium és az Oktatási Hivatal elküldte az iskolafenntartóknak és igazgatóknak is. Azt írták, hogy a tanulók egészségének védelme népegészségügyi, pedagógiai és össztársadalmi cél, a sok ülés miatt pedig fontos, hogy a szék segítse a helyes ülést.  A minisztérium ezzel az Országos Gerincgyógyászati Központ kezdeményezését támogatja. Ajánlásuk alapján azt kérik, hogy amikor tanulói székeket vásárolnak az iskoláknak, az Oktatási Hivatal honlapján az oktatási intézmények kötelező felszereléseiről szóló módosított jegyzékben közzétett új méreteknek megfelelő, gerincvédelmet biztosító tanulói székeket szerezzék be. A döntésről a Magyar Nemzetnek Somhegyi Annamária reumatológus azt mondta: ezzel a lépéssel közelebb kerültünk ahhoz, hogy a hazánkat is sújtó porckopásos gerincbetegségeket visszaszorítsuk. „Azért fontos, hogy a gyerekek gerincvédő székeken üljenek, mert így a derekukat a szék támlája meg tudja támasztani, ez pedig az ülés által a porckorongokra gyakorolt többletnyomást csökkenti. Ha mozgásszervi szakember vizsgálja meg az iskolások gerincét, akár 80 százalékuknál is eltérést talál, s ennek túlnyomó részét szintén a tartási rendellenességek képezik” – fogalmazott.'
                }
]

def preprocess(word_list):
    result = []
    for token in word_list:
        token=token.lemma_
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stopwords:
            result.append(token)
    return result

processed_docs = [preprocess(hu(article_data_test['article_text']))] # [preprocess(elem) for elem in token_docs]
print(processed_docs)

article_data_test['article_text'] = [preprocess(hu(article_data_test['article_text']))]


dictionary = gensim.corpora.Dictionary(processed_docs)
print(dictionary)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(bow_corpus)

word_freq = {}
for doc in bow_corpus:
    for word_id, freq in doc:
        word = dictionary[word_id]
        word_freq[word] = word_freq.get(word, 0) + freq

# Create and display the word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='plasma'
).generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud from Gensim BoW Corpus')
plt.show()

# Step 1: Group tokenized texts by topic (tag)
topic_docs = defaultdict(list)
for doc in article_data2:
    if doc['tags']:  # Ensure tag exists
        topic = doc['tags'][0]  # Use first tag as topic
        text = doc['article_text'].lower().split()  # Simple tokenization
        topic_docs[topic].append(text)

# Step 2: Generate word frequencies for each topic
topic_wordclouds = {}
for topic, tokenized_docs in topic_docs.items():
    dictionary = Dictionary(tokenized_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    word_freq = {}
    for doc in bow_corpus:
        for word_id, freq in doc:
            word = dictionary[word_id]
            word_freq[word] = word_freq.get(word, 0) + freq

    # Generate WordCloud object (but don't show it yet)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(word_freq)

    topic_wordclouds[topic] = wordcloud

# Step 3: Plot all word clouds on one page
num_topics = len(topic_wordclouds)
cols = 2
rows = math.ceil(num_topics / cols)

fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
axes = axes.flatten() if num_topics > 1 else [axes]

for i, (topic, wc) in enumerate(topic_wordclouds.items()):
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].set_title(f'Topic: {topic}', fontsize=16)
    axes[i].axis('off')

# Turn off any empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()