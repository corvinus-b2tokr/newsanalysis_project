#https://github.com/huspacy/huspacy?tab=readme-ov-file

# Example URL
url = 'https://telex.hu/belfold/2025/03/04/oktatas-iskolak-gerincvedo-szek-valtozas-vasarlas-julius-1'

# Fetch and store the article data
#article_data = fetch_article_data(url)

#text_test = article_data["article_text"]
text_test = """Változás lép életbe az iskolákban július 1-től: új iskolai szék vásárlásakor már csak gerincvédő székeket lehet beszerezni – írja a Magyar Nemzet az Országos Gerincgyógyászat Központ közleménye alapján. Az erről szóló tájékoztatást a Belügyminisztérium és az Oktatási Hivatal elküldte az iskolafenntartóknak és igazgatóknak is. Azt írták, hogy a tanulók egészségének védelme népegészségügyi, pedagógiai és össztársadalmi cél, a sok ülés miatt pedig fontos, hogy a szék segítse a helyes ülést.  A minisztérium ezzel az Országos Gerincgyógyászati Központ kezdeményezését támogatja. Ajánlásuk alapján azt kérik, hogy amikor tanulói székeket vásárolnak az iskoláknak, az Oktatási Hivatal honlapján az oktatási intézmények kötelező felszereléseiről szóló módosított jegyzékben közzétett új méreteknek megfelelő, gerincvédelmet biztosító tanulói székeket szerezzék be. A döntésről a Magyar Nemzetnek Somhegyi Annamária reumatológus azt mondta: ezzel a lépéssel közelebb kerültünk ahhoz, hogy a hazánkat is sújtó porckopásos gerincbetegségeket visszaszorítsuk. „Azért fontos, hogy a gyerekek gerincvédő székeken üljenek, mert így a derekukat a szék támlája meg tudja támasztani, ez pedig az ülés által a porckorongokra gyakorolt többletnyomást csökkenti. Ha mozgásszervi szakember vizsgálja meg az iskolások gerincét, akár 80 százalékuknál is eltérést talál, s ennek túlnyomó részét szintén a tartási rendellenességek képezik” – fogalmazott."
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
import gensim
import spacy
from spacy.lang.hu.stop_words import STOP_WORDS
from spacy.lang.hu import Hungarian
nlp=Hungarian()

nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()


hu = spacy.load('hu_core_news_lg')
stopwords = hu.Defaults.stop_words
print(stopwords)

doc = hu(text_test)

#if we need to add extra stopwords
#stopwords.update(["covid19", "coronavirus", "ncov2019", "2019nco", "ncov", "ncov2019", "2019ncov", "covid", "covid-", "covid_", "'s", "'m", "'s", "'ve"])

def preprocess(word_list):
    result = []
    text = " ".join(word_list)
    for token in doc:
        token=token.lemma_
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stopwords:
            result.append(token)

    return result

#token_docs= wordpunct_tokenize(text_test)
#print(token_docs)

processed_docs = [preprocess(text_test)] # [preprocess(elem) for elem in token_docs]

print(processed_docs)