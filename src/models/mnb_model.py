from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .. import config

PARAMS = {"alpha": 0.01}


def build_pipeline() -> Pipeline:
    """TF-IDF + Multinomial Naive Bayes como un solo Pipeline sklearn."""
    tfidf = TfidfVectorizer(
        ngram_range=config.N_GRAMS,
        max_df=0.25,
        stop_words="english",
        max_features=config.MAX_WORDS_NUM,
    )
    clf = MultinomialNB(**PARAMS)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])
