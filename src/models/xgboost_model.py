from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .. import config

PARAMS = {
    "colsample_bytree": 0.7,
    "gamma": 0.2,
    "learning_rate": 0.1,
    "max_depth": 12,
    "min_child_weight": 2,
    "n_estimators": 100,
    "subsample": 0.8,
}


def build_pipeline() -> Pipeline:
    """TF-IDF + XGBoost como un solo Pipeline sklearn (entrada: texto crudo)."""
    tfidf = TfidfVectorizer(
        ngram_range=config.N_GRAMS,
        max_df=0.25,
        stop_words="english",
        max_features=config.MAX_WORDS_NUM,
    )
    clf = XGBClassifier(objective="binary:logistic", **PARAMS)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])
