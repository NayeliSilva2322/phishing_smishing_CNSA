import tensorflow as tf
from keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Embedding,
    GlobalMaxPooling1D,
    SpatialDropout1D,
    TextVectorization,
)
from keras.metrics import AUC, Precision, Recall
from keras.models import Sequential
from keras.optimizers import Adam

from .. import config

LEARNING_RATE = 1e-4

PARAMS = {
    "max_words_num": config.MAX_WORDS_NUM,
    "embedding_dim": config.EMBEDDING_DIM,
    "max_sequence_length": config.MAX_SEQUENCE_LENGTH,
    "learning_rate": LEARNING_RATE,
}


def build_model(train_texts) -> Sequential:
    """LSTM-CNN con TextVectorization incluida (entrada: texto crudo)."""
    tv = TextVectorization(
        max_tokens=config.MAX_WORDS_NUM,
        output_sequence_length=config.MAX_SEQUENCE_LENGTH,
        ngrams=config.N_GRAMS,
    )
    tv.adapt(train_texts)

    model = Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(tv)
    model.add(Embedding(config.MAX_WORDS_NUM, config.EMBEDDING_DIM, input_length=config.MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Conv1D(50, kernel_size=3, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall(), AUC()],
    )
    return model
