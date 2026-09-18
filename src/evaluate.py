"""Fase de TEST/evaluacion de cada clasificador, con logging + registro en MLflow.

Los modelos clasicos (XGBoost, MNB, LSTM-CNN) se ajustan aqui porque no existe
una version "pre-entrenada" de ellos -- ajustarlos es la unica forma de obtener
predicciones para evaluar. Los modelos BERT, en cambio, NUNCA se reentrenan en
este modulo: se cargan checkpoints ya afinados desde disco y solo se corre
inferencia sobre el set de test.
"""
from datasets import Dataset
from tqdm.auto import tqdm

from . import config, metrics, mlflow_tracking
from .models import bert_model


def evaluate_sklearn_pipeline(
    pipeline_builder, params: dict, model_key: str, dataset_key: str, xtrain, xtest, ytrain, ytest
) -> dict:
    pipe = pipeline_builder()
    pipe.fit(xtrain, ytrain)

    pred = pipe.predict(xtest)
    predproba = pipe.predict_proba(xtest)[:, 1]

    computed = metrics.compute_metrics(ytest, pred, predproba)
    figures = {
        "roc_curve.png": metrics.roc_curve_figure(ytest, predproba, model_key, computed),
        "pr_curve.png": metrics.pr_curve_figure(ytest, predproba, model_key),
    }

    mlflow_tracking.log_run(
        run_name=f"{model_key}-{dataset_key}",
        params=params,
        metrics=computed,
        figures=figures,
        model=pipe,
        flavor="sklearn",
        registered_model_name=f"phishing-{model_key}-{dataset_key}",
        input_example=xtest.iloc[:2],
    )
    return computed


def evaluate_lstm_cnn(dataset_key: str, xtrain, xtest, ytrain, ytest, epochs=4, batch_size=32) -> dict:
    import tensorflow as tf
    from keras.utils import to_categorical

    from .models import lstm_cnn_model

    model = lstm_cnn_model.build_model(xtrain)
    model.fit(xtrain, to_categorical(ytrain, num_classes=2), epochs=epochs, batch_size=batch_size)

    proba = model.predict(xtest)
    predproba = proba[:, 1]
    pred = tf.argmax(proba, axis=1).numpy()

    computed = metrics.compute_metrics(ytest, pred, predproba)
    figures = {
        "roc_curve.png": metrics.roc_curve_figure(ytest, predproba, "LSTM-CNN", computed),
        "pr_curve.png": metrics.pr_curve_figure(ytest, predproba, "LSTM-CNN"),
    }

    mlflow_tracking.log_run(
        run_name=f"lstm-cnn-{dataset_key}",
        params={**lstm_cnn_model.PARAMS, "epochs": epochs, "batch_size": batch_size},
        metrics=computed,
        figures=figures,
        model=model,
        flavor="tensorflow",
        registered_model_name=f"phishing-lstm-cnn-{dataset_key}",
    )
    return computed


def evaluate_bert_pipeline(pipe, model_key: str, dataset_key: str, xtest, ytest, params: dict) -> dict:
    """Solo inferencia -- nunca llama a Trainer.train()."""
    pred, predproba = [], []
    for out in tqdm(pipe(Dataset.from_pandas(xtest.to_frame())["text"])):
        pred.append(bert_model.LABEL2ID[out["label"]])
        score = out["score"] if out["label"] == "phishing" else 1 - out["score"]
        predproba.append(score)

    computed = metrics.compute_metrics(ytest, pred, predproba)
    figures = {
        "roc_curve.png": metrics.roc_curve_figure(ytest, predproba, model_key, computed),
        "pr_curve.png": metrics.pr_curve_figure(ytest, predproba, model_key),
    }

    mlflow_tracking.log_run(
        run_name=f"{model_key}-{dataset_key}",
        params=params,
        metrics=computed,
        figures=figures,
        model=pipe,
        flavor="transformers",
        registered_model_name=f"phishing-{model_key}-{dataset_key}",
    )
    return computed
