"""CLI: evalua clasificadores de phishing sobre el set de test y los registra en MLflow.

No entrena BERT (usa checkpoints ya afinados en disco) y no genera nada de Docker.

Ejemplos:
    python run_evaluation.py
    python run_evaluation.py --datasets text combined --models xgboost mnb bert-base
"""
import argparse
import os

from src import config, data, mlflow_tracking
from src.evaluate import evaluate_bert_pipeline, evaluate_lstm_cnn, evaluate_sklearn_pipeline
from src.models import bert_model, mnb_model, xgboost_model

ALL_DATASETS = list(config.HF_DATASET_CONFIGS.keys())  # ["text", "url", "web", "combined"]
ALL_MODELS = ["xgboost", "mnb", "lstm-cnn", "bert-base", "bert-finetuned", "bert-phishing-checkpoint"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--lstm-epochs", type=int, default=4)
    parser.add_argument("--lstm-batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    mlflow_tracking.setup_mlflow()
    datasets = data.load_all_datasets()

    bert_base_pipe = None
    bert_phishing_pipe = None
    results = {}

    for dataset_key in args.datasets:
        xtrain, xtest, ytrain, ytest = data.split_dataset(datasets[dataset_key])

        if "xgboost" in args.models:
            results[f"xgboost-{dataset_key}"] = evaluate_sklearn_pipeline(
                xgboost_model.build_pipeline, xgboost_model.PARAMS, "xgboost", dataset_key,
                xtrain, xtest, ytrain, ytest,
            )

        if "mnb" in args.models:
            results[f"mnb-{dataset_key}"] = evaluate_sklearn_pipeline(
                mnb_model.build_pipeline, mnb_model.PARAMS, "mnb", dataset_key,
                xtrain, xtest, ytrain, ytest,
            )

        if "lstm-cnn" in args.models:
            results[f"lstm-cnn-{dataset_key}"] = evaluate_lstm_cnn(
                dataset_key, xtrain, xtest, ytrain, ytest,
                epochs=args.lstm_epochs, batch_size=args.lstm_batch_size,
            )

        if "bert-base" in args.models:
            if bert_base_pipe is None:
                bert_base_pipe = bert_model.load_bert_base_pipeline()
            results[f"bert-base-{dataset_key}"] = evaluate_bert_pipeline(
                bert_base_pipe, "bert-base", dataset_key, xtest, ytest,
                params={"base_model": config.BERT_BASE_MODEL, "finetuned": False},
            )

        if "bert-finetuned" in args.models:
            model_dir = config.BERT_FINETUNED_DIR_TEMPLATE.format(dataset=dataset_key)
            if os.path.isdir(model_dir):
                pipe = bert_model.load_finetuned_pipeline(model_dir)
                results[f"bert-finetuned-{dataset_key}"] = evaluate_bert_pipeline(
                    pipe, "bert-finetuned", dataset_key, xtest, ytest,
                    params={"base_model": config.BERT_BASE_MODEL, "checkpoint": model_dir},
                )
            else:
                print(
                    f"[aviso] No existe {model_dir}; se omite bert-finetuned para '{dataset_key}' "
                    "(este script solo evalua checkpoints existentes, no entrena)."
                )

        if "bert-phishing-checkpoint" in args.models:
            if bert_phishing_pipe is None:
                bert_phishing_pipe = bert_model.load_finetuned_pipeline(
                    config.BERT_FINETUNED_PHISHING_PATH
                )
            results[f"bert-phishing-checkpoint-{dataset_key}"] = evaluate_bert_pipeline(
                bert_phishing_pipe, "bert-phishing-checkpoint", dataset_key, xtest, ytest,
                params={"checkpoint": config.BERT_FINETUNED_PHISHING_PATH},
            )

    print("\n=== Resumen ===")
    for name, m in results.items():
        print(f"{name}: {m}")


if __name__ == "__main__":
    main()
