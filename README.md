# Phishing Classifiers & MLflow Evaluation

A reproducible evaluation framework for **phishing URL/text classification**, integrating multiple machine learning and deep learning approaches with **MLflow** for experiment tracking, model logging, and registry management.

The project evaluates classical machine learning, neural networks, and transformer-based models under a unified evaluation pipeline, enabling systematic comparison across datasets and model architectures.

## Overview

The framework supports the evaluation of:

* **XGBoost** with TF-IDF features
* **Multinomial Naive Bayes** with TF-IDF features
* **LSTM-CNN** with integrated text vectorization
* **BERT-base** for zero-shot inference
* **Dataset-specific fine-tuned BERT** checkpoints
* **Pretrained phishing-domain BERT checkpoints**

BERT models are evaluated using existing checkpoints. Fine-tuning is intentionally handled outside this repository to keep the evaluation pipeline independent from model training.

## Evaluation Pipeline

Each experiment follows a consistent workflow:

1. Load and prepare the selected dataset.
2. Train lightweight classifiers when applicable.
3. Load pretrained or fine-tuned checkpoints for transformer models.
4. Generate predictions on the test set.
5. Compute classification metrics.
6. Generate ROC and Precision-Recall curves.
7. Track parameters, metrics, figures, and models in MLflow.
8. Register the resulting models in the MLflow Model Registry.

The evaluation metrics include:

* Accuracy
* Precision
* Recall
* ROC-AUC

## MLflow Tracking

Each evaluation run is recorded in **MLflow** with:

* Model and dataset configuration
* Hyperparameters
* Evaluation metrics
* ROC curves
* Precision-Recall curves
* Serialized model artifacts
* Model Registry entries

Registered models follow a consistent naming convention:

```text
phishing-{model}-{dataset}
```

Examples:

```text
phishing-xgboost-text
phishing-mnb-combined
phishing-bert-finetuned-combined
```

### Reproducible Inference

Classical models are stored as complete `sklearn.Pipeline` objects containing both TF-IDF preprocessing and the classifier.

The LSTM-CNN model includes its `TextVectorization` layer.

This allows registered models to receive **raw text input**, minimizing the need to reproduce preprocessing manually during inference.

## Model Scope

| Model                    | Training in this project | Evaluation | MLflow |
| ------------------------ | -----------------------: | ---------: | -----: |
| XGBoost                  |                      Yes |        Yes |    Yes |
| Multinomial Naive Bayes  |                      Yes |        Yes |    Yes |
| LSTM-CNN                 |                      Yes |        Yes |    Yes |
| BERT-base                |                       No |        Yes |    Yes |
| Fine-tuned BERT          |                       No |        Yes |    Yes |
| Phishing BERT checkpoint |                       No |        Yes |    Yes |

Fine-tuned BERT evaluation requires a previously generated checkpoint. If the expected checkpoint is unavailable, the corresponding evaluation is skipped without interrupting the remaining experiments.

## Installation

```bash
pip install -r requirements.txt
```

Configure the environment:

```bash
cp .env.example .env
```

Update `.env` with the dataset paths, BERT checkpoint locations, and MLflow tracking URI.

## MLflow

For local experiment tracking:

```bash
mlflow ui --port 5000
```

Configure the tracking server in `.env`:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
```

An existing remote MLflow server can also be used by changing the tracking URI.

## Running Evaluations

Run the complete evaluation suite:

```bash
python run_evaluation.py
```

Run selected datasets and models:

```bash
python run_evaluation.py \
  --datasets text combined \
  --models xgboost mnb bert-base
```

Configure LSTM-CNN training parameters:

```bash
python run_evaluation.py \
  --models lstm-cnn \
  --lstm-epochs 7 \
  --lstm-batch-size 32
```

After execution, experiments and registered models can be inspected through the MLflow UI.

## Testing

Run the test suite with:

```bash
pip install pytest
pytest tests/ -v
```

The tests cover:

* Metric computation
* ROC and Precision-Recall figure generation
* MLflow parameter logging
* MLflow metric logging
* Figure and model artifact logging

The test suite is designed to run without real datasets, GPU resources, or an active MLflow server.

## Project Structure

```text
phishing-classifiers-mlflow/
├── requirements.txt
├── .env.example
├── run_evaluation.py
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── mlflow_tracking.py
│   ├── evaluate.py
│   │
│   └── models/
│       ├── xgboost_model.py
│       ├── mnb_model.py
│       ├── lstm_cnn_model.py
│       └── bert_model.py
│
└── tests/
    ├── test_metrics.py
    └── test_mlflow_tracking.py
```

## Configuration

Project configuration is managed through `.env`.

Relevant variables include:

```env
PHISHING_DATASET_PATH=
BERT_FINETUNED_PHISHING_PATH=
MLFLOW_TRACKING_URI=
```

Additional configuration includes model parameters, dataset definitions, evaluation settings, and checkpoint locations.

Paths should be updated according to the local project environment.

## Technical Focus

This project demonstrates an end-to-end approach to **machine learning evaluation and experiment management**, combining:

* Classical machine learning
* Deep learning for text classification
* Transformer-based inference
* Reproducible preprocessing pipelines
* Automated evaluation
* MLflow experiment tracking
* Model artifact management
* Model Registry integration
* Automated testing
