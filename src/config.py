import os

# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "phishing-classifiers")

# --- Datos y checkpoints (mismas rutas relativas que el notebook original) ---
DATASET_PATH = os.getenv("PHISHING_DATASET_PATH", "../phishing-dataset")
BERT_FINETUNED_PHISHING_PATH = os.getenv("BERT_FINETUNED_PHISHING_PATH", "../bert-finetuned-phishing")
BERT_BASE_MODEL = os.getenv("BERT_BASE_MODEL", "bert-large-uncased")
BERT_FINETUNED_DIR_TEMPLATE = os.getenv(
    "BERT_FINETUNED_DIR_TEMPLATE", "./bert-finetuned-phishing-{dataset}"
)

# --- Datasets (mismos nombres de config de HF que en el notebook) ---
HF_DATASET_CONFIGS = {
    "text": "texts",
    "url": "urls",
    "web": "webs",
    "combined": "combined_reduced",
}

# --- Vectorizacion / chunking (mismos valores que el notebook original) ---
MAX_WORDS_NUM = 2000
N_GRAMS = (1, 2)
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 10

RANDOM_STATE = 42
TEST_SIZE = 0.2
URL_SUBSAMPLE_TEST_SIZE = 0.95  # se conserva solo 5% de urls, igual que el notebook
