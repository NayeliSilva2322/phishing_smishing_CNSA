from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .. import config

ID2LABEL = {0: "benign", 1: "phishing"}
LABEL2ID = {"benign": 0, "phishing": 1}


def _device() -> int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def load_bert_base_pipeline():
    """BERT-base pre-entrenado, sin ningun fine-tuning (zero-shot)."""
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_BASE_MODEL, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    return pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        device=_device(),
    )


def load_finetuned_pipeline(model_path: str, tokenizer_name: str | None = None):
    """Carga un pipeline BERT YA afinado desde disco. No reentrena nada:
    este proyecto cubre solo la fase de test/evaluacion."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or config.BERT_BASE_MODEL)
    return pipeline(
        model=model_path,
        tokenizer=tokenizer,
        truncation=True,
        device=_device(),
    )
