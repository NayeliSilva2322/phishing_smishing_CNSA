from datasets import load_dataset
from sklearn.model_selection import train_test_split

from . import config


def import_dataset_from_hf(hf_config_name: str):
    return load_dataset(
        config.DATASET_PATH, hf_config_name, trust_remote_code=True
    )["train"].to_pandas()


def load_all_datasets() -> dict:
    """Carga los 4 datasets del notebook: text, url, web, combined."""
    datasets = {
        key: import_dataset_from_hf(hf_name)
        for key, hf_name in config.HF_DATASET_CONFIGS.items()
    }

    # el dataset de urls se reduce al 5%, igual que en el notebook original
    datasets["url"], _ = train_test_split(
        datasets["url"],
        test_size=config.URL_SUBSAMPLE_TEST_SIZE,
        stratify=datasets["url"]["label"],
        random_state=config.RANDOM_STATE,
    )
    return datasets


def split_dataset(df):
    return train_test_split(
        df["text"],
        df["label"],
        stratify=df["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
