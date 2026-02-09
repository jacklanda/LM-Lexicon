from typing import List

from datasets import Dataset

from transformers import AutoTokenizer


def dump_transform_dataset(
    dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 8192
):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize_function, batched=True)


def postprocess_text(preds: List[str], labels: List[str]):
    """Preprocess text for computing metrics."""
    # heuristic rules for preprocessing
    preds = [" ".join(pred.split('"?\n\n', maxsplit=1)[-1].split()) for pred in preds]
    # preds = [
    # " ".join(pred.replace("\n", "").rsplit("?", maxsplit=1)[-1].split())
    # for pred in preds
    # ]
    preds = [p if (p != "" and not p.isspace()) else "<unk>" for p in preds]
    preds = [" ".join(pred.split()) for pred in preds]
    labels = [" ".join(label.replace("\n", "").split()) for label in labels]
    return preds, labels


def get_dataset_path_prefix(data_path: str):
    """Get the dataset name from the data path.

    Args:
    - data_path: str, the path to the dataset.

    Returns:
    - str, the name of the dataset.
    """
    return (
        "dataset/3D-EX"
        if "3D-EX" in data_path
        else (
            "dataset/wordnet"
            if "wordnet" in data_path
            else (
                "dataset/oxford"
                if "oxford" in data_path
                else (
                    "dataset/wiki"
                    if "wiki" in data_path
                    else ("dataset/slang" if "slang" in data_path else None)
                )
            )
        )
    )
