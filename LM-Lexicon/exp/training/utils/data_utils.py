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
