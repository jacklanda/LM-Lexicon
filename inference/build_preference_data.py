#!/usr/bin/env python

import os
import json
from tqdm import tqdm
from rich.console import Console

console = Console()

dataset = "wordnet"

train_data_path = f"results/{dataset}/word-interpretation-scratch2-nlp-liuyang-lm-lexicon-dense-{dataset}-best-ckpt.train.json"
test_data_path = f"results/{dataset}/word-interpretation-scratch2-nlp-liuyang-lm-lexicon-dense-{dataset}-best-ckpt.test.json"

console.log(f"Train data path: {train_data_path}")
console.log(f"Test data path: {test_data_path}")

data_paths = [train_data_path, test_data_path]
data_dir = train_data_path.split("/", maxsplit=1)[-1].split("/", maxsplit=1)[0].strip()


def build_preference_data():
    for path in data_paths:
        preference_data = []
        with open(path, "r") as f:
            data = json.load(f)
        for dp in tqdm(data):
            preds = dp["prediction"]
            log_probs = dp["log_probs"]
            bleu = dp["bleu"]
            rouge = dp["rouge"]
            if len(preds) <= 1:
                # console.log(
                # f'Skipping "{dp["word"]}" as it has less than 2 predictions.'
                # )
                continue
            for idx, pred in enumerate(preds):
                log_prob = log_probs[idx]
                bleu_score = bleu[idx]
                rouge_score = rouge[idx]
                if bleu_score == 100.0 or rouge_score == 100.0 or log_prob > -0.1:
                    # console.log(f'Skipping "{dp["word"]}" as it has a perfect score.')
                    continue
                preference_data.append(
                    {
                        "word": dp["word"],
                        "source": dp["source"],
                        "context": dp["context"],
                        "instruction": dp["instruction"],
                        # "chosen": dp["definition"],
                        "chosen": dp["definition"],
                        "rejected": pred,
                        "logprobs": log_prob,
                        "bleu": bleu_score,
                        "rouge": rouge_score,
                    }
                )

        os.makedirs(f"results/{data_dir}/preference", exist_ok=True)

        if "train" in path:
            split = "train"
        elif "test" in path:
            split = "test"
        else:
            raise ValueError("Invalid data split.")

        with open(f"results/{dataset}/preference/{split}.json", "w") as f:
            json.dump(preference_data, f, ensure_ascii=False, indent=4)
        console.print(
            f'Preference data saved to "results/{data_dir}/preference/{split}.json".'
        )


if __name__ == "__main__":
    build_preference_data()
