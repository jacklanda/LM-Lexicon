import json

import pandas as pd
from rich.console import Console

console = Console()


def preprocess_lesc(input_path: str, output_path: str, option_limit: int = 4):
    datapoints = []
    df = pd.read_json(input_path)
    instruction = ""
    for i in range(len(df)):
        word = df.loc[i, "word"]
        input_text = df.loc[i, "input_text_en"]
        label = df.loc[i, "label"]
        num_of_options = int(df.loc[i, "num_of_options"])
        options = [
            o.strip()
            for o in df.loc[i, "input_text_en"]
            .split("\n", 1)[-1]
            .rsplit("\n", 1)[0]
            .split("\n")
            if (o.startswith(alpha) for alpha in ["A", "B", "C", "D"])
        ]
        if num_of_options != option_limit:
            continue
        datapoint = {
            "word": word,
            "input_text": input_text,
            "instruction": instruction,
            "options": options,
            "label": label,
        }
        console.log(datapoint)
        datapoints.append(datapoint)
    with open(output_path, "w") as f:
        for datapoint in datapoints:
            # f.write(json.dumps(datapoint, ensure_ascii=False, indent=4) + "\n")
            f.write(json.dumps(datapoint, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_path = "dataset/lesc.json"
    preprocess_lesc(
        input_path="dataset/lesc.json",
        output_path="dataset/prepared_lesc.jsonl",
        option_limit=4,
    )
