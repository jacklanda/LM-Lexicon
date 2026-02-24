#!/usr/bin/env python3

import json

import spacy
from rich.console import Console

console = Console()

# Load English tokenizer, tagger,
# parser, NER and word vectors
nlp = spacy.load("en_core_web_trf")

# Process whole documents
dps = json.load(open("results/human-eval.new.json", "r"))

for idx in range(len(dps)):
    word = dps[idx]["word"]
    context = dps[idx]["context"]
    if word not in context:
        raise Exception("Word is not in context!")
        dps[idx]["pos"] = "N/A"
    doc = nlp(context)
    # select the word's token
    token = [token for token in doc if token.text == word][0]
    dps[idx]["pos"] = token.pos_
    console.print(f"Idx: {idx}, Word: {word}, POS: {dps[idx]['pos']}")

# reorder the dps
dps_new = []
for dp in dps:
    dps_new.append(
        {
            "word": dp["word"],
            "source": dp["source"],
            "context": dp["context"],
            "pos": dp["pos"],
            "definition": dp["definition"],
            "instruction": dp["instruction"],
            "prediction": dp["prediction"],
        }
    )
dps = dps_new

with open("results/human-eval.new.json", "w") as f:
    json.dump(dps, f, ensure_ascii=False, indent=4)
