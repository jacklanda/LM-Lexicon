#!/usr/bin/env python

import json

result_paths = [
    "results/wordnet/word-interpretation-claude-3-5-sonnet-20241022.test.json",
    "results/wordnet/word-interpretation-gemini-1.5-pro-latest.test.json",
    "results/wordnet/word-interpretation-gpt-4o-2024-11-20.test.json",
    "results/wordnet/word-interpretation-scratch2-nlp-liuyang-lm-lexicon-dense-wordnet-best-ckpt.test.json",
]

models = [
    "claude-3-opus-20240229",
    "gemini-1.5-pro-latest",
    "gpt-4-turbo-2024-04-09",
    "lm-lexicon-dense",
]

results_list = []
instruction2results = {}
for result_path in result_paths:
    with open(result_path) as f:
        results = json.load(f)
        # add model name to each result
        for item in results:
            item["model"] = models[result_paths.index(result_path)]
    for item in results:
        instruction = item["instruction"]
        if instruction not in instruction2results:
            instruction2results[instruction] = []
        instruction2results[instruction].append(item)

# filter out instructions that have less than 4 results
for instruction, results in instruction2results.items():
    if len(results) >= 4:
        # merge results by model
        merged_result = {
            "word": results[0]["word"],
            "source": results[0]["source"],
            "context": results[0]["context"],
            "definition": results[0]["definition"],
            "instruction": results[0]["instruction"],
            "prediction": {
                "claude-3-opus-20240229": None,
                "gemini-1.5-pro-latest": None,
                "gpt-4-turbo-2024-04-09": None,
                "lm-lexicon-dense": None,
            },
        }
        for result in results:
            model = result["model"]
            merged_result["prediction"][model] = (
                result["prediction"]
                if isinstance(result["prediction"], str)
                else result["prediction"][0]
            )

        results_list.append(merged_result)

with open("results/wordnet/human-eval.json", "w") as f:
    json.dump(results_list, f, indent=4)
