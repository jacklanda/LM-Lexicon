# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2024/01/10

import re
import json
import argparse
from statistics import mean
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from evaluate import load
from nltk import word_tokenize
from rich.console import Console
from transformers import logging
from sklearn.metrics import f1_score

# from sentence_transformers import SentenceTransformer, util
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

from calculate_scores import compute_sentence_bleu, compute_mover, compute_mauve

logging.set_verbosity_warning()


if __name__ != "__main__":
    # lexical-level scorers
    # BLEU_SCORER = load("bleu")
    # BLEU_SCORER = load("sacrebleu")
    BLEU_SCORER = compute_sentence_bleu
    # BLEU_SCORER = load("google_bleu")
    ROUGE_SCORER = load("rouge")
    METEOR_SCORER = load("meteor")
    # BLEURT_SCORER = load("bleurt")
    EXACT_MATCH = load("exact_match")
    # semantic-level scorers
    BERT_SCORER = load("bertscore")
    MOVER_SCORER = compute_mover
    MAUVE_SCORER = compute_mauve
    # PERPLEXITY = load("perplexity", module_type="metric")
    # MODEL_ST = SentenceTransformer(
    # model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    # )


def compute_macro_f1(preds: List[str], golds: List[str]) -> float:
    """
    Compute macro f1 score according to the input `pred_class_map` and `gold_class_map`,
    which include prediction class/gold class (str) to number.
    """
    return f1_score(y_pred=preds, y_true=golds, average="macro")


def compute_micro_f1(preds: List[str], golds: List[str]) -> float:
    """
    Compute micro f1 score dynamically according to the input `pred_class_map` and `gold_class_map`,
    which include prediction class/gold class (str) to number.
    """
    return f1_score(y_pred=preds, y_true=golds, average="micro")


def compute_weighted_f1(preds: List[str], golds: List[str]) -> float:
    """
    Compute weighted f1 score dynamically according to the input `pred_class_map` and `gold_class_map`,
    which include prediction class/gold class (str) to number.
    """
    return f1_score(y_pred=preds, y_true=golds, average="weighted")


def compute_metric(
    metric_name: str,
    pred: str,
    gold: str,
    *args,
    word: Optional[str] = None,
    refs: Optional[List[str]] = None,
    preds: Optional[List[str]] = None,
    golds: Optional[List[str]] = None,
    instance: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> float:
    """
    Compute metric according to the input `metric_name`, `pred` and `gold`.

    Args:
    - metric_name: str, the name of the metric.
    - pred: str, the prediction.
    - gold: str, the ground truth.
    - instance: Dict[str, Any], the instance to be evaluated.

    Returns:
    - float, the metric value.
    """
    # preprocess
    # remove trailing spaces for each item
    if isinstance(pred, list):
        pred = [p.strip() for p in pred]
    if isinstance(gold, list):
        gold = [g.strip() for g in gold]
    if isinstance(pred, str):
        pred = pred.strip()
    if isinstance(gold, str):
        gold = gold.strip()

    # lexical-level measurements: bleus, rouge, meteor, and exact match
    if metric_name == "rouge":
        if isinstance(pred, list) and isinstance(gold, list):
            return (
                max(
                    [
                        round(
                            ROUGE_SCORER.compute(
                                predictions=[p], references=[g], tokenizer=word_tokenize
                            )["rougeL"]
                            * 100,
                            2,
                        )
                        for p in pred
                        for g in gold
                    ]
                ),
                [
                    max(
                        [
                            round(
                                ROUGE_SCORER.compute(predictions=[p], references=[g])[
                                    "rougeL"
                                ]
                                * 100,
                                2,
                            )
                            for g in gold
                        ]
                    )
                    for p in pred
                ],
            )
        elif isinstance(pred, list):
            return (
                max(
                    [
                        round(
                            ROUGE_SCORER.compute(
                                predictions=[p],
                                references=[gold],
                                tokenizer=word_tokenize,
                            )["rougeL"]
                            * 100,
                            2,
                        )
                        for p in pred
                    ]
                ),
                [
                    round(
                        ROUGE_SCORER.compute(
                            predictions=[p], references=[gold], tokenizer=word_tokenize
                        )["rougeL"]
                        * 100,
                        2,
                    )
                    for p in pred
                ],
            )
        elif isinstance(gold, list):
            return (
                max(
                    [
                        round(
                            ROUGE_SCORER.compute(
                                predictions=[pred],
                                references=[g],
                                tokenizer=word_tokenize,
                            )["rougeL"]
                            * 100,
                            2,
                        )
                        for g in gold
                    ]
                ),
                [
                    max(
                        [
                            round(
                                ROUGE_SCORER.compute(
                                    predictions=[pred],
                                    references=[g],
                                    tokenizer=word_tokenize,
                                )["rougeL"]
                                * 100,
                                2,
                            )
                            for g in gold
                        ]
                    )
                ],
            )
        else:
            return round(
                ROUGE_SCORER.compute(
                    predictions=[pred], references=[gold], tokenizer=word_tokenize
                )["rougeL"]
                * 100,
                2,
            )
    dataset = kwargs.get("dataset", None)
    if metric_name == "bleu-hf":
        return BLEU_SCORER(
            pred, gold, word, refs, mode="hf"
        )  # sentence-bleu by Hugging Face
    if metric_name == "bleu-sacre":
        return BLEU_SCORER(pred, gold, word, refs, mode="sacre")
    if metric_name == "bleu-nltk":
        return BLEU_SCORER(pred, gold, word, refs, mode="nltk")  # sentence-bleu by nltk
    if metric_name == "bleu-cpp":
        return BLEU_SCORER(
            pred, gold, word, refs, mode="cpp", dataset=dataset
        )  # sentence-bleu by cpp
    if metric_name == "meteor":
        if isinstance(pred, list) and isinstance(gold, list):
            return max(
                [
                    METEOR_SCORER.compute(predictions=[p], references=[g])["meteor"]
                    for p in pred
                    for g in gold
                ]
            )
        elif isinstance(pred, list):
            return max(
                [
                    METEOR_SCORER.compute(predictions=[p], references=[gold])["meteor"]
                    for p in pred
                ]
            )
        elif isinstance(gold, list):
            return max(
                [
                    METEOR_SCORER.compute(predictions=[pred], references=[g])["meteor"]
                    for g in gold
                ]
            )
        else:
            return METEOR_SCORER.compute(predictions=[pred], references=[gold])[
                "meteor"
            ]
    if metric_name == "exact-match":
        if isinstance(pred, list) and isinstance(gold, list):
            return max(
                [
                    EXACT_MATCH.compute(
                        predictions=[p],
                        references=[g],
                        ignore_case=True,
                        ignore_punctuation=True,
                    )["exact_match"]
                    for p in pred
                    for g in gold
                ]
            )
        elif isinstance(pred, list):
            return max(
                [
                    EXACT_MATCH.compute(
                        predictions=[p],
                        references=[gold],
                        ignore_case=True,
                        ignore_punctuation=True,
                    )["exact_match"]
                    for p in pred
                ]
            )
        elif isinstance(gold, list):
            return max(
                [
                    EXACT_MATCH.compute(
                        predictions=[pred],
                        references=[g],
                        ignore_case=True,
                        ignore_punctuation=True,
                    )["exact_match"]
                    for g in gold
                ]
            )
        else:
            return EXACT_MATCH.compute(
                predictions=[pred],
                references=[gold],
                ignore_case=True,
                ignore_punctuation=True,
            )["exact_match"]

    # semantic-level measurements: bert score, mover score, and mauve
    if metric_name == "bert-score":
        if isinstance(pred, list) and isinstance(gold, list):
            return max(
                [
                    BERT_SCORER.compute(
                        predictions=[p], references=[g], lang="en", nthreads=16
                    )["f1"]
                    for p in pred
                    for g in gold
                ]
            )
        elif isinstance(pred, list):
            return max(
                [
                    BERT_SCORER.compute(
                        predictions=[p], references=[gold], lang="en", nthreads=16
                    )["f1"]
                    for p in pred
                ]
            )
        elif isinstance(gold, list):
            return max(
                [
                    BERT_SCORER.compute(
                        predictions=[pred], references=[g], lang="en", nthreads=16
                    )["f1"]
                    for g in gold
                ]
            )
        else:
            return mean(
                BERT_SCORER.compute(
                    predictions=[pred], references=[gold], lang="en", nthreads=16
                )["f1"]
            )
    if metric_name == "mover-score":
        if isinstance(pred, list) and isinstance(gold, list):
            return max(
                [
                    MOVER_SCORER(preds=[p], refs=[g], words=word)
                    for p in pred
                    for g in gold
                ]
            )
        elif isinstance(pred, list):
            return max([MOVER_SCORER(preds=[p], refs=[gold], words=word) for p in pred])
        elif isinstance(gold, list):
            return max([MOVER_SCORER(preds=[pred], refs=[g], words=word) for g in gold])
        else:
            return MOVER_SCORER(preds=[pred], refs=[gold], words=word)
        # if metric_name == "mauve":
        if isinstance(pred, list) and isinstance(gold, list):
            return max([MAUVE_SCORER(preds=[p], refs=[g]) for p in pred for g in gold])
        elif isinstance(pred, list):
            return max([MAUVE_SCORER(preds=[p], refs=[gold]) for p in pred])
        elif isinstance(gold, list):
            return max([MAUVE_SCORER(preds=[pred], refs=[g]) for g in gold])
        else:
            return MAUVE_SCORER(
                preds=preds, refs=golds
            )  # two distributions that each must contains at least a few thousand of samples
    # if metric_name == "bleurt":
    # return mean(BLEURT_SCORER.compute(predictions=[pred], references=[gold])["scores"])
    # if metric_name == "cos-sim":
    # return cosineSimilarity(pred, gold)
    # if metric_name == "perplexity":
    # return PERPLEXITY.compute(predictions=[pred], model_id="gpt2")["perplexities"][
    # 0
    # ]
    else:
        raise NotImplementedError(f"Metric {metric_name} not implemented!")


def perform_eval(task="vmwe", result_file: Optional[str] = None):
    console = Console()
    if task == "vmwe":
        label2count = dict()
        label2count_macro = dict()
        with open(result_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                result_obj = json.loads(line)
                label = result_obj["label"]
                root_label = label.split(".")[0]
                vmwe = result_obj["vmwe"]
                prediction = result_obj["prediction"]

                if label not in label2count:
                    label2count[label] = {"total": 0, "correct": 0}
                label2count[label]["total"] += 1

                if root_label not in label2count_macro:
                    label2count_macro[root_label] = {"total": 0, "correct": 0}
                label2count_macro[root_label]["total"] += 1

                if vmwe == prediction or "is " + vmwe in prediction:
                    label2count[label]["correct"] += 1
                    label2count_macro[root_label]["correct"] += 1

        for label, count in label2count.items():
            console.print(
                f"Label: {label}, Total: {count['total']}, Correct: {count['correct']}, Accuracy: {round(count['correct']/count['total'], 4) * 100}"
            )
        # display accuracy for each root label
        for label, count in label2count_macro.items():
            console.print(
                f"Label: {label}, Total: {count['total']}, Correct: {count['correct']}, Accuracy: {round(count['correct']/count['total'], 4) * 100}"
            )
    elif task == "nce":
        total, correct = 0, 0
        with open(result_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                result_obj = json.loads(line)
                label = result_obj["noun_compound"]
                prediction = result_obj["prediction"]
                if prediction == label:
                    correct += 1
                total += 1
        console.print(
            f"Total: {total}, Correct: {correct}, Accuracy: {round(correct/total, 4) * 100}"
        )
    elif task == "lcc":
        label2count = dict()
        gold2pred = dict()
        labels = [
            "Magn",
            "AntiMagn",
            "Ver",
            "AntiVer",
            "Bon",
            "AntiBon",
            "Son",
            "Oper1",
        ]
        with open(result_file, "r") as f:
            lines = f.readlines()
            # calc each category total number and prediction distribution in the category
            for line in lines:
                result_obj = json.loads(line)
                label = result_obj["label"]
                prediction = result_obj["prediction"]
                if label not in label2count:
                    label2count[label] = {"total": 0, "correct": 0}
                label2count[label]["total"] += 1
                if prediction == label:
                    label2count[label]["correct"] += 1
                if label not in gold2pred:
                    gold2pred[label] = dict()
                if prediction not in gold2pred[label]:
                    gold2pred[label][prediction] = 0
                gold2pred[label][prediction] += 1
        # display prediction distribution for each label
        for label in labels:
            console.print("  " + label, end="")
        print()
        for label in labels:
            console.print(label[:7], end="\t")
            dist = "\t".join([str(gold2pred[label].get(pred, 0)) for pred in labels])
            console.print(dist)
        print()
        # generate a 2-D matrix from gold2pred
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for i, label in enumerate(labels):
            for j, pred in enumerate(labels):
                matrix[i][j] = gold2pred[label].get(pred, 0)
        matrix_str = "\n".join([",".join([str(e) for e in row]) for row in matrix])
        matrix_str = "[[" + matrix_str.replace("\n", "],\n[") + "]]"
        console.print(matrix_str)
        # display global accuracy
        total, correct = 0, 0
        for label, count in label2count.items():
            total += count["total"]
            correct += count["correct"]
            console.print(
                f"Label: {label}, Total: {count['total']}, Correct: {count['correct']}, Accuracy: {round(count['correct']/count['total'], 4) * 100}"
            )
        print()
        # total accuracy
        console.print(
            f"Total: {total}, Correct: {correct}, Accuracy: {round(correct/total, 4) * 100}"
        )
    elif task in ["iep", "lcp", "ncp"]:
        rouge_scorer = load("rouge")
        bert_scorer = load("bertscore")
        rouge_l, bert_score_f1 = 0.0, 0.0
        with open(result_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                result_obj = json.loads(line)
                pred = result_obj["prediction"]
                if "references" in result_obj:
                    gold = result_obj["references"]
                elif "paraphrases" in result_obj:
                    gold = result_obj["paraphrases"]
                if isinstance(gold, list):
                    gold = [gold]
                elif isinstance(gold, str):
                    gold = [gold]
                rouge_l += rouge_scorer.compute(predictions=[pred], references=gold)[
                    "rougeL"
                ]
                bert_score_f1 += bert_scorer.compute(
                    predictions=[pred],
                    references=gold,
                    model_type="roberta-large",
                )["f1"][0]
        console.print(
            f"ROUGE-L: {round(rouge_l/len(lines), 4) * 100}, BERT-Score F1: {round(bert_score_f1/len(lines), 4) * 100}"
        )
    print()


def get_rid_of_period(lst: List[str]) -> List[str]:
    """removes all periods from each string in a given list,
    except for those periods that are part of decimal numbers

    Args:
        lst (List[str]): list of strings

    Returns:
        List[str]: list of strings with periods removed
    """
    pattern = re.compile(r"\.(?!\d)")
    return (
        [pattern.sub("", sent) for sent in lst]
        if isinstance(lst[0], str)
        else [[pattern.sub("", sent) for sent in sub_lst] for sub_lst in lst]
    )


def get_word2refs(
    words: List[str],
    refs: List[str],
    dedup_refs: bool = False,
) -> Dict[str, List[str]]:
    """creates a dictionary of words to references

    Args:
        words (List[str]): list of words
        refs (List[str]): list of references
        dedup_refs (bool, optional): deduplicate references. Defaults to False.

    Returns:
        Dict[str, List[str]]: dictionary of words to references
    """
    seen_refs = set()
    word2refs = dict()
    for word, ref in zip(words, refs):
        if word not in word2refs:
            word2refs[word] = []
        if dedup_refs:
            # NOTE: deduplicating references seems to be more reasonable
            if word + ref not in seen_refs:
                seen_refs.add(word + ref)
                word2refs[word].append(ref)
        else:
            # NOTE: follow Huang et al. (2021) to use un-deduplicated references to keep consistency
            word2refs[word].append(ref)

    return word2refs


if __name__ == "__main__":
    # bert_score = compute_metric(
    # metric_name="bert-score",
    pred = ("Michael is a good man.",)
    gold = ("Michael is a good dog.",)
    # )
    # print(bert_score)
    # accept first argument
    # parser = argparse.ArgumentParser(description="Description of your program")
    # parser.add_argument("--task", required=True, help="Task to evaluate.")
    # parser.add_argument(
    #     "--result_file_path", required=True, help="Path to result file."
    # )
    # parsed_args = parser.parse_args()
    # # print(parsed_args.result_file_path)
    # perform_eval(task=parsed_args.task, result_file=parsed_args.result_file_path)
    ROUGE_SCORER = load("rouge")
    rouge_l = ROUGE_SCORER.compute(predictions=pred, references=gold)["rougeL"]
