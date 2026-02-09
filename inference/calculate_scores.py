#!/usr/bin/env python3

import os
import re
import sys
import json
import math
import time
import random
from statistics import mean
from collections import defaultdict
from typing import Dict, List, Optional
from subprocess import Popen, PIPE

from evaluate import load
from datasets import Dataset
from nltk import word_tokenize
from rich.console import Console
from nltk.translate import bleu_score
# from moverscore_v2 import word_mover_score

console = Console()
SACREBLEU = load("sacrebleu")
# MAUVE_SCORER = load("mauve")
HF_BLEU = load("bleu")
ROUGE_SCORER = load("rouge")
METEOR_SCORER = load("meteor")


def compute_meteor(
    pred: str,
    ref: str,
    word: Optional[str] = None,
    refs: Optional[List[str]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    policy: Optional[str] = "max",
    dataset: Optional[str] = "3D-EX",
    **kwargs,
) -> float:
    if isinstance(pred, list) and isinstance(ref, list):
        return max(
            [
                METEOR_SCORER.compute(predictions=[p], references=[g])["meteor"]
                for p in pred
                for g in ref
            ]
        )
    elif isinstance(pred, list):
        return max(
            [
                METEOR_SCORER.compute(predictions=[p], references=[ref])["meteor"]
                for p in pred
            ]
        )
    elif isinstance(ref, list):
        return max(
            [
                METEOR_SCORER.compute(predictions=[pred], references=[g])["meteor"]
                for g in ref
            ]
        )
    else:
        return METEOR_SCORER.compute(predictions=[pred], references=[ref])["meteor"]


def compute_rouge(
    pred: str,
    ref: str,
    word: Optional[str] = None,
    refs: Optional[List[str]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    policy: Optional[str] = "max",
    dataset: Optional[str] = "3D-EX",
    **kwargs,
) -> float:
    preds = get_rid_of_period([pred])

    if refs is None:
        refs = get_rid_of_period([ref])
    else:
        refs = get_rid_of_period(refs)

    rouge_scores = []
    for pred in preds:
        if isinstance(pred, str):
            preds = [pred]
        else:
            preds = pred
        for pred in preds:
            if one_to_one:
                ref = refs[0]
            else:
                ref = refs
            if isinstance(pred, list) and isinstance(ref, list):
                score = max(
                    max(
                        [
                            round(
                                ROUGE_SCORER.compute(
                                    predictions=[p],
                                    references=[g],
                                    tokenizer=word_tokenize,
                                )["rougeL"]
                                * 100,
                                2,
                            )
                            for p in pred
                            for g in ref
                        ]
                    ),
                    [
                        max(
                            [
                                round(
                                    ROUGE_SCORER.compute(
                                        predictions=[p], references=[g]
                                    )["rougeL"]
                                    * 100,
                                    2,
                                )
                                for g in ref
                            ]
                        )
                        for p in pred
                    ],
                )
            elif isinstance(pred, list):
                score = max(
                    max(
                        [
                            round(
                                ROUGE_SCORER.compute(
                                    predictions=[p],
                                    references=[ref],
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
                                predictions=[p],
                                references=[ref],
                                tokenizer=word_tokenize,
                            )["rougeL"]
                            * 100,
                            2,
                        )
                        for p in pred
                    ],
                )
            elif isinstance(ref, list):
                score = max(
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
                            for g in ref
                        ]
                    ),
                    [
                        max(
                            [
                                round(
                                    ROUGE_SCORER.compute(
                                        predictions=[pred],
                                        references=[g],
                                        # tokenizer=word_tokenize,
                                    )["rougeL"]
                                    * 100,
                                    2,
                                )
                                for g in ref
                            ]
                        )
                    ],
                )
            else:
                score = round(
                    ROUGE_SCORER.compute(
                        predictions=[pred], references=[ref], tokenizer=word_tokenize
                    )["rougeL"]
                    * 100,
                    2,
                )
            if isinstance(score, list):
                rouge_scores += score
            else:
                rouge_scores.append(score)

    rouge_score = max(rouge_scores) if len(rouge_scores) > 0 else 0.0

    return rouge_score


def compute_sentence_bleu(
    pred: str,
    ref: str,
    word: Optional[str] = None,
    refs: Optional[List[str]] = None,
    mode: str = "nltk",
    one_to_one: bool = False,
    dedup_refs: bool = False,
    policy: Optional[str] = "max",
    dataset: Optional[str] = "3D-EX",
) -> float:
    def is_float(s: str) -> bool:
        """checks if a given string is a float

        Args:
            s (str): string

        Returns:
            bool: True if the string is a float, False otherwise
        """
        try:
            float(s)
            return True
        except ValueError:
            return False
        except Exception:
            return False

    preds = get_rid_of_period([pred])

    if refs is None:
        refs = get_rid_of_period([ref])
    else:
        refs = get_rid_of_period(refs)

    tmp_dir = "tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # prepare reference files for sentence-bleu.cpp
    seen_refs = set()
    word2ref_paths = dict()
    ref_path_prefix = "tmp/ref"

    if refs is None:
        word2ref_file_path = f"dataset/{dataset}/word2refs.json"
        console.log('Re-loading "word2refs" from path: "{word2ref_file_path}"')
        word2refs = get_word2refs(cache_path=word2ref_file_path)
        refs = word2refs[word] if word is not None else refs
    for ref in refs:
        ref_path = (
            ref_path_prefix
            + "-"
            + str(random.randint(0, 10000000))
            + "-"
            + str(round(time.time() * 1000))
            + ".txt"
        )
        with open(ref_path, "w") as f:
            # f.write(ref + "\n")
            f.write(ref)
        if word not in word2ref_paths:
            word2ref_paths[word] = []
        if dedup_refs:
            # NOTE: deduplicating references seems to be more reasonable
            if word + ref not in seen_refs:
                seen_refs.add(word + ref)
                word2ref_paths[word].append(ref_path)
        else:
            # NOTE: follow Huang et al. (2021) to use un-deduplicated references to keep consistency
            word2ref_paths[word].append(ref_path)

    bleus = []
    for pred in preds:
        if mode == "hf":
            sentence_bleu_mean = 0.0
            for ref in refs:
                sentence_bleu_mean += HF_BLEU.compute(
                    predictions=[pred],
                    references=[ref],
                )["bleu"]
            sentence_bleu_mean /= len(refs)
            sentence_bleu_max = max(
                [
                    HF_BLEU.compute(
                        predictions=[pred],
                        references=[ref],
                    )["bleu"]
                    for ref in refs
                ]
            )
            sentence_bleu = max([sentence_bleu_mean, sentence_bleu_max])
            bleus.append(sentence_bleu)
        elif mode == "sacre":
            sentence_bleu_mean = 0.0
            for ref in refs:
                sentence_bleu_mean += SACREBLEU.compute(
                    predictions=[pred],
                    references=[ref],
                )["score"]
            sentence_bleu_mean /= len(refs)
            sentence_bleu_max = max(
                [
                    SACREBLEU.compute(
                        predictions=[pred],
                        references=[ref],
                    )["score"]
                    for ref in refs
                ]
            )
            sentence_bleu = max([sentence_bleu_mean, sentence_bleu_max])
            bleus.append(sentence_bleu)
        elif mode == "nltk":
            # prepare tokens list for predicted definition
            pred_tokens = pred.split()
            # prepare tokens list for reference definitions
            ref_tokens_list = (
                [ref.split()] if one_to_one else [ref.split() for ref in refs]
            )
            # if policy == "mean":
            sentence_bleu_mean = bleu_score.sentence_bleu(
                ref_tokens_list,
                pred_tokens,
                smoothing_function=bleu_score.SmoothingFunction().method2,
                auto_reweigh=False if len(pred_tokens) == 0 else True,
            )
            # elif policy == "max":
            sentence_bleu_max = max(
                [
                    bleu_score.sentence_bleu(
                        [ref_tokens],
                        pred_tokens,
                        smoothing_function=bleu_score.SmoothingFunction().method2,
                        auto_reweigh=False if len(pred_tokens) == 0 else True,
                    )
                    for ref_tokens in ref_tokens_list
                ]
            )
            sentence_bleu = max([sentence_bleu_mean, sentence_bleu_max])
            bleus.append(sentence_bleu)
        elif mode == "cpp":
            if isinstance(pred, str):
                preds = [pred]
            else:
                preds = pred
            for pred in preds:
                with open(os.devnull, "w") as devnull:
                    failed_times = 0
                    word_ref_paths = word2ref_paths[word]
                    while failed_times < 10:
                        rp = Popen(["echo", pred], stdout=PIPE)
                        # if policy == "mean":
                        bp = Popen(
                            ["artifact/sentence-bleu"] + word_ref_paths,
                            stdin=rp.stdout,
                            stdout=PIPE,
                            stderr=devnull,
                        )
                        # rp.stdout.close()
                        out, err = bp.communicate()
                        bp.wait()
                        bp.stdout.close()
                        rp.stdout.close()
                        out = out.decode().strip()
                        if out == "":
                            console.log(
                                f"[red]Error: Empty output for {word}: {pred}![/red]"
                            )
                            out = "-1.0"
                            failed_times += 1
                        else:
                            break
                    sentence_bleu_mean = (
                        float(out)
                        # if is_float(out)
                        # else compute_sentence_bleu(
                        # pred,
                        # ref,
                        # word=word,
                        # refs=refs,
                        # mode="nltk",
                        # )  # use sentence-bleu-nltk as backup eval scheme
                    )
                    # elif policy == "max":
                    sentence_bleus = []
                    for ref_path in word_ref_paths:
                        failed_times = 0
                        while failed_times < 10:
                            rp = Popen(["echo", pred], stdout=PIPE)
                            bp = Popen(
                                ["artifact/sentence-bleu", ref_path],
                                stdin=rp.stdout,
                                stdout=PIPE,
                                stderr=devnull,
                            )
                            out, err = bp.communicate()
                            bp.wait()
                            bp.stdout.close()
                            rp.stdout.close()
                            out = out.decode().strip()
                            if out == "":
                                console.log(
                                    f"[red]Error: Empty output for {word}: {pred}![/red]"
                                )
                                out = "-1.0"
                                failed_times += 1
                            else:
                                break
                        sentence_bleus.append(
                            float(out)
                            # if is_float(out)
                            # else compute_sentence_bleu(
                            # pred,
                            # ref,
                            # word=word,
                            # refs=refs,
                            # mode="nltk",
                            # )  # use sentence-bleu-nltk as backup eval scheme
                        )
                        # console.print(out.decode().strip())
                    sentence_bleu_max = max(sentence_bleus)
                    sentence_bleu = max([sentence_bleu_mean, sentence_bleu_max])
                    bleus.append(sentence_bleu)

    sentence_bleu = max(bleus) if len(bleus) > 0 else 0.0

    if mode != "sacre":
        bleus = [round(bleu * 100, 2) for bleu in bleus]
        sentence_bleu = round(sentence_bleu * 100, 2)
    else:
        bleus = [round(bleu, 2) for bleu in bleus]
        sentence_bleu = round(sentence_bleu, 2)

    # console.log(f"BLEU ({mode},max):", sentence_bleu)

    # return (sentence_bleu, bleus)
    return sentence_bleu


def compute_mover(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    ngram: int = 1,
    policy: str = "max",
) -> float:
    """computes sentence-level MoverScore

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.
        ngram (int, optional): n-gram order. Defaults to 1.
        policy (str, optional): aggregation policy. Defaults to "max".

    Returns:
        float: average sentence-level MoverScore
    """
    idf_dict_preds = defaultdict(lambda: 1.0)
    idf_dict_refs = defaultdict(lambda: 1.0)

    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    # assert len(words) == len(preds) == len(refs), (
    # f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
    # f"{len(words)}, {len(preds)}, {len(refs)}"
    # )

    movers = []
    scores = word_mover_score(
        refs=refs,
        hyps=preds,
        idf_dict_ref=idf_dict_refs,
        idf_dict_hyp=idf_dict_preds,
        stop_words=[],
        n_gram=ngram,
        remove_subwords=True,
        batch_size=32,
        device="cuda:0",
    )

    movers += scores

    sentence_mover_score = mean(movers)
    # debug the error that sentence_mover_score is NaN
    if math.isnan(sentence_mover_score):
        print("Error: sentence_mover_score is NaN")
        sentence_mover_score = 0
    else:
        sentence_mover_score = sentence_mover_score

    sentence_mover_score = math.ceil(sentence_mover_score * 10000) / 100

    return sentence_mover_score


def compute_mauve(
    preds: List[str],
    refs: List[str],
) -> float:
    """computes sentence-level MAUVE score

    Args:
    - preds (List[str]): list of predicted sentences
    - refs (List[str]): list of reference sentences

    Returns:
    - float: sentence-level MAUVE score

    https://krishnap25.github.io/mauve/
    """
    try:
        mauve_score = MAUVE_SCORER.compute(
            predictions=preds,
            references=refs,
            num_buckets=int(len(preds) * 0.01) if len(preds) > 100 else "auto",
            featurize_model_name="gpt2-large",  # gpt2, gpt2-base, gpt2-large, gpt2-xl, gpt2-xxl
            max_text_length=128,
            device_id=0,
            seed=84,
            mauve_scaling_factor=2,  # to tune
            verbose=False,
        ).mauve
    except ValueError as _:
        mauve_score = 0.0
    except RuntimeError as _:
        mauve_score = 0.0
    # mauve_score = math.ceil(mauve_score * 10000) / 100

    return mauve_score


def get_word2refs(
    dataset: Optional[Dataset] = None,
    words: Optional[List[str]] = None,
    refs: Optional[List[str]] = None,
    dedup_refs: bool = False,
    cache_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    """creates a dictionary of words to references

    Args:
        dataset (Dataset): dataset
        dedup_refs (bool, optional): deduplicate references. Defaults to False.
        cache_path (Optional[str], optional): cache path. Defaults to None.

    Returns:
        Dict[str, List[str]]: dictionary of words to references
    """
    # load and return the cache if it exists
    if cache_path is not None and os.path.exists(cache_path):
        console.log(f"Loading word2refs from {cache_path}")
        with open(cache_path, "r") as f:
            word2refs = json.load(f)
        return word2refs

    # construct words and refs lists if not provided
    if words is None and refs is None:
        words, refs = [], []
    if dataset is not None:
        for sample in dataset:
            words.append(sample["term"])
            refs.append(sample["definition"])

    # construct a dictionary of words to references
    seen_refs = set()
    word2refs = dict()
    for word, ref in zip(words, refs):
        if word not in word2refs:
            word2refs[word] = []
        if dedup_refs and word + ref not in seen_refs:
            # NOTE: deduplicating references seems to be more reasonable
            seen_refs.add(word + ref)
            word2refs[word].append(ref)
        else:
            # NOTE: follow Huang et al. (2021) to use un-deduplicated references to keep consistency
            word2refs[word].append(ref)

    # save the cache if cache_path is provided
    if cache_path is not None:
        if not os.path.exists(cache_path):
            with open(cache_path, "w") as f:
                json.dump(word2refs, f, indent=4, ensure_ascii=False)

    return word2refs


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


if __name__ == "__main__":
    input_path = sys.argv[1]
    scoring_functions = [compute_sentence_bleu, compute_rouge, compute_meteor]
    with open(input_path, "r") as f:
        # lines = [l.strip() for l in f.readlines()]
        lines = json.load(f)
        for scoring_func in scoring_functions:
            for mode in ["hf", "sacre", "nltk", "cpp"]:
                scores = []
                # for line in lines[1:]:
                # word, pred, ref = line.split("\t")
                for line in lines:
                    word, pred, ref = (
                        line["word"],
                        line["prediction"],
                        line["definition"],
                    )
                    score = scoring_func(pred, ref, word, mode=mode, dataset="3D-EX")
                    scores.append(score)
                # print(scores)
                mean_value = mean(scores)
                mean_value = (
                    round(mean_value * 100, 3)
                    if mean_value < 1
                    else round(mean_value, 3)
                )
                print(scoring_func.__name__)
                print(compute_sentence_bleu.__name__)
                if scoring_func.__name__ == compute_sentence_bleu.__name__:
                    if mode != "sacre":
                        console.print(f"Mean BLEU ({mode}):", mean_value)
                    else:
                        console.print(f"Mean BLEU ({mode}):", mean_value)
                elif scoring_func.__name__ == compute_rouge.__name__:
                    console.print(f"Mean ROUGE:", mean_value)
                    break
                elif scoring_func.__name__ == compute_meteor.__name__:
                    console.print(f"Mean METEOR:", mean_value)
                    break
