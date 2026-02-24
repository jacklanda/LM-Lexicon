# -*- coding: utf-8 -*-
#
# This script is a singleton part of the evaluation toolkit for LM-Lexicon.
# It can be run independently to evaluate the performance of the models.

import os
import re
import time
import json
import math
import random
import argparse
import warnings
import numpy as np
import transformers
import multiprocess as mp
from statistics import mean
from functools import partial
from itertools import zip_longest
from subprocess import Popen, PIPE
from collections import defaultdict
from multiprocessing import Manager, Pool
from concurrent.futures import TimeoutError
from typing import Dict, List, Optional, Union, Iterable

import GPUtil
import sacrebleu
from evaluate import load
from datasets import Dataset
from pebble import ProcessPool
from rich.console import Console
from moverscore_v2 import word_mover_score
from nltk.translate import bleu_score, nist_score


# set up logging level
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# set up rich console
console = Console()

# set up fixed seed
random.seed(42)


def compute_sentence_bleu_nltk(
    word: str,
    pred: str,
    ref: str,
    word2refs: Dict[str, List[str]],
    one_to_one: bool = False,
) -> float:
    # prepare tokens list for predicted definition
    pred_tokens = pred.split()
    # prepare tokens list for reference definitions
    ref_tokens_list = (
        [ref.split()] if one_to_one else [ref.split() for ref in word2refs[word]]
    )
    try:
        # use one pred to many refs like Huang et al. (2021)
        bleu = bleu_score.sentence_bleu(
            ref_tokens_list,  # [[tok1, tok2, ...]] / [[tok1, tok2, ...], ...]
            pred_tokens,  # [tok1, tok2, ...]
            smoothing_function=bleu_score.SmoothingFunction().method2,
            auto_reweigh=False if len(pred_tokens) == 0 else True,
        )
    except Exception as _:
        bleu = 0.0

    return bleu


def compute_sentence_bleu_cpp(
    word: str,
    pred: str,
    ref: str,
    word2ref_paths: List[str],
    word2refs: Dict[str, List[str]],
) -> float:
    with open(os.devnull, "w") as devnull:
        word_ref_paths = word2ref_paths[word]
        rp = Popen(["echo", pred], stdout=PIPE)
        bp = Popen(
            ["artifact/sentence-bleu"] + word_ref_paths,
            stdin=rp.stdout,
            stdout=PIPE,
            stderr=devnull,
        )
        rp.stdout.close()
        out, err = bp.communicate()
        bp.wait()
        bp.stdout.close()
        out = str(out, encoding="utf-8").strip()
        bleu = (
            float(out)
            if is_float(out)
            else compute_sentence_bleu(
                [pred],
                [ref],
                mode="nltk",
                words=[word],
                word2refs=word2refs,
                n_workers=1,
            )  # use sentence-bleu-nltk as backup eval scheme
        )

    return bleu


def compute_sentence_bleu(
    preds: List[str],
    refs: List[str],
    mode: str = "nltk",
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    debug: bool = False,
    n_workers: int = 32,
) -> float:
    """computes sentence-level BLEU score

    The evaluation of LM-Lexicon follows five of the existing work:
    1. Learning to Describe Unknown Phrases with Local and Global Contexts, Ishiwatari (2019)
      - https://github.com/shonosuke/ishiwatari-naacl2019
    2. Definition Modelling for Appropriate Specificity, Huang et al. (2021)
      - https://github.com/amanotaiga/Definition_Modeling_Project
    3. Multitasking Framework for Unsupervised Simple Definition Generation, Kong et al. (2022)
      - https://github.com/blcuicall/SimpDefiner
    4. Fine-grained Contrastive Learning for Definition Generation, Zhang et al. (2022)
      - https://github.com/rattlesnakey/Definition-Gneration-Contrastive
    5. Interpretable Word Sense Representations via Definition Generation: The Case of Semantic Change Analysis, Giulianelli et al. (2023)
      - https://github.com/ltgoslo/definition_modeling

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        mode (str, optional): mode of BLEU computation. Defaults to "nltk".
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.
        debug (bool, optional): debug mode. Defaults to False.
        n_workers (int, optional): number of workers. Defaults to 32.

    Returns:
        float: sentence-level BLEU score
    """

    def task_done(future, bleus: List[float]):
        try:
            bleu = future.result()
            bleus.append(bleu)
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised: %s" % error)

    time_start = get_time()

    bleus = []
    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    # prepare reference files for sentence-bleu.cpp
    seen_refs = set()
    word2ref_paths = dict()
    ref_path_prefix = "/tmp/ref"
    for _, (word, ref) in enumerate(zip(words, refs)):
        timecode = int(get_time() * 1000) + random.randint(0, 100000)
        ref_path = ref_path_prefix + "-" + str(timecode) + ".txt"
        with open(ref_path, "w") as f:
            f.write(ref + "\n")
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

    # prepare word to references mapping for sentence-bleu-nltk
    word2refs = (
        word2refs
        if word2refs is not None
        else get_word2refs(words=words, refs=refs, dedup_refs=dedup_refs)
    )

    if n_workers > 1:
        # parallel mode
        with ProcessPool(max_workers=n_workers, max_tasks=n_workers) as pool:
            for i, (word, pred, ref) in enumerate(zip(words, preds, refs)):
                if mode == "nltk":
                    future = pool.schedule(
                        compute_sentence_bleu_nltk,
                        (
                            word,
                            pred,
                            ref,
                            word2refs,
                        ),
                        timeout=60,  # 1 mins
                    )
                elif mode == "cpp":
                    future = pool.schedule(
                        compute_sentence_bleu_cpp,
                        (
                            word,
                            pred,
                            ref,
                            word2ref_paths,
                            word2refs,
                        ),
                        timeout=60,  # 1 mins
                    )
                callback = partial(task_done, bleus=bleus)
                future.add_done_callback(callback)
    else:
        for idx, (word, pred, ref) in enumerate(zip(words, preds, refs)):
            if mode == "nltk":  # 3 to 5 points lower than sentence_bleu.cpp
                bleu = compute_sentence_bleu_nltk(word, pred, ref, word2refs)
            elif mode == "cpp":
                bleu = compute_sentence_bleu_cpp(
                    word, pred, ref, word2ref_paths, word2refs
                )
            else:
                raise ValueError("Invalid mode. Choose either 'nltk' or 'cpp'.")

            bleus.append(bleu)

    time_end = get_time()

    return math_round(mean(bleus)), math_round(
        time_end - time_start, in_percentile=False
    ) if debug else math_round(mean(bleus))


def compute_nist(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    policy: str = "max",
    debug: bool = False,
    n_workers: int = 32,
) -> float:
    """computes sentence-level NIST score

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.
        policy (str, optional): aggregation policy. Defaults to "max".
        debug (bool, optional): debug mode. Defaults to False.
        n_workers (int, optional): number of workers. Defaults to 32.

    Returns:
        float: average sentence-level NIST score
    """

    def compute_nist_fn(preds, refs, n):
        try:
            score = nist_score.sentence_nist(refs, preds, n=n)
        except Exception as _:
            score = 0.0
        return score

    def task_done(result, nists: List[float]):
        try:
            nists.append(result)
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised: %s" % error)

    time_start = get_time()

    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    # prepare word to references mapping for sentence-nist
    seen_refs = set()
    word2refs = (
        word2refs
        if word2refs is not None
        else get_word2refs(words=words, refs=refs, dedup_refs=dedup_refs)
    )

    # compute nist score for each sample <word, prediction, reference(s)>
    nists = []
    if n_workers > 1:
        # parallel mode
        ref_list, pred_list = [], []
        for word, pred, ref in zip(words, preds, refs):
            sub_ref_list = (
                [ref.split()]
                if one_to_one
                else [ref.split() for ref in word2refs[word]]
            )
            sub_pred_list = pred.split()
            ref_list.append(sub_ref_list)
            pred_list.append(sub_pred_list)
        assert len(ref_list) == len(pred_list), (
            f"Length mismatch between `refs` and `preds` in evaluation: "
            f"{len(ref_list)}, {len(pred_list)}"
        )
        chunk_size = 1
        with mp.get_context("spawn").Pool(n_workers) as pool:
            for preds, refs in get_chunks(refs=ref_list, preds=pred_list, n=chunk_size):
                task = pool.apply_async(
                    compute_nist_fn,
                    (
                        preds[0],
                        refs[0],
                        5,
                    ),
                    callback=partial(task_done, nists=nists),
                )
                task.get()
    else:
        for word, pred, ref in zip(words, preds, refs):
            pred_tokens = pred.split()
            ref_tokens_list = (
                [ref.split()]
                if one_to_one
                else [ref.split() for ref in word2refs[word]]
            )
            n = len(pred_tokens) if len(pred_tokens) < 5 else 5
            nist = compute_nist_fn(pred_tokens, ref_tokens_list, n)
            nists.append(nist)

    time_end = get_time()

    return math_round(mean(nists)), math_round(
        time_end - time_start, in_percentile=False
    ) if debug else math_round(mean(nists))


def compute_rouge(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    metric_key: str = "rougeL",
    debug: bool = False,
    n_workers: int = 32,
) -> float:
    """computes sentence-level ROUGE score

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.
        metric_key (str, optional): metric key. Defaults to "rougeL".
        debug (bool, optional): debug mode. Defaults to False.
        n_workers (int, optional): number of workers. Defaults to 32.

    Returns:
        float: average sentence-level ROUGE score
    """

    def compute_rouge_fn(preds, refs, metric_key):
        return load("rouge")._compute(predictions=preds, references=refs)[metric_key]

    def task_done(result, rouge_scores: List[float]):
        try:
            rouge_scores.append(result)
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised: %s" % error)

    time_start = get_time()

    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    # prepare word to references mapping for sentence-rouge
    word2refs = (
        word2refs
        if word2refs is not None
        else get_word2refs(words=words, refs=refs, dedup_refs=dedup_refs)
    )

    # compute rouge score for each sample <word, prediction, reference(s)>
    # TODO: retrieve the max rouge value for each sample
    rouge_scores = []
    if n_workers > 1:
        # parallel mode
        ref_list, pred_list = [], []
        for word, pred, ref in zip(words, preds, refs):
            sub_ref_list = [ref] if one_to_one else word2refs[word]
            sub_pred_list = [pred]
            ref_list.append(sub_ref_list)
            pred_list += sub_pred_list
        assert len(ref_list) == len(pred_list), (
            f"Length mismatch between `refs` and `preds` in evaluation: "
            f"{len(ref_list)}, {len(pred_list)}"
        )
        chunk_size = (
            (len(ref_list) // n_workers) + 1 if len(ref_list) // n_workers > 0 else 1
        )
        with mp.get_context("spawn").Pool(n_workers) as pool:
            for preds, refs in get_chunks(refs=ref_list, preds=pred_list, n=chunk_size):
                task = pool.apply_async(
                    compute_rouge_fn,
                    (
                        preds,
                        refs,
                        metric_key,
                    ),
                    callback=partial(task_done, rouge_scores=rouge_scores),
                )
                task.get()
    else:
        rouge_scorer = load("rouge")
        for _, (word, pred, ref) in enumerate(zip(words, preds, refs)):
            pred_list = [pred]
            ref_list = [ref] if one_to_one else [word2refs[word]]
            rouge_score = compute_rouge_fn(pred_list, ref_list, metric_key)
            rouge_scores.append(rouge_score)

    time_end = get_time()

    return math_round(mean(rouge_scores)), math_round(
        time_end - time_start, in_percentile=False
    ) if debug else math_round(mean(rouge_scores))


def compute_moverscore(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    ngram: int = 1,
    policy: str = "mean",
    debug: bool = False,
    n_workers: int = 4,
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
        debug (bool, optional): debug mode. Defaults to False.
        n_workers (int, optional): number of workers. Defaults to 1.

    Returns:
        float: average sentence-level MoverScore
    """

    def task_done(results, scores: List[float] = None):
        try:
            scores += results
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised %s" % error)

    time_start = get_time()

    idf_dict_preds = defaultdict(lambda: 1.0)
    idf_dict_refs = defaultdict(lambda: 1.0)

    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    word2refs = (
        word2refs
        if word2refs is not None
        else get_word2refs(words=words, refs=refs, dedup_refs=dedup_refs)
    )

    scores = []
    ref_list, pred_list, span_list = [], [], []
    for _, (word, pred, ref) in enumerate(zip(words, preds, refs)):
        word_ref_list = [ref] if one_to_one else word2refs[word]
        ref_list += word_ref_list
        word_pred_list = [pred] * len(word_ref_list)
        pred_list += word_pred_list
        last_span_end = span_list[-1][1] if len(span_list) > 0 else 0
        span_list.append((last_span_end, last_span_end + len(word_ref_list)))

    if n_workers > 1:
        # parallelize mode
        with mp.get_context("spawn").Pool(n_workers) as pool:
            # devide refs into balanced chunks of len(refs) // n_workers
            chunk_size = (
                (len(ref_list) // n_workers) + 1
                if len(ref_list) // n_workers > 0
                else 1
            )
            for i, (preds, refs) in enumerate(
                get_chunks(preds=pred_list, refs=ref_list, n=chunk_size)
            ):
                task = pool.apply_async(
                    word_mover_score,
                    (
                        refs,
                        preds,
                        idf_dict_refs,
                        idf_dict_preds,
                        [],
                        ngram,
                        True,
                        32,
                        f"cuda:{i % 4}",
                    ),
                    callback=partial(task_done, scores=scores),
                )
                task.get()
    else:
        # sequential mode
        device = GPUtil.getGPUs()[2]
        scores = word_mover_score(
            refs=ref_list,
            hyps=pred_list,
            idf_dict_ref=idf_dict_refs,
            idf_dict_hyp=idf_dict_preds,
            stop_words=[],
            n_gram=ngram,
            remove_subwords=True,
            batch_size=32
            if round(device.memoryFree / device.memoryTotal, 2) < 0.3
            else 64,
            device="cuda:3",
        )

    if policy == "max":
        # compute max value for each sub span in span_list indexing scores
        raise NotImplementedError("MoverScore does not support max policy.")
    elif policy == "mean":
        sentence_mover_score = mean(scores)
    else:
        raise ValueError("Invalid policy. Choose either 'max' or 'mean'.")

    time_end = get_time()

    return math_round(sentence_mover_score), math_round(
        time_end - time_start, in_percentile=False
    ) if debug else math_round(sentence_mover_score)


def compute_bert_score(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    metric_key: Optional[str] = None,
    policy: str = "max",
    debug: bool = False,
    n_workers: int = 4,
) -> float:
    """computes sentence-level BERTScore

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.
        metric_key (Optional[str], optional): metric key. Defaults to None.
        policy (str, optional): aggregation policy. Defaults to "max".
        debug (bool, optional): debug mode. Defaults to False.
        n_workers (int, optional): number of workers. Defaults to 4.

    Returns:
        float: average sentence-level BERTScore
    """

    def compute_bert_score_fn(preds, refs, lang, nthreads, device, batch_size):
        return load("bertscore")._compute(
            predictions=preds,
            references=refs,
            lang=lang,
            nthreads=nthreads,
            device=device,
            batch_size=batch_size,
        )

    def task_done(results, bert_scores: Dict[str, List[float]] = None):
        try:
            for key in ["precision", "recall", "f1"]:
                bert_scores[key] += results[key]
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised %s" % error)

    time_start = get_time()

    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    word2refs = (
        word2refs
        if word2refs is not None
        else get_word2refs(words=words, refs=refs, dedup_refs=dedup_refs)
    )

    bert_scores = {"precision": [], "recall": [], "f1": []}
    if n_workers > 1:
        # parallel mode
        ref_list, pred_list = [], []
        for _, (word, ref, pred) in enumerate(zip(words, refs, preds)):
            sub_ref_list = [ref] if one_to_one else word2refs[word]
            sub_pred_list = [pred] * len(sub_ref_list)
            ref_list += sub_ref_list
            pred_list += sub_pred_list
        assert len(ref_list) == len(pred_list), (
            f"Length mismatch between `refs` and `preds` in evaluation: "
            f"{len(ref_list)}, {len(pred_list)}"
        )
        # devide refs into balanced chunks of len(ref_list) // n_workers
        chunk_size = (
            (len(ref_list) // n_workers) + 1 if len(ref_list) // n_workers > 0 else 1
        )
        with mp.get_context("spawn").Pool(n_workers) as pool:
            for i, (preds, refs) in enumerate(
                get_chunks(refs=ref_list, preds=pred_list, n=chunk_size)
            ):
                task = pool.apply_async(
                    compute_bert_score_fn,
                    (
                        preds,
                        refs,
                        "en",
                        32,
                        f"cuda:{i % 4}",
                        64,
                    ),
                    callback=partial(task_done, bert_scores=bert_scores),
                )
                task.get()
    else:
        bert_scorer = load("bertscore")
        for _, (word, pred, ref) in enumerate(zip(words, preds, refs)):
            ref_list = [ref] if one_to_one else word2refs[word]
            pred_list = [pred] * len(ref_list)
            bert_score = compute_bert_score_fn(
                preds=pred_list,
                refs=ref_list,
                lang="en",
                nthreads=32,
                device="cuda:2",
                batch_size=64,
            )
            for key in ["precision", "recall", "f1"]:
                if policy == "max":
                    bert_scores[key] += [max(bert_score[key])]
                elif policy == "mean":
                    bert_scores[key] += [mean(bert_score[key])]
                else:
                    bert_scores[key] += bert_score[key]

    bert_scores = {k: math_round(mean(v)) for k, v in bert_scores.items()}

    time_end = get_time()

    return (
        (
            bert_scores[metric_key]
            if metric_key is not None and metric_key != "all"
            else bert_scores,
            math_round(time_end - time_start, in_percentile=False),
        )
        if debug
        else (
            bert_scores[metric_key]
            if metric_key is not None and metric_key != "all"
            else bert_scores
        )
    )


def compute_meteor(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    debug: bool = False,
    n_workers: int = 32,
) -> float:
    """computes sentence-level METEOR score

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.
        debug (bool, optional): debug mode. Defaults to False.
        n_workers (int, optional): number of workers. Defaults to 8.

    Returns:
        float: average sentence-level METEOR score
    """

    def compute_meteor_fn(preds, refs):
        return load("meteor")._compute(predictions=preds, references=refs)["meteor"]

    def task_done(result, meteors: List[float]):
        try:
            meteors.append(result)
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised: %s" % error)

    time_start = get_time()

    meteor_scorer = load("meteor")

    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    word2refs = (
        word2refs
        if word2refs is not None
        else get_word2refs(words=words, refs=refs, dedup_refs=dedup_refs)
    )

    meteors = []
    if n_workers > 1:
        # parallel mode
        ref_list, pred_list = [], []
        for word, ref, pred in zip(words, refs, preds):
            sub_ref_list = [ref] if one_to_one else word2refs[word]
            sub_pred_list = [pred]
            ref_list.append(sub_ref_list)
            pred_list += sub_pred_list
        assert len(ref_list) == len(pred_list), (
            f"Length mismatch between `refs` and `preds` in evaluation: "
            f"{len(ref_list)}, {len(pred_list)}"
        )
        chunk_size = (
            (len(ref_list) // n_workers) + 1 if len(ref_list) // n_workers > 0 else 1
        )
        with mp.get_context("spawn").Pool(n_workers) as pool:
            for preds, refs in get_chunks(preds=pred_list, refs=ref_list, n=chunk_size):
                task = pool.apply_async(
                    compute_meteor_fn,
                    (
                        preds,
                        refs,
                    ),
                    callback=partial(task_done, meteors=meteors),
                )
                task.get()
    else:
        # sequential mode
        for _, (word, pred, ref) in enumerate(zip(words, preds, refs)):
            ref_list = [ref] if one_to_one else [word2refs[word]]  # [str] or [[str]]
            pred_list = [pred]  # [str]
            meteor = compute_meteor_fn(pred_list, ref_list)
            meteors.append(meteor)

    time_end = get_time()

    return math_round(mean(meteors)), math_round(
        time_end - time_start, in_percentile=False
    ) if debug else math_round(mean(meteors))


def compute_exact_match(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    debug: bool = False,
    n_workers: int = 32,
) -> float:
    """computes sentence-level Exact Match score

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.
        debug (bool, optional): debug mode. Defaults to False.
        n_workers (int, optional): number of workers. Defaults to 32.

    Returns:
        float: average sentence-level Exact Match score
    """

    time_start = get_time()

    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    word2refs = (
        word2refs
        if word2refs is not None
        else get_word2refs(words=words, refs=refs, dedup_refs=dedup_refs)
    )

    exact_matches = []
    exact_match_scorer = load("exact_match")
    for _, (word, pred, ref) in enumerate(zip(words, preds, refs)):
        pred_list = [pred]  # List[str]
        ref_list = [ref] if one_to_one else word2refs[word]  # List[str]
        for i in range(len(ref_list)):
            exact_match = exact_match_scorer.compute(
                predictions=pred_list,
                references=[ref_list[i]],
            )["exact_match"]
            if exact_match == 1.0:
                exact_matches.append(exact_match)
                break
            elif i == len(ref_list) - 1:
                exact_matches.append(exact_match)

    time_end = get_time()

    return math_round(mean(exact_matches)), math_round(
        time_end - time_start, in_percentile=False
    ) if debug else math_round(mean(exact_matches))


def compute_sari(
    sources: List[str],
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    debug: bool = False,
    n_workers: int = 32,
) -> float:
    """computes sentence-level SARI score

    Args:
        sources (List[str]): list of source sentences
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.
        debug (bool, optional): debug mode. Defaults to False.
        n_workers (int, optional): number of workers. Defaults to 32.

    Returns:
        float: average sentence-level SARI score
    """

    def compute_sari_fn(sources, preds, refs):
        return load("sari")._compute(
            sources=sources,
            predictions=preds,
            references=refs,
        )["sari"]

    def task_done(result, saris: List[float]):
        try:
            saris.append(result)
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised: %s" % error)

    time_start = get_time()

    sources, preds, refs = (
        get_rid_of_period(sources),
        get_rid_of_period(preds),
        get_rid_of_period(refs),
    )

    assert len(words) == len(sources) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `sources`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(sources)}, {len(preds)}, {len(refs)}"
    )

    word2refs = (
        word2refs
        if word2refs is not None
        else get_word2refs(words=words, refs=refs, dedup_refs=dedup_refs)
    )

    saris = []
    if n_workers > 1:
        # parallel mode
        ref_list, pred_list, source_list = [], [], []
        for word, source, pred, ref in zip(words, sources, preds, refs):
            sub_ref_list = [ref] if one_to_one else word2refs[word]
            sub_pred_list = [pred]
            ref_list.append(sub_ref_list)
            pred_list += sub_pred_list
            source_list += [source]
        assert len(ref_list) == len(pred_list) == len(source_list), (
            f"Length mismatch between `refs`, `preds`, and `sources` in evaluation: "
            f"{len(ref_list)}, {len(pred_list)}, {len(source_list)}"
        )
        chunk_size = (
            (len(ref_list) // n_workers) + 1 if len(ref_list) // n_workers > 0 else 1
        )
        with mp.get_context("spawn").Pool(n_workers) as pool:
            for sources, preds, refs in get_chunks(
                words=source_list, preds=pred_list, refs=ref_list, n=chunk_size
            ):
                task = pool.apply_async(
                    compute_sari_fn,
                    (
                        sources,
                        preds,
                        refs,
                    ),
                    callback=partial(task_done, saris=saris),
                )
                task.get()
    else:
        for word, source, pred, ref in zip(words, sources, preds, refs):
            ref_list = [[ref]] if one_to_one else [word2refs[word]]
            pred_list = [pred]
            source_list = [source]
            sari = compute_sari_fn(source_list, pred_list, ref_list)
            saris.append(sari)

    time_end = get_time()

    return math_round(mean(saris), in_percentile=False), math_round(
        time_end - time_start, in_percentile=False
    ) if debug else math_round(mean(saris), in_percentile=False)


def compute_mauve(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    base_model: bool = "gpt2-large",
    debug: bool = False,
    n_workers: int = 4,
) -> float:
    """computes sentence-level Mauve score

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.
        base_model (str, optional): base model. Defaults to "gpt2-large".
            Options: gpt2, gpt2-base, gpt2-large, gpt2-xl, gpt2-xxl
        debug (bool, optional): debug mode. Defaults to False.
        n_workers (int, optional): number of workers. Defaults to 1.

    Returns:
        float: average sentence-level Mauve score
    """

    def compute_mauve_fn(
        preds, refs, base_model, max_text_length, device_id, scaling_factor
    ):
        return (
            load("mauve")
            ._compute(
                predictions=preds,
                references=refs,
                num_buckets=int(len(preds) * 0.01) if len(preds) > 100 else "auto",
                featurize_model_name=base_model,
                max_text_length=max_text_length,
                device_id=device_id,
                seed=42,
                mauve_scaling_factor=scaling_factor,
                verbose=False,
            )
            .mauve
        )

    def task_done(result, mauves: List[float]):
        try:
            mauves.append(result)
        except TimeoutError as error:
            print("Function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("Function raised: %s" % error)

    time_start = get_time()

    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    word2refs = (
        word2refs
        if word2refs is not None
        else get_word2refs(words=words, refs=refs, dedup_refs=dedup_refs)
    )

    mauves = []
    if n_workers > 1:
        # parallel mode
        chunk_size = (len(preds) // n_workers) + 1 if len(preds) // n_workers > 0 else 1
        with mp.get_context("spawn").Pool(n_workers) as pool:
            for i, (ref_list, pred_list) in enumerate(
                get_chunks(preds=preds, refs=refs, n=chunk_size)
            ):
                task = pool.apply_async(
                    compute_mauve_fn,
                    (
                        pred_list,
                        ref_list,
                        base_model,
                        128,
                        i % 4,
                        16,
                    ),
                    callback=partial(task_done, mauves=mauves),
                )
                task.get()
    else:
        mauves = compute_mauve_fn(
            preds=preds,
            refs=refs,
            base_model=base_model,
            max_text_length=128,
            device_id=2,
            scaling_factor=16,
        )

    mauve_score = math_round(mean(mauves))

    time_end = get_time()

    return (
        mauve_score if not debug else mauve_score,
        math_round(time_end - time_start, in_percentile=False)
        if debug
        else mauve_score,
    )


def batch_eval(data: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, float]:
    """batch evaluation

    Args:
        data (List[Dict[str, Union[str, List[int]]]): list of dictionaries

    Returns:
        Dict[str, float]: evaluation results
    """
    # prepare data
    words = [
        d["inp"].split("word: ", maxsplit=1)[-1].split("%", maxsplit=1)[0].strip()
        for d in data
    ]
    preds = [d["pred"].strip() for d in data]
    refs = [d["tgt"] for d in data]

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    word2refs = get_word2refs(words=words, refs=refs, dedup_refs=True)

    with console.status("Evaluating ...") as _:
        # compute evaluation metrics
        bleu_nltk, time = compute_sentence_bleu(
            preds,
            refs,
            words=words,
            mode="nltk",
            word2refs=word2refs,
            debug=True,
            n_workers=32,
        )
        console.log(f"bleu-nltk ({time} sec.): {bleu_nltk}%")
        bleu_cpp, time = compute_sentence_bleu(
            preds,
            refs,
            words=words,
            mode="cpp",
            word2refs=word2refs,
            debug=True,
            n_workers=32,
        )
        console.log(f"bleu-cpp ({time} sec.): {bleu_cpp}%")
        rougeL, time = compute_rouge(
            preds,
            refs,
            words=words,
            word2refs=word2refs,
            debug=True,
            n_workers=4,
        )
        console.log(f"rougeL ({time} sec.): {rougeL}%")
        meteor, time = compute_meteor(
            preds,
            refs,
            words=words,
            word2refs=word2refs,
            debug=True,
            n_workers=4,
        )
        console.log(f"meteor ({time} sec.): {meteor}%")
        bertscore, time = compute_bert_score(
            preds,
            refs,
            words=words,
            word2refs=word2refs,
            debug=True,
            n_workers=4,
        )
        console.log(f"bertscore ({time} sec.): {bertscore}%")
        mover, time = compute_moverscore(
            preds,
            refs,
            words=words,
            word2refs=word2refs,
            debug=True,
            n_workers=4,
        )
        console.log(f"moverscore ({time} sec.): {mover}%")
        mauve, time = compute_mauve(
            preds,
            refs,
            words=words,
            word2refs=word2refs,
            debug=True,
            n_workers=4,
        )
        console.log(f"mauve ({time} sec.): {mauve}%")
        exact_match, time = compute_exact_match(
            preds,
            refs,
            words=words,
            word2refs=word2refs,
            debug=True,
            n_workers=4,
        )
        console.log(f"exact-match ({time} sec.): {exact_match}%")
        nist, time = compute_nist(
            preds,
            refs,
            words=words,
            word2refs=word2refs,
            debug=True,
            n_workers=4,
        )
        console.log(f"nist ({time} sec.): {nist}%")
        sari, time = compute_sari(
            preds,
            preds,
            refs,
            words=words,
            word2refs=word2refs,
            debug=True,
            n_workers=4,
        )
        console.log(f"sari ({time} sec.): {sari}%")

    return {
        "bleu-nltk": bleu_nltk,
        "bleu-cpp": bleu_cpp,
        "rouge-l": rougeL,
        "meteor": meteor,
        "bertscore": bertscore,
        "moverscore": mover,
        "mauve": mauve,
        "exact-match": exact_match,
        "nist": nist,
        "sari": sari,
    }


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


def get_time() -> float:
    return round(time.time(), 2)


def load_jsonl(file_path: str) -> List[Dict[str, Union[str, List[int]]]]:
    """loads a jsonl file

    Args:
        file_path (str): file path

    Returns:
        List[Dict[str, Union[str, List[str]]]: list of dictionaries
    """
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]

    return data


def dump_json(metrics: Dict[str, float], file_path: str) -> None:
    """dumps a list of dictionaries to a json file

    Args:
        metrics (Dict[str, float]): evaluation results
        file_path (str): file path
    """
    with open(file_path, "w") as f:
        f.write(json.dumps(metrics, ensure_ascii=False, indent=4))


def math_round(x: float, n: int = 2, in_percentile: bool = True) -> float:
    """rounds a float number to n decimal places.
    Example: 0.23456 -> 2345.6 -> 2346.0 -> 23.46

    Args:
        x (float): float number
        n (int, optional): number of decimal places. Defaults to 2.

    Returns:
        float: rounded float number
    """
    return (
        math.ceil(x * 10**n * 100) / 10**n
        if in_percentile
        else math.ceil(x * 10**n) / 10**n
    )


def get_chunks(
    words: Optional[List[str]] = None,
    preds: Optional[List[str]] = None,
    refs: Optional[List[str]] = None,
    n: int = 1,
) -> Iterable:
    """Yield successive n-sized slice from list."""
    # detect if "words" is provided
    if words is not None:
        assert len(words) == len(preds) == len(refs), (
            f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
            f"{len(words)}, {len(preds)}, {len(refs)}"
        )
        for i in range(0, len(words), n):
            yield words[i : i + n], preds[i : i + n], refs[i : i + n]
    elif refs is not None and preds is not None:
        assert len(preds) == len(refs), (
            f"Length mismatch between `preds` and `refs` in evaluation: "
            f"{len(preds)}, {len(refs)}"
        )
        for i in range(0, len(preds), n):
            yield preds[i : i + n], refs[i : i + n]
    else:
        raise ValueError("Invalid input. Provide either `words` or `refs` and `preds`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file_path", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, default=None)
    args = parser.parse_args()

    console.log(f"Loading data from {args.pred_file_path} ...")
    data = load_jsonl(args.pred_file_path)

    eval_results = batch_eval(data)

    if args.output_file_path is None:
        output_file_path = (
            args.pred_file_path.rsplit("/", 1)[0] + "/metrics_predictions.json"
        )
    console.log(f"Saving evaluation results to {output_file_path} ...")
    dump_json(eval_results, output_file_path)
