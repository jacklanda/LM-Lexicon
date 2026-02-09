import os
import re
import numpy as np
from statistics import mean
from itertools import zip_longest
from subprocess import Popen, PIPE
from collections import defaultdict
from typing import Dict, List, Optional, Union, Iterable

import sacrebleu
from evaluate import load
from moverscore_v2 import word_mover_score
from nltk.translate import bleu_score, nist_score

from utils import timeit


@timeit
def compute_sentence_bleu(
    preds: List[str],
    refs: List[str],
    mode: str = "nltk",
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
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

    Returns:
        float: sentence-level BLEU score
    """

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
    for i, (word, ref) in enumerate(zip(words, refs)):
        ref_path = ref_path_prefix + str(i) + ".txt"
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
        word2refs if word2refs is not None else get_word2refs(words, refs, dedup_refs)
    )

    for idx, (word, pred, ref) in enumerate(zip(words, preds, refs)):
        if mode == "nltk":  # 3 to 5 points lower than sentence_bleu.cpp
            # prepare tokens list for predicted definition
            pred_tokens = pred.split()
            # prepare tokens list for reference definitions
            ref_tokens_list = (
                [ref.split()]
                if one_to_one
                else [ref.split() for ref in word2refs[word]]
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
        elif mode == "cpp":
            # TODO: optimize the code efficiency with subprocess parallelism
            # especially for sentence-bleu-cpp in multiple refs case
            # sharding the refs and parallelizing the subprocesses
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
                    )  # use sentence-bleu-nltk as backup eval scheme
                )
        else:
            raise ValueError("Invalid mode. Choose either 'nltk' or 'cpp'.")

        bleus.append(bleu)

    return mean(bleus)


@timeit
def compute_nist(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    policy: str = "max",
) -> float:
    """computes sentence-level NIST score

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.

    Returns:
        float: average sentence-level NIST score
    """
    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    # prepare word to references mapping for sentence-nist
    seen_refs = set()
    word2refs = (
        word2refs if word2refs is not None else get_word2refs(words, refs, dedup_refs)
    )

    # compute nist score for each sample <word, prediction, reference(s)>
    nists = []
    for _, (word, pred, ref) in enumerate(zip(words, preds, refs)):
        pred_tokens = pred.split()
        ref_tokens_list = (
            [ref.split()] if one_to_one else [ref.split() for ref in word2refs[word]]
        )
        n = len(pred_tokens) if len(pred_tokens) < 5 else 5
        try:
            nist = nist_score.sentence_nist(ref_tokens_list, pred_tokens, n=n)
        except Exception as _:
            nist = 0.0

        nists.append(nist)

    return mean(nists)


@timeit
def compute_rouge(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    metric_key: str = "rougeL",
) -> float:
    """computes sentence-level ROUGE score

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.

    Returns:
        float: average sentence-level ROUGE score
    """
    rouge_scorer = load("rouge")
    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    # prepare word to references mapping for sentence-rouge
    word2refs = (
        word2refs if word2refs is not None else get_word2refs(words, refs, dedup_refs)
    )

    # compute rouge score for each sample <word, prediction, reference(s)>
    rouge_scores = []
    for _, (word, pred, ref) in enumerate(zip(words, preds, refs)):
        pred_list = [pred]
        ref_list = [ref] if one_to_one else [word2refs[word]]
        rouge_score = rouge_scorer.compute(
            predictions=pred_list,
            references=ref_list,
            tokenizer=lambda s: s.split(),
        )[metric_key]

        rouge_scores.append(rouge_score)

    return mean(rouge_scores)


@timeit
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

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    word2refs = (
        word2refs if word2refs is not None else get_word2refs(words, refs, dedup_refs)
    )

    movers = []
    ref_list, pred_list, span_list = [], [], []
    for _, (word, pred, ref) in enumerate(zip(words, preds, refs)):
        word_ref_list = [ref] if one_to_one else word2refs[word]
        ref_list += word_ref_list
        word_pred_list = [pred] * len(word_ref_list)
        pred_list += word_pred_list
        last_span_end = span_list[-1][1] if len(span_list) > 0 else 0
        span_list.append((last_span_end, last_span_end + len(word_ref_list)))

    scores = word_mover_score(
        refs=ref_list,
        hyps=pred_list,
        idf_dict_ref=idf_dict_refs,
        idf_dict_hyp=idf_dict_preds,
        stop_words=[],
        n_gram=ngram,
        remove_subwords=True,
        batch_size=128,
        device="cuda:3",
    )

    if policy == "max":
        # compute max value for each sub span in span_list indexing scores
        movers += [max(scores[span[0] : span[1]]) for span in span_list]
    elif policy == "mean":
        movers += [mean(scores)]
    else:
        movers += scores

    sentence_mover_score = mean(movers)

    return sentence_mover_score


@timeit
def compute_bert_score(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
    metric_key: Optional[str] = None,
    policy: str = "max",
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

    Returns:
        float: average sentence-level BERTScore
    """
    bert_scorer = load("bertscore")

    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    word2refs = (
        word2refs if word2refs is not None else get_word2refs(words, refs, dedup_refs)
    )

    bert_scores = {"precision": [], "recall": [], "f1": []}
    for _, (word, pred, ref) in enumerate(zip(words, preds, refs)):
        ref_list = [ref] if one_to_one else word2refs[word]
        pred_list = [pred] * len(ref_list)
        # predictions (List[str]) --> references (List[str])
        for key in ["precision", "recall", "f1"]:
            # Dict[str, List[float]]
            bert_score = bert_scorer.compute(
                predictions=pred_list,
                references=ref_list,
                lang="en",
                nthreads=32,
                # device="cuda:2",
                # batch_size=64,
            )
            if policy == "max":
                bert_scores[key] += [max(bert_score[key])]
            elif policy == "mean":
                bert_scores[key] += [mean(bert_score[key])]
            else:
                bert_scores[key] += bert_score[key]

    bert_scores = {k: mean(v) for k, v in bert_scores.items()}

    return (
        bert_scores[metric_key]
        if metric_key is not None and metric_key != "all"
        else bert_scores
    )


@timeit
def compute_meteor(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
) -> float:
    """computes sentence-level METEOR score

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.

    Returns:
        float: average sentence-level METEOR score
    """
    meteor_scorer = load("meteor")

    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    word2refs = (
        word2refs if word2refs is not None else get_word2refs(words, refs, dedup_refs)
    )

    meteors = []
    for _, (word, pred, ref) in enumerate(zip(words, preds, refs)):
        ref_list = [ref] if one_to_one else [word2refs[word]]  # [str] or [[str]]
        pred_list = [pred]  # [str]
        meteor = meteor_scorer.compute(
            predictions=pred_list,
            references=ref_list,
        )["meteor"]
        meteors.append(meteor)

    return mean(meteors)


@timeit
def compute_exact_match(
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
) -> float:
    """computes sentence-level Exact Match score

    Args:
        preds (List[str]): list of predicted sentences
        refs (List[str]): list of reference sentences
        words (Optional[List[str]], optional): list of words. Defaults to None.
        word2refs (Optional[Dict[str, List[str]]], optional): dictionary of words to references. Defaults to None.
        one_to_one (bool, optional): one-to-one mapping of word to reference. Defaults to False.
        dedup_refs (bool, optional): deduplicate references. Defaults to False.

    Returns:
        float: average sentence-level Exact Match score
    """
    exact_match_scorer = load("exact_match")

    preds, refs = get_rid_of_period(preds), get_rid_of_period(refs)

    assert len(words) == len(preds) == len(refs), (
        f"Length mismatch between `words`, `preds`, and `refs` in evaluation: "
        f"{len(words)}, {len(preds)}, {len(refs)}"
    )

    word2refs = (
        word2refs if word2refs is not None else get_word2refs(words, refs, dedup_refs)
    )

    exact_matches = []
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

    return mean(exact_matches)


@timeit
def compute_sari(
    sources: List[str],
    preds: List[str],
    refs: List[str],
    words: Optional[List[str]] = None,
    word2refs: Optional[Dict[str, List[str]]] = None,
    one_to_one: bool = False,
    dedup_refs: bool = False,
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

    Returns:
        float: average sentence-level SARI score
    """
    sari_scorer = load("sari")

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
        word2refs if word2refs is not None else get_word2refs(words, refs, dedup_refs)
    )

    saris = []
    for _, (word, source, pred, ref) in enumerate(zip(words, sources, preds, refs)):
        ref_list = [[ref]] if one_to_one else [word2refs[word]]
        pred_list = [pred]
        source_list = [source]
        sari = sari_scorer.compute(
            sources=source_list,
            predictions=pred_list,
            references=ref_list,
        )["sari"]
        saris.append(sari)

    return mean(saris)


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
