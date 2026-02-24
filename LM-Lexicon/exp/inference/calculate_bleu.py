#!/usr/bin/env python3

import sys

from statistics import mean
from nltk.translate import bleu_score


def calc(pred, ref):
    pred, ref = pred.split(), [ref.split()]
    bleu = bleu_score.sentence_bleu(
        ref,
        pred,
        smoothing_function=bleu_score.SmoothingFunction().method2,
        auto_reweigh=True,
    )
    print(bleu)
    return bleu


if __name__ == "__main__":
    scores = []

    input_path = sys.argv[1]

    with open(
        input_path, "r"
    ) as f:
        lines = [l.strip() for l in f.readlines()]
        for line in lines:
            _, pred, ref = line.split("\t")
            score = calc(pred, ref)
            scores.append(score)

    mean_bleu = mean(scores)

    print("Mean BLEU score:", round(mean_bleu * 100, 2))
