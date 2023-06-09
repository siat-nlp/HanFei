"""
@Desc:
@Reference:
@Notes:
"""

import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple, Union
import numpy as np

from rouge_score import rouge_scorer, scoring
import nltk
import re
import string

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def line_normalize(line: str):
    line = " ".join(line.strip().split())
    return line


def calculate_bleu(ref_lines, gen_lines, metrics: dict = None):
    if metrics is None:
        metrics = {}
    for bleu_i in range(1, 5):
        weights = tuple([1. / bleu_i for _ in range(bleu_i)])
        metrics[f"bleu-{bleu_i}"] = round(nltk.translate.bleu_score.corpus_bleu(
            list_of_references=[[ref] for ref in ref_lines],
            hypotheses=gen_lines,
            weights=weights), 4)
    return metrics


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict


def calculate_rouge(
        pred_lines: List[str],
        tgt_lines: List[str],
        use_stemmer=True,
        rouge_keys=ROUGE_KEYS,
        return_precision_and_recall=False,
        bootstrap_aggregation=True,
        newline_sep=True,
) -> Dict:
    """Calculate rouge using rouge_scorer package.

    Args:
        pred_lines: list of summaries generated by model
        tgt_lines: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).

    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys

    """
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lines, pred_lines):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = pred + "\n"
            tgt = tgt + "\n"
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)


def repetition_distinct_metric(gen_lines, metrics: dict = None, repetition_times=2):
    if metrics is None:
        metrics = {}

    for gram_n in range(1, 5):
        repetition_count = 0
        all_ngram = defaultdict(int)
        all_ngram_num = 0
        for gen_idx, line in enumerate(gen_lines):
            n_grams = ["_".join(gram) for gram in nltk.ngrams(line, n=gram_n)]
            all_ngram_num += len(n_grams)
            # for distinct
            for gram in n_grams:
                all_ngram[gram] += 1
            # for repetition
            for gram in set(n_grams):
                if n_grams.count(gram) >= repetition_times:
                    repetition_count += 1
                    break
        metrics[f"repetition-{gram_n}"] = "%.4f" % (repetition_count / float(len(gen_lines)))
        metrics[f"distinct-{gram_n}"] = "%.4f" % (len(all_ngram) / float(all_ngram_num))
    return metrics


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, predictions: Union[str, List[str]], ground_truths: List[str]):
    scores_for_ground_truths = []

    if isinstance(predictions, str):
        predictions = [predictions]

    for prediction in predictions:
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)

    return max(scores_for_ground_truths)
