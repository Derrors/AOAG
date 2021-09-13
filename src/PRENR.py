# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
# I have just rewritten stuff so that it works without tensorflow
# I also added multi-precessing to greatly speedup the computations

"""Script to compute PARENT metric."""
import collections
import itertools
import math
import multiprocessing as mp
from functools import partial

import numpy as np
from tqdm import tqdm


def overlap_probability(ngram, reviews, smoothing=0.0, stopwords=None):
    """Returns the probability that the given n-gram overlaps with the reviews.

    A simple implementation which checks how many tokens in the n-gram are also
    in the reviews.

    Args:
    ngram: List of tokens.
    reviews: List of reviews, each review is a list of strings.
    smoothing: (Optional) Float parameter for laplace smoothing.
    stopwords: (Optional) List of stopwords to ignore (assign P = 1).

    Returns:
    prob: Float probability of ngram being entailed by the reviews.
    """

    review_tokens = set([tok for review in reviews for tok in review])

    overlap = 0
    for token in ngram:
        if stopwords is not None and token in stopwords:
            overlap += 1
            continue
        if token in review_tokens:
            overlap += 1
    return float(overlap + smoothing) / float(len(ngram) + smoothing)


def _mention_probability(review, sentence, smoothing=0.0):
    """Returns the probability that the table entry is mentioned in the sentence.

    A simple implementation which checks the longest common subsequence between
    the review and the sentence.

    Args:
    review: A list of strings.
    sentence: List of tokens.
    smoothing: Float parameter for laplace smoothing.

    Returns:
    prob: Float probability of review being in sentence.
    """

    overlap = _len_lcs(review, sentence)
    return float(overlap + smoothing) / float(len(review) + smoothing)


def _len_lcs(x, y):
    """Returns the length of the Longest Common Subsequence between two seqs.

    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
    x: sequence of words
    y: sequence of words

    Returns
    integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """Computes the length of the LCS between two seqs.

    The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
    x: collection of words
    y: collection of words

    Returns:
    Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def nwise(iterable, n=2):
    """Yields all ngrams of given order n in iterable."""
    iterables = itertools.tee(iterable, n)
    [next(iterables[i]) for i in range(n) for j in range(i)]
    return zip(*iterables)


def _ngram_counts(sequence, order):
    """Returns count of all ngrams of given order in sequence."""
    if len(sequence) < order:
        return collections.Counter()
    return collections.Counter(nwise(sequence, order))


def parent_instance_level(package,
                          smoothing=0.00001,
                          max_order=4,
                          entailment_fn=overlap_probability,
                          mention_fn=_mention_probability):

    prediction, reviews = package  # unpacking

    # Compute recall against review fields (it doesn't depend on the ref).
    review_mention_probs = [mention_fn(review, prediction) for review in reviews]
    review_rec = sum(review_mention_probs) / len(reviews) or smoothing

    # Weighted ngram precisions and recalls for each order.
    ngram_prec = list()
    for order in range(1, max_order + 1):
        # Collect n-grams and their entailment probabilities.
        pred_ngram_counts = _ngram_counts(prediction, order)
        pred_ngram_weights = {ngram: entailment_fn(ngram, reviews) for ngram in pred_ngram_counts}      # w(g)

        # Precision.
        numerator, denominator = 0., 0.
        for ngram, count in pred_ngram_counts.items():
            denominator += count
            # prob_ngram_in_ref = min(1., float(ref_ngram_counts.get(ngram, 0) / count))
            numerator += count * pred_ngram_weights[ngram]
        if denominator == 0.:
            # Set precision to 0.
            ngram_prec.append(0.0)
        else:
            ngram_prec.append(numerator / denominator)

    # Smoothing.
    for order in range(1, max_order):
        if ngram_prec[order] == 0.:
            ngram_prec[order] = smoothing

    # Compute geometric averages of precision and recall for all orders.
    w = 1. / max_order
    if any(prec == 0. for prec in ngram_prec):
        c_prec = 0
    else:
        sp = (w * math.log(p_i) for p_i in ngram_prec)
        c_prec = math.exp(math.fsum(sp))

    # Combine reference and review recalls.
    if review_rec == 0.:
        c_rec = 0
    else:
        c_rec = review_rec

    # F-score.
    c_f1 = (2. * c_prec * c_rec) / (c_prec + c_rec + 1e-8)

    return c_prec, c_rec, c_f1


def _parent(predictions,
            reviews_list,
            smoothing=0.00001,
            max_order=4,
            entailment_fn=overlap_probability,
            mention_fn=_mention_probability,
            n_jobs=-1):
    """
    Metric for comparing predictions to references given reviews_list.
    It now uses multiprocessing to go faster (minutes to seconds).

    ARGS:
    predictions: An iterator over tokenized predictions.
                 Each prediction is a list.
    references: An iterator over lists of tokenized references.
                Each prediction can have multiple references.
    reviews_list: A list of review sets, each review set have several reviews sentecnes.
    lambda_weight: Float weight in [0, 1] to multiply table recall.
    smoothing: Float value to replace zero values of precision and recall.
    max_order: Maximum order of the ngrams to use.
    entailment_fn: A python function for computing the probability that an
                   ngram is entailed by the reviews. Its signature should match
                   that of `overlap_probability` above.
    mention_fn: A python function for computing the probability that a
                review is mentioned in the text. Its signature should
                match that of `_mention_probability` above.
    n_jobs: An int to specify number of parallel workers.
            -1 to use all available.

    RETURNS:
    precision: Precision of all predictions.
    recall: Recall of all predictions.
    f1: F-scores of all predictions.
    """

    precisions, recalls, f1_scores = list(), list(), list()

    _parent = partial(parent_instance_level,
                      smoothing=smoothing,
                      max_order=max_order,
                      entailment_fn=entailment_fn,
                      mention_fn=mention_fn)

    n_jobs = mp.cpu_count() if n_jobs < 0 else n_jobs

    print(f'Using {n_jobs} processes, starting now.')

    with mp.Pool(processes=n_jobs) as pool:
        _iterable = pool.imap(
            _parent,
            zip(predictions, reviews_list),
            chunksize=n_jobs  # empirically seems to be the best, could be wrong though
        )

        for p, r, f in tqdm(_iterable, total=len(reviews_list), desc='Computing PARENT', leave=False):
            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f)

    return precisions, recalls, f1_scores


def parent(predictions,
           reviews_list,
           smoothing=0.00001,
           max_order=4,
           entailment_fn=overlap_probability,
           mention_fn=_mention_probability,
           avg_results=True,
           n_jobs=-1):
    """
    Metric for comparing predictions to references given reviews_list.
    It now uses multiprocessing to go faster (minutes to seconds).

    ARGS:
    predictions: An iterator over tokenized predictions.
                 Each prediction is a list.
    references: An iterator over lists of tokenized references.
                Each prediction can have multiple references.
    reviews_list: A list of review sets, each review set have several reviews sentecnes.
    lambda_weight: Float weight in [0, 1] to multiply table recall.
    smoothing: Float value to replace zero values of precision and recall.
    max_order: Maximum order of the ngrams to use.
    entailment_fn: A python function for computing the probability that an
                   ngram is entailed by the reviews. Its signature should match
                   that of `overlap_probability` above.
    mention_fn: A python function for computing the probability that a
                review is mentioned in the text. Its signature should
                match that of `_mention_probability` above.
    n_jobs: An int to specify number of parallel workers.
            -1 to use all available.

    RETURNS:
    precision, recall, f1_score: either three floats or three lists of floats.
    """

    precisions, recalls, f1_scores = _parent(
        predictions,
        reviews_list,
        smoothing=smoothing,
        max_order=max_order,
        entailment_fn=entailment_fn,
        mention_fn=mention_fn,
        n_jobs=n_jobs)

    if avg_results:
        precisions = sum(precisions) / len(precisions)
        recalls = sum(recalls) / len(recalls)
        f1_scores = sum(f1_scores) / len(f1_scores)

    return precisions, recalls, f1_scores


def call_fn(predictions, reviews_list, smoothing=0.00001, max_order=4, avg_results=True, n_jobs=10):
    pre_tokens = [s.split() for s in predictions]
    precisions, recalls, f1_scores = parent(
        pre_tokens,
        reviews_list,
        smoothing=smoothing,
        max_order=max_order,
        avg_results=avg_results,
        n_jobs=n_jobs)

    return precisions, recalls, f1_scores
