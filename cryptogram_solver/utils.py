#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-01-25

Basic utility functions.

Some of these are basic versions of numpy functions. They're recreated
here so that we don't have to rely on numpy as a dependency.
"""
from collections import OrderedDict
from math import exp
from pathlib import Path
from random import uniform

from cryptogram_solver import defaults


def get_project_dir():
    file_stem = 'cryptogram-solver'
    parents = Path(__file__).resolve().parents
    for filepath in parents:
        if filepath.stem == file_stem:
            break
    return filepath


def read_freqs(path, n=50000):
    """Creates dictionary of unigram frequencies.

    Can handle CSVs with or without headers.

    Args:
        path: Path or str that points to a CSV with two columns, where
            the first column is the unigram and the second is the frequency.
        n: Top n most frequent terms are read, assuming CSV is ordered
            descending by frequency.

    Returns:
        OrderedDict of {n-gram: frequency}.
    """
    freqs = OrderedDict()
    max_ix = n
    with open(path) as f:
        for i, line in enumerate(f):
            if i == max_ix:
                break
            word, freq = line.strip().split(',')
            try:
                freqs[word] = int(freq)
            except ValueError:
                if i == 0:
                    max_ix += 1  # increment to account for CSV header
                else:
                    raise Exception('character value for frequency')
    return freqs


def read_docs(path, n=None):
    """Generates documents from corpus."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i == n:
                return
            yield line.strip()


def impute_defaults(d, default_d):
    """Imputes values of dictionary with defaults if key doesn't exist.

    Args:
        d: Dict you want imputed.
        default_d: Dict used to impute values of other dict.
    """
    assert all([k in default_d for k in d])
    for k, v in default_d.items():
        if k not in d:
            d[k] = v
    return d


def linspace(start, stop, num):
    """Returns evenly spaced numbers over a specified interval.

    This is a simple version of numpy's `linspace`:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
    """
    step_size = (stop - start) / (num - 1)
    return (start + step_size * i for i in range(num))


def rpoisson(lamb):
    """Draws a random sample from a Poisson distribution given lambda.

    When lambda is 0, output is 0.

    Algorithm from here:
    https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
    """
    if lamb == 0:
        return 0
    L = exp(-lamb)
    k = 0
    p = 1
    while p > L:
        k += 1
        p *= uniform(0, 1)
    return k - 1


def main():
    # Quick spot check.
    print(read_freqs(defaults.FREQS_PATH, 3))
    print(list(read_docs(defaults.CORPUS_PATH))[1][:100])


if __name__ == '__main__':
    main()
