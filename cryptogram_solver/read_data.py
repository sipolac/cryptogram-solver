#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-02-23

Functions for processing data used to fit the solver.
"""
from collections import OrderedDict

from cryptogram_solver import defaults


def read_freqs(path, n=50000):
    """Create word frequencies dictionary.

    Args:
        path: Path or str that points to a CSV with two columns, where
            the first column is the unigram and the second is the frequency.
        n: Top n most frequent terms are read, assuming CSV is ordered
            descending by frequency.

    Returns:
        OrderedDict of {n-gram: frequency} pairs.
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
    """Create generator of documents from corpus."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i == n:
                return
            yield line.strip()


def main():
    # Quick spot check.
    print(read_freqs(defaults.FREQS_PATH, 3))
    print(list(read_docs(defaults.CORPUS_PATH))[1][:100])


if __name__ == '__main__':
    main()
