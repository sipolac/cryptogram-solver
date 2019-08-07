#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-02-23

Turn a corpus into a list of unigram frequencies.
"""
from collections import defaultdict

from tqdm import tqdm

from cryptogram_solver import defaults
from cryptogram_solver import solver
from cryptogram_solver import utils


def make_unigram_freqs(docs_path, outfile, n=None):
    """Creates list of word frequencies sorted descending given a corpus.

    This just makes loading easier instead of having to fit to a corpus
    each time. Only works if largest word n-gram degree is a unigram.
    """
    tk = solver.Tokenizer(char_ngram_range=None,
                          word_ngram_range=(1, 1))
    docs = utils.read_docs(docs_path)
    unigram_counts = defaultdict(int)
    for doc in tqdm(docs):
        doc_tokens = tk.tokenize(doc)
        for token, count in doc_tokens.items():
            ngram_str = ' '.join(token.ngrams)
            unigram_counts[ngram_str] += count

    sorted_tups = sorted(unigram_counts.items(), key=lambda x: -x[1])
    subsetted = sorted_tups[:n]

    outfile.parent.mkdir(exist_ok=True, parents=True)
    if outfile.exists():
        outfile.unlink()
    with open(outfile, 'w') as f:
        for unigram, count in subsetted:
            f.write(f'{unigram},{str(count)}\n')


def main():
    make_unigram_freqs(defaults.CORPUS_PATH, defaults.FREQS_PATH)


if __name__ == '__main__':
    main()
