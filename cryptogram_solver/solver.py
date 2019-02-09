#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-02-07

Functions for solving cryptograms.
"""
from collections import Counter, defaultdict, namedtuple
from math import exp, log
from random import sample, uniform
from string import ascii_lowercase as LETTERS
# from time import time
import logging
import re

from tqdm import tqdm
import numpy as np   # for softmax; TODO: remove this dependency later


Token = namedtuple('Token', 'ngrams kind n')


class Doc(str):
    def __init__(self, text):
        self.text = text
        self._letters = None

    @property
    def letters(self):
        pass

    @letters.getter
    def letters(self):
        if self._letters is None:
            chars = [l.lower() for l in set(self.text)]
            self._letters = list(set(chars) & set(LETTERS))
        return self._letters


class Mapping:
    def __init__(self, key=None):
        self.key = key or LETTERS

    def scramble(self):
        self.key = ''.join(sample(self.key, len(self.key)))

    def random_swap(self, n=1):
        key = list(self.key)
        for _ in range(n):
            i1, i2 = sample(range(len(LETTERS)), 2)
            key[i1], key[i2] = key[i2], key[i1]
        return Mapping(''.join(key))

    def translate(self, text):
        trans = str.maketrans(self.key, LETTERS)
        return type(text)(text.translate(trans))


class Tokenizer:
    def __init__(
        self,
        char_ngram_range=(1, 3),
        word_ngram_range=(1, 1),
        vocab_size=100000
    ):
        self.char_ngram_range = char_ngram_range
        self.word_ngram_range = word_ngram_range
        self.vocab_size = vocab_size
        self.vocab = None
        self.totals = None

    def clean_text(self, text):
        text = text.lower()
        text = re.sub('[^a-z ]', '?', text)
        return text

    def get_words(self, text, token_pattern=r'(?u)\b\w+\b'):
        words = re.findall(token_pattern, text)
        return words

    def count_ngrams(self, lst, n):
        return Counter(zip(*[lst[i:] for i in range(n)]))

    def add_ngram_tokens(self, lst, ngram_range, kind, tokens):
        """Add n-gram tokens to given dictionary of tokens.

        Mutates tokens, but also returns it for readability.
        """
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngrams = self.count_ngrams(lst, n)
            for ngram, count in ngrams.items():
                token = Token(ngram, kind, n)
                tokens[token] = tokens.get(token, 0) + count
        return tokens

    def tokenize(self, text, tokens=None):
        """Tokenize text into char- and word-level n-grams.

        Mutates tokens, but also returns it for readability.
        """
        if tokens is None:
            tokens = dict()
        text = self.clean_text(text)
        words = self.get_words(text)
        if self.word_ngram_range is not None:
            tokens = self.add_ngram_tokens(
                words,
                self.word_ngram_range,
                'word',
                tokens
            )
        if self.char_ngram_range is not None:
            for word in words:
                word = '<' + word + '>'
                tokens = self.add_ngram_tokens(
                    word,
                    self.char_ngram_range,
                    'char',
                    tokens
                )
        return tokens

    def fit(self, texts):
        """Fit tokenizer to data.

        Also subset for most frequent tokens. Keep track of frequencies
        of individual tokens and of token types.
        """
        self.vocab = dict()
        for text in tqdm(texts):
            self.vocab = self.tokenize(text, self.vocab)

        # Count totals by token type (kind & n-gram).
        self.totals = defaultdict(int)
        for token, count in self.vocab.items():
            self.totals[(token.kind, token.n)] += count
        self.totals = dict(self.totals)

        # Subset vocab for most frequent.
        sorted_tups = sorted(self.vocab.items(), key=lambda x: -x[1])
        subsetted = sorted_tups[:self.vocab_size]
        self.vocab = dict(subsetted)


class Solver:
    def __init__(self, tokenizer, pseudo_count=1, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        self.tokenizer = tokenizer
        self.pseudo_count = pseudo_count

    def score(self, text):
        """Caluclate (mean) negative log likelihood."""
        tokens = self.tokenizer.tokenize(text)
        nll = 0  # negative log likelihood
        for token, count in tokens.items():
            vocab_cnt = self.tokenizer.vocab.get(token, 0) + self.pseudo_count
            total = self.tokenizer.totals[(token.kind, token.n)]
            log_prob = log(vocab_cnt) - log(total)
            nll += -1 * log_prob * count
        return nll / len(tokens)  # take mean

    def solve(self, text, num_epochs=10000):
        """Solve using simulated annealing."""

        doc = Doc(text)
        mapping = Mapping()

        # Schedule temperature and number of letter swaps to be made.
        temps = np.exp(np.linspace(0, -6, num_epochs))
        n_swap_list = np.round(np.linspace(3, 1, num_epochs)).astype(int)

        best_mapping = mapping
        best_score = self.score(doc)
        epoch = 0

        decisions = defaultdict(int)

        for temp, n_swaps in tqdm(zip(temps, n_swap_list)):

            new_mapping = mapping.random_swap(n_swaps)
            new_doc = new_mapping.translate(doc)
            score = self.score(new_doc)

            score_change = score - best_score

            if score_change < 0:
                best_mapping = new_mapping
                best_score = score
                decisions['good'] += 1
            elif exp(-score_change / temp) > uniform(0, 1):
                # Break this out as different section just for debugging.
                best_mapping = new_mapping
                best_score = score
                decisions['bad_keep'] += 1
            else:
                # Again, just for debugging.
                decisions['bad_pass'] += 1

            mapping = best_mapping
            epoch += 1

            if epoch % 1000 == 0:
                self.logger.debug(f'{score:0.5g}, {mapping.key}, {mapping.translate(doc).text}')
                self.logger.debug(sorted(list(decisions.items())))
                # logger.debug(pd.DataFrame(sorted(list(decisions.items()))))
                decisions = defaultdict(int)

        self.logger.info(f'\nfinal best ({epoch} epochs): {best_score:0.5g}')
        return mapping.translate(doc).text
