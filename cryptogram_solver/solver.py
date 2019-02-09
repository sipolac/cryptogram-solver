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
import logging
import re

from tqdm import tqdm

from cryptogram_solver import data
from cryptogram_solver import utils


Token = namedtuple('Token', 'ngrams kind n')


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
        return text.translate(trans)


class Tokenizer:
    def __init__(self, char_ngram_range=(2, 3), word_ngram_range=(1, 1)):
        self.char_ngram_range = char_ngram_range
        self.word_ngram_range = word_ngram_range

    def get_words(self, text, token_pattern=r'(?u)\b\w+\b'):
        words = re.findall(token_pattern, text)
        return words

    def count_ngrams(self, lst, n):
        return Counter(zip(*[lst[i:] for i in range(n)]))

    def get_tokens(self, lst, ngram_range, kind):
        tokens = dict()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngrams = self.count_ngrams(lst, n)
            for ngram, count in ngrams.items():
                token = Token(ngram, kind, n)
                tokens[token] = tokens.get(token, 0) + count
        return tokens

    def tokenize(self, text):
        """Tokenize text into char- and word-level n-grams."""
        text = clean_text(text)
        words = self.get_words(text)
        word_tokens = self.get_tokens(words, self.word_ngram_range, 'word')
        if self.char_ngram_range is None:
            return word_tokens

        char_tokens = defaultdict(int)
        for word in words:
            word = '<' + word + '>'
            tk = self.get_tokens(word, self.char_ngram_range, 'char')
            for token, count in tk.items():
                char_tokens[token] += count

        tokens = {**word_tokens, **char_tokens}
        return tokens


class Solver:
    def __init__(
        self,
        tokenizer,
        vocab_size,
        pseudo_count,
        logger=None
    ):
        self.logger = logger or logging.getLogger(__name__)

        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.pseudo_count = pseudo_count

        self.vocab = None
        self.totals = None

    def fit(self, texts):
        """Compute token probabilities from data.

        Also subset for most frequent tokens. Keep track of frequencies
        of individual tokens and of token types.
        """
        vocab = defaultdict(int)
        for text in tqdm(texts):
            for token, count in self.tokenizer.tokenize(text).items():
                vocab[token] += count

        # Count totals by token type (kind & n-gram).
        totals = defaultdict(int)
        for token, count in vocab.items():
            totals[(token.kind, token.n)] += count
        totals = dict(totals)

        # Subset vocab for most frequent.
        sorted_tups = sorted(vocab.items(), key=lambda x: -x[1])
        subsetted = sorted_tups[:self.vocab_size]
        vocab = dict(subsetted)

        self.vocab = vocab
        self.totals = totals

    def score(self, text):
        """Caluclate (mean) negative log likelihood."""
        tokens = self.tokenizer.tokenize(text)
        nll = 0  # negative log likelihood
        for token, count in tokens.items():
            vocab_cnt = self.vocab.get(token, 0) + self.pseudo_count
            total = self.totals[(token.kind, token.n)]
            log_prob = log(vocab_cnt) - log(total)
            nll += -1 * log_prob * count
        return nll / len(tokens)  # take mean

    def encrypt(self, text):
        mapping = Mapping()
        mapping.scramble()
        encrypted = mapping.translate(clean_text(text))
        return encrypted

    def decrypt(self, encrypted, num_epochs):
        """Solve cryptogram using simulated annealing.

        This uses a pre-set scheduler for both temperature (from simulated
        annealing) and the number of letters randomly swapped in an iteration
        of simulated annealing.  In the beginning there's a higher temperature
        and larger number of letter swaps to encourage exploration.
        """
        encrypted = clean_text(encrypted)
        mapping = Mapping()

        # Schedule temperature and number of letter swaps to be made.
        temps = [exp(x) for x in utils.linspace(0, -6, num_epochs)]
        n_swap_list = [round(x) for x in utils.linspace(3, 1, num_epochs)]

        best_mapping = mapping
        best_score = self.score(encrypted)

        for epoch, (temp, n_swaps) in tqdm(enumerate(zip(temps, n_swap_list)), total=num_epochs):

            new_mapping = mapping.random_swap(n_swaps)
            new_text = new_mapping.translate(encrypted)
            score = self.score(new_text)

            score_change = score - best_score

            if exp(-score_change / temp) > uniform(0, 1):
                best_mapping = new_mapping
                best_score = score

            mapping = best_mapping

            if self.logger.level < 20 and epoch % 1000 == 0:
                self.logger.debug((
                    f'\nscore: {score:0.5g}'
                    f'\nkey: {mapping.key}'
                    f'\ndecrypted: {mapping.translate(encrypted)}'
                    '\n'
                ))

        decrypted = mapping.translate(encrypted)
        return decrypted


def clean_text(text):
    return text.lower()


def main(
    text,
    num_epochs,
    char_ngram_range,
    word_ngram_range,
    vocab_size,
    n_docs,
    pseudo_count,
    log_level='INFO'
):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_fmt)
    logger = logging.getLogger(__name__)

    tokenizer = Tokenizer(
        char_ngram_range=char_ngram_range,
        word_ngram_range=word_ngram_range
    )
    slv = Solver(tokenizer, vocab_size, pseudo_count)

    logger.info('reading data for training solver...')
    docs = data.get_news_articles()
    logger.info('computing character and word frequencies...')
    slv.fit(docs[:n_docs])

    encrypted = slv.encrypt(text)
    logger.info('decrypting...')
    decrypted = slv.decrypt(encrypted, num_epochs)
    logger.info(f'decrypted text: {decrypted}')
    return decrypted


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', help='text to be decrypted')
    parser.add_argument(
        '-e', '--num_epochs', default=10000,
        help='number of epochs during simulated annealing process'
    )
    parser.add_argument(
        '-c', '--char_ngram_range', nargs=2, default=(2, 3), type=int,
        help='range of character n-grams to use in tokenization'
    )
    parser.add_argument(
        '-w', '--word_ngram_range', nargs=2, default=(1, 1), type=int,
        help='range of word n-grams to use in tokenization'
    )
    parser.add_argument(
        '-b', '--vocab_size', default=10000,
        help='size of vocabulary to use for scoring (other words are OOV)'
    )
    parser.add_argument(
        '-d', '--n_docs', default=100,
        help='number of documents used to estimate token frequencies'
    )
    parser.add_argument(
        '-p', '--pseudo_count', default=1,
        help='number added to all token frequencies for smoothing'
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='verbose output for showing solve process'
    )
    args = parser.parse_args()

    args.verbose = min(args.verbose, 1)
    log_level = {0: 'INFO', 1: 'DEBUG'}.get(args.verbose, 0)

    main(
        text=args.text,
        num_epochs=args.num_epochs,
        char_ngram_range=args.char_ngram_range,
        word_ngram_range=args.word_ngram_range,
        vocab_size=args.vocab_size,
        n_docs=args.n_docs,
        pseudo_count=args.pseudo_count,
        log_level=log_level
    )