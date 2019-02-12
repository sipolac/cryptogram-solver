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
import argparse
import json
import logging
import pickle
import re

from tqdm import tqdm

from cryptogram_solver import data
from cryptogram_solver import utils


PROJECT_DIR = utils.get_project_dir()
tokenizer_path = PROJECT_DIR / 'models' / 'tokenizer.pkl'
vocab_path = PROJECT_DIR / 'models' / 'vocab.json'
totals_path = PROJECT_DIR / 'models' / 'totals.json'
pseudo_count_path = PROJECT_DIR / 'models' / 'pseudo_count.txt'


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

    def print_pretty(self):
        # Order by original letters.
        lst1 = [f'{l} --> {k.upper()}' for l, k in zip(LETTERS, self.key)]

        # Order by key.
        idx = sorted(range(len(self.key)), key=self.key.__getitem__)
        lst2 = [f'{LETTERS[i]} --> {self.key[i].upper()}' for i in idx]

        # Combine to show both.
        lst = [f'{l1}    {l2}' for l1, l2 in zip(lst1, lst2)]

        print(*lst, sep='\n')


class Tokenizer:
    def __init__(self, char_ngram_range, word_ngram_range):
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
        vocab=None,
        totals=None,
        logger=None
    ):
        self.logger = logger or logging.getLogger(__name__)

        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.pseudo_count = pseudo_count

        self.vocab = vocab
        self.totals = totals

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

        decrypt_tqdm = tqdm(
            enumerate(zip(temps, n_swap_list)),
            total=num_epochs,
            desc='decrypting'
        )
        for epoch, (temp, n_swaps) in decrypt_tqdm:

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
        return {'mapping': mapping, 'decrypted': decrypted}


def save_solver_fn(slv, tokenizer_path, vocab_path, totals_path):
    pickle.dump(slv.tokenizer, open(tokenizer_path, 'wb'))
    with open(vocab_path, 'w') as f:
        json.dump(jsonify_vocab(slv.vocab), f)
    with open(totals_path, 'w') as f:
        json.dump(jsonify_totals(slv.totals), f)
    pseudo_count_path.write_text(str(slv.pseudo_count))


def load_solver_fn():
    # Solver can be reconstructed from the the vocab, totals and tokenizer.
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    pseudo_count = int(pseudo_count_path.read_text())
    with open(vocab_path) as f:
        vocab = unjsonify_vocab(json.load(f))
    with open(totals_path) as f:
        totals = unjsonify_totals(json.load(f))
    slv = Solver(tokenizer, len(vocab), pseudo_count, vocab, totals)
    return slv


def clean_text(text):
    return text.lower()


def jsonify_vocab(vocab):
    return [[list(t), c] for t, c in vocab.items()]


def unjsonify_vocab(vocab):
    return {Token(tuple(t[0]), t[1], t[2]): c for t, c in vocab}


def jsonify_totals(totals):
    return list(totals.items())


def unjsonify_totals(totals):
    return {tuple(t): c for t, c in totals}


def run_solver(
    text,
    num_epochs,
    char_ngram_range=None,
    word_ngram_range=None,
    vocab_size=None,
    n_docs=None,
    pseudo_count=None,
    load_solver=False,
    save_solver=False,
    logger=None
):
    if load_solver:
        slv = load_solver_fn()

    else:
        tokenizer = Tokenizer(
            char_ngram_range=char_ngram_range,
            word_ngram_range=word_ngram_range
        )
        slv = Solver(tokenizer, vocab_size, pseudo_count)

        print('reading data for training solver...')
        docs = data.get_news_articles(n_docs)
        print('computing character and word frequencies...')
        slv.fit(docs)

        if save_solver:
            save_solver_fn(slv, tokenizer_path, vocab_path, totals_path)

    res = slv.decrypt(text, num_epochs)
    mapping, decrypted = res['mapping'], res['decrypted']
    print('\nmapping:')
    mapping.print_pretty()
    print(f'\ndecrypted text:\n{decrypted}')
    return decrypted


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('text', help='text to be decrypted')
    parser.add_argument(
        '-e', '--num_epochs', default=10000, type=int,
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
        '-b', '--vocab_size', default=10000, type=int,
        help='size of vocabulary to use for scoring (other words are OOV)'
    )
    parser.add_argument(
        '-n', '--n_docs', default=1000, type=int,
        help='number of documents used to estimate token frequencies'
    )
    parser.add_argument(
        '-p', '--pseudo_count', default=1, type=float,
        help='number added to all token frequencies for smoothing'
    )
    parser.add_argument(
        '-l', '--load_solver', action='store_true', default=False,
        help='load solver to save time'
    )
    parser.add_argument(
        '-s', '--save_solver', action='store_true', default=False,
        help='save solver for use later'
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='verbose output for showing solve process'
    )
    args = parser.parse_args()

    args.verbose = min(args.verbose, 1)
    log_level = {0: 'INFO', 1: 'DEBUG'}.get(args.verbose, 0)

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_fmt)
    logger = logging.getLogger(__name__)

    run_solver(
        text=args.text,
        num_epochs=args.num_epochs,
        char_ngram_range=args.char_ngram_range,
        word_ngram_range=args.word_ngram_range,
        vocab_size=args.vocab_size,
        n_docs=args.n_docs,
        pseudo_count=args.pseudo_count,
        load_solver=args.load_solver,
        save_solver=args.save_solver,
        logger=logger
    )


if __name__ == '__main__':
    main()
