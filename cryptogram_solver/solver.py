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
import re

from tqdm import tqdm

from cryptogram_solver import data
from cryptogram_solver import utils


PROJECT_DIR = utils.get_project_dir()


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

    def _get_words(self, text, token_pattern=r'(?u)\b\w+\b'):
        words = re.findall(token_pattern, text)
        return words

    def _count_ngrams(self, lst, n):
        return Counter(zip(*[lst[i:] for i in range(n)]))

    def _get_tokens(self, lst, ngram_range, kind):
        tokens = dict()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngrams = self._count_ngrams(lst, n)
            for ngram, count in ngrams.items():
                token = Token(ngram, kind, n)
                tokens[token] = tokens.get(token, 0) + count
        return tokens

    def tokenize(self, text):
        """Tokenize text into char- and word-level n-grams."""
        text = re.sub(r'[^a-zA-Z ]', '?', text.lower())  # remove punct
        words = self._get_words(text)
        word_tokens = self._get_tokens(words, self.word_ngram_range, 'word')
        if self.char_ngram_range is None:
            return word_tokens

        char_tokens = defaultdict(int)
        for word in words:
            word = '<' + word + '>'
            tk = self._get_tokens(word, self.char_ngram_range, 'char')
            for token, count in tk.items():
                char_tokens[token] += count

        tokens = {**word_tokens, **char_tokens}
        return tokens


class Solver:
    def __init__(self, cfg=dict(), logger=None):
        self.logger = logger or logging.getLogger(__name__)

        self.cfg = dict(
            char_ngram_range=None,
            word_ngram_range=None,
            vocab_size=None,
            pseudo_count=None
        )
        assert all([k in self.cfg.keys() for k in cfg.keys()])
        self.cfg.update(cfg)

        self.tokenizer = Tokenizer(
            cfg['char_ngram_range'],
            cfg['word_ngram_range']
        )

        # These are defined when fitted.
        self.vocab = None
        self.totals = None

    def fit(self, texts):
        """Compute token probabilities from data.

        Also subset for most frequent tokens. Keep track of frequencies
        of individual tokens and of token types.
        """
        vocab = defaultdict(int)
        for text in tqdm(texts, desc='fitting solver'):
            for token, count in self.tokenizer.tokenize(text).items():
                vocab[token] += count

        # Count totals by token type (kind & n-gram).
        totals = defaultdict(int)
        for token, count in vocab.items():
            totals[(token.kind, token.n)] += count
        totals = dict(totals)

        # Subset vocab for most frequent.
        sorted_tups = sorted(vocab.items(), key=lambda x: -x[1])
        subsetted = sorted_tups[:self.cfg['vocab_size']]
        vocab = dict(subsetted)

        self.vocab = vocab
        self.totals = totals

    def score(self, text):
        """Caluclate (mean) negative log likelihood."""
        tokens = self.tokenizer.tokenize(text)
        nll = 0  # negative log likelihood
        for token, count in tokens.items():
            vocab_cnt = self.vocab.get(token, 0) + self.cfg['pseudo_count']
            total = self.totals[(token.kind, token.n)]
            log_prob = log(vocab_cnt) - log(total)
            nll += -1 * log_prob * count
        return nll / len(tokens)  # take mean

    def decrypt(
        self,
        encrypted,
        num_iters,
        log_temp_start,
        log_temp_end,
        lamb_start,
        lamb_end
    ):
        """Solve cryptogram using simulated annealing.

        This uses a pre-set scheduler for both temperature (from simulated
        annealing) and the number of letters randomly swapped in an iteration
        of simulated annealing.  In the beginning there's a higher temperature
        and larger number of letter swaps to encourage exploration.
        """
        encrypted = encrypted.lower()
        mapping = Mapping()

        temps = self._schedule_temp(log_temp_start, log_temp_end, num_iters)
        swap_list = self._schedule_swaps(lamb_start, lamb_end, num_iters)

        best_mapping = mapping
        best_score = self.score(encrypted)

        for i in tqdm(list(range(num_iters)), desc='decrypting'):
            temp = temps[i]
            swaps = swap_list[i]

            new_mapping = mapping.random_swap(swaps)
            new_text = new_mapping.translate(encrypted)
            score = self.score(new_text)

            score_change = score - best_score

            if exp(-score_change / temp) > uniform(0, 1):
                best_mapping = new_mapping
                best_score = score

            mapping = best_mapping

            if self.logger.level < 20 and i % 1000 == 0:
                self.logger.debug((
                    f'\nscore: {score:0.5g}'
                    f'\nkey: {mapping.key}'
                    f'\ndecrypted: {mapping.translate(encrypted)}'
                    '\n'
                ))

        decrypted = mapping.translate(encrypted)
        return {'mapping': mapping, 'decrypted': decrypted}

    def save(self, path):
        with open(path / 'vocab.json', 'w') as f:
            json.dump(self._jsonify_vocab(self.vocab), f)
        with open(path / 'totals.json', 'w') as f:
            json.dump(self._jsonify_totals(self.totals), f)
        (path / 'cfg.json').write_text(json.dumps(self.cfg))

    @classmethod
    def load(cls, path):
        cfg = json.loads((path / 'cfg.json').read_text())
        with open(path / 'vocab.json') as f:
            vocab = cls._unjsonify_vocab(json.load(f))
        with open(path / 'totals.json') as f:
            totals = cls._unjsonify_totals(json.load(f))
        slv = cls(cfg)
        slv.vocab = vocab
        slv.totals = totals
        return slv

    def _schedule_temp(self, start, end, n):
        return [exp(x) for x in utils.linspace(start, end, n)]

    def _schedule_swaps(self, start, end, n):
        return [utils.poisson(l) + 1 for l in utils.linspace(start, end, n)]

    def _jsonify_vocab(self, vocab):
        return [[list(t), c] for t, c in vocab.items()]

    @classmethod
    def _unjsonify_vocab(cls, vocab):
        return {Token(tuple(t[0]), t[1], t[2]): c for t, c in vocab}

    def _jsonify_totals(self, totals):
        return list(totals.items())

    @classmethod
    def _unjsonify_totals(cls, totals):
        return {tuple(t): c for t, c in totals}


def encrypt(text):
    mapping = Mapping()
    mapping.scramble()
    encrypted = mapping.translate(text).upper()
    return encrypted


def run_solver(
    text,
    cfg=None,
    num_iters=None,
    log_temp_start=None,
    log_temp_end=None,
    lamb_start=None,
    lamb_end=None,
    n_docs=None,
    load_solver=False,
    save_solver=False,
    logger=None
):
    models_path = PROJECT_DIR / 'models' / 'cached'
    logger = logger or logging.getLogger(__name__)

    if load_solver:
        slv = Solver.load(models_path)

    else:
        tokenizer = Tokenizer(
            char_ngram_range=cfg['char_ngram_range'],
            word_ngram_range=cfg['word_ngram_range']
        )
        slv = Solver(cfg)

        print('reading data for training solver...')
        docs = data.get_news_articles(n_docs)
        slv.fit(docs)
        if save_solver:
            slv.save(models_path)

    res = slv.decrypt(
        text,
        num_iters,
        log_temp_start,
        log_temp_end,
        lamb_start,
        lamb_end
    )
    mapping, decrypted = res['mapping'], res['decrypted']
    print('\ncipher:')
    mapping.print_pretty()
    print(f'\ndecrypted text:\n{decrypted}')
    return decrypted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'text',
        help='text to be decrypted'
    )
    parser.add_argument(
        '-i', '--num_iters', default=10000, type=int,
        help='number of iterations during simulated annealing process'
    )
    parser.add_argument(
        '-c', '--char_ngram_range', nargs=2, default=(2, 2), type=int,
        help='range of character n-grams to use in tokenization'
    )
    parser.add_argument(
        '-w', '--word_ngram_range', nargs=2, default=(1, 1), type=int,
        help='range of word n-grams to use in tokenization'
    )
    parser.add_argument(
        '-n', '--n_docs', default=1000, type=int,
        help='number of documents used to estimate token frequencies'
    )
    parser.add_argument(
        '-b', '--vocab_size', default=10000, type=int,
        help='size of vocabulary to use for scoring'
    )
    parser.add_argument(
        '-p', '--pseudo_count', default=1, type=float,
        help='number added to all token frequencies for smoothing'
    )
    parser.add_argument(
        '--log_temp_start', default=-1, type=int,
        help='log of initial temperature'
    )
    parser.add_argument(
        '--log_temp_end', default=-6, type=int,
        help='log of final temperature'
    )
    parser.add_argument(
        '--lamb_start', default=0.5, type=int,
        help='poisson lambda for number of letter swaps in beginning'
    )
    parser.add_argument(
        '--lamb_end', default=0, type=int,
        help='poisson lambda for number of letter swaps at end'
    )
    parser.add_argument(
        '-l', '--load_solver', action='store_true', default=False,
        help='load pre-fitted solver'
    )
    parser.add_argument(
        '-s', '--save_solver', action='store_true', default=False,
        help='save fitted solver for use later'
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

    cfg = dict(
        char_ngram_range=args.char_ngram_range,
        word_ngram_range=args.word_ngram_range,
        vocab_size=args.vocab_size,
        pseudo_count=args.pseudo_count
    )

    run_solver(
        text=args.text,
        cfg=cfg,
        num_iters=args.num_iters,
        log_temp_start=args.log_temp_start,
        log_temp_end=args.log_temp_end,
        lamb_start=args.lamb_start,
        lamb_end=args.lamb_end,
        n_docs=args.n_docs,
        load_solver=args.load_solver,
        save_solver=args.save_solver,
        logger=logger
    )


if __name__ == '__main__':
    main()
