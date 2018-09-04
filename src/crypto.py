from collections import Counter, defaultdict, namedtuple
from enum import Enum
from math import log
from random import sample
from string import ascii_lowercase as LETTERS
from time import time
import re

from tqdm import tqdm


Token = namedtuple('Token', 'ngrams kind n')


class NgramKind(Enum):
    word, char = range(2)

    def __repr__(self):
        return f'<NgramKind: {self.name}>'


class Doc(str):
    def __init__(self, text):
        self.text = text
        self._letters = None

    @property
    def letters(self):
        return self._letters

    @letters.getter
    def letters(self):
        if self._letters is None:
            chars = [l.lower() for l in set(self.text)]
            self._letters = list(set(chars) & set(LETTERS))
        return self._letters


class Mapping:
    def __init__(self, mapping=None):
        if mapping is None:
            self.mapping = LETTERS  # initialize as a -> a, b -> b, etc.
        else:
            self.mapping = mapping

    def scramble(self):
        self.mapping = ''.join(sample(self.mapping, len(self.mapping)))

    def swap(self, l1, l2, inplace=False):
        tmp = '_'
        mapping = self.mapping\
            .replace(l1, tmp)\
            .replace(l2, l1)\
            .replace(tmp, l2)
        if inplace:
            self.mapping = mapping
        else:
            return Mapping(mapping)

    def random_swap(self, letters=None, inplace=False):
        if letters is None:
            letters = LETTERS
        l1 = sample(letters, 1)[0]
        l2 = sample(set(LETTERS) - set(l1), 1)[0]
        return self.swap(l1, l2, inplace)

    def translate(self, text):
        trans = str.maketrans(self.mapping, LETTERS)
        return type(text)(text.translate(trans))


class Tokenizer:
    def __init__(
        self,
        char_ngram_range=(1, 3),
        word_ngram_range=(1, 1),
        vocab_size=1000000
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
        """Add n-gram tokens to given dictionary of tokens."""
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngrams = self.count_ngrams(lst, n)
            for ngram, count in ngrams.items():
                token = Token(ngram, kind, n)
                tokens[token] += count

    def tokenize(self, text, tokens=None):
        if tokens is None:
            tokens = defaultdict(int)
        text = self.clean_text(text)
        words = self.get_words(text)
        if self.word_ngram_range is not None:
            self.add_ngram_tokens(
                words,
                self.word_ngram_range,
                NgramKind.word,
                tokens
            )
        if self.char_ngram_range is not None:
            for word in words:
                word = '<' + word + '>'
                self.add_ngram_tokens(
                    word,
                    self.char_ngram_range,
                    NgramKind.char,
                    tokens
                )
        return tokens

    def fit(self, texts):
        self.vocab = defaultdict(int)
        for text in tqdm(texts):
            self.vocab = self.tokenize(text, self.vocab)
        self.calc_totals()
        self.subset_vocab()

    def calc_totals(self):
        self.totals = defaultdict(int)
        for token, count in self.vocab.items():
            self.totals[(token.kind, token.n)] += count

    def subset_vocab(self):
        sorted_tups = sorted(self.vocab.items(), key=lambda x: -x[1])
        subsetted = sorted_tups[:self.vocab_size]
        self.vocab = dict(subsetted)


class Solver:
    def __init__(self, tokenizer, text, pseudo_count):
        self.tokenizer = tokenizer
        self.text = text
        self.pseudo_count = pseudo_count

    def score(self, text):
        """Caluclate (mean) negative log likelihood."""
        tokens = self.tokenizer.tokenize(text)
        nll = 0
        for token, count in tokens.items():
            vocab_count = self.tokenizer.vocab.get(token, 0) + self.pseudo_count
            total = self.tokenizer.totals[(token.kind, token.n)]  # could break
            log_prob = log(vocab_count) - log(total)
            nll += -1 * log_prob * count
        return nll / len(tokens)  # take mean

    # def softmax(x, temp=1):
    #     return np.exp(x / temp) / np.sum(np.exp(x / temp), axis=0)

    # Need to add simulated annealing code or other algo here.


class Timer:
    def __init__(self):
        self.t0 = None
        self.t1 = None

    def tic(self):
        self.t0 = time()

    def toc(self):
        self.t1 = time()
        return self.t1 - self.t0


def get_swap_options(letters, p=1):
    combos = set()
    for l1 in letters:
        for l2 in set(LETTERS) - set(l1):
            swap = tuple(sorted([l1, l2]))
            if swap not in combos:
                combos.add(swap)
    n = max(int(len(combos) * p), 1)
    combos = sample(list(combos), n)
    return combos


# -----------------------------------------------------------------------------
# Graveyard
# -----------------------------------------------------------------------------

# def p_scheduler():
#     p = 0.
#     step = 0.01
#     while True:
#         yield p
#         if p > 0.5:
#             p = 1
#         p = min(p + step, 1)
