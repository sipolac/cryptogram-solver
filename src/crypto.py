from collections import Counter, defaultdict, namedtuple
from enum import Enum
from math import log
from random import shuffle
from string import ascii_lowercase
from tqdm import tqdm
import re


LETTERS = set(ascii_lowercase)


Token = namedtuple('Token', 'ngrams kind n')


class NgramKind(Enum):
    word, char = range(2)


class Tokenizer:
    def __init__(
        self,
        char_ngram_range=(1, 6),
        word_ngram_range=(1, 2),
        pseudo_count=1,
        vocab_size=10000
    ):
        self.char_ngram_range = char_ngram_range
        self.word_ngram_range = word_ngram_range
        self.pseudo_count = pseudo_count
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
            tokens = defaultdict(lambda: self.pseudo_count)
        text = self.clean_text(text)
        words = self.get_words(text)
        self.add_ngram_tokens(
            words,
            self.word_ngram_range,
            NgramKind.word,
            tokens
        )
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
        self.vocab = defaultdict(lambda: self.pseudo_count)
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

    def calc_nll(self, text):
        """Caluclate mean negative log likelihood."""
        tokens = self.tokenize(text)
        nll = 0
        for token, count in tokens.items():
            try:
                vocab_count = self.vocab[token]
            except KeyError:
                vocab_count = self.pseudo_count
            total = self.totals[(token.kind, token.n)]  # this could break
            prob = vocab_count / total
            nll += -1 * log(prob) * count
        return nll / len(tokens)  # take mean


class Doc:
    def __init__(self, text):
        self.text = text.lower()
        self.mapping = LETTERS  # initialize as a=a, b=b, etc.

    def swap(self, l1, l2):
        """Swap two letters in mapping."""
        tmp = '_'
        self.mapping = self.mapping\
            .replace(l1, tmp)\
            .replace(l2, l1)\
            .replace(tmp, l2)

    def scramble(self):
        shuffle(self.mapping)

    def translate(self):
        trans = ''.maketrans(self.mapping, LETTERS)
        return self.text.translate(trans)
