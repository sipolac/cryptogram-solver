from collections import Counter, defaultdict
from enum import Enum
from math import log
from string import ascii_lowercase
from tqdm import tqdm
import random
import re


LETTERS = set(ascii_lowercase)


class NgramKind(Enum):
    word = 1
    char = 2

    def __repr__(self):
        return self.name


class Token:
    def __init__(self, ngrams, kind, n):
        self.ngrams = ngrams
        self.kind = kind
        self.n = n  # n in n-gram

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return str((self.ngrams, self.kind, self.n))

    def __eq__(self, other):
        return str(self) == str(other)


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
        self.add_ngram_tokens(words, self.word_ngram_range, NgramKind.word, tokens)
        for word in words:
            word = '<' + word + '>'
            self.add_ngram_tokens(word, self.char_ngram_range, NgramKind.char, tokens)
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
        self.letters = None
        self.set_letters()

    def set_letters(self):
        letters = re.findall('[a-z]', self.text)
        self.letters = set(letters)

    def swap(self, letter1, letter2):
        """Swap two letter assignments in text."""
        to_swap = letter1 + letter2
        transtab = self.text.maketrans(to_swap, to_swap[::-1])
        swapped = self.text.translate(transtab)
        return swapped

    def swap_random(self):
        """Swap two random letter assignments."""
        letter1 = random.choice(list(self.letters))
        letter2 = random.choice(list(LETTERS - set(letter1)))
        swapped = self.swap(letter1, letter2)
        return swapped

    def scramble(self, n=15):
        for _ in range(n):
            self.__init__(self.swap_random())


# -----------------------------------------------------------------------------
# Code graveyard
# -----------------------------------------------------------------------------

# def get_ngrams(self, lst, ngram_range):
#     """Get n-grams of words (if input is list) or characters (str)."""
#     char = isinstance(lst, str)  # parse char n-grams?
#     tokens = defaultdict(int)
#     for j in range(ngram_range[0], ngram_range[1] + 1):
#         for i in range(len(lst) - j + 1):
#             token = lst[i:i + j]
#             if not char:
#                 token = ' '.join(token)
#             tokens[token] += 1
#     return tokens

# def get_ngrams(lst, n):
#     """Get n-grams of words (if input is list) or characters (str)."""
#     char = isinstance(lst, str)  # parse char n-grams?
#     tokens = defaultdict(int)
#     for i in range(len(lst) - n + 1):
#         token = lst[i:i + n]
#         if not char:
#             token = ' '.join(token)
#         tokens[token] += 1
#     return tokens

# def counts_to_log_probs(self, tokens):
#     self.total = sum(tokens.values())
#     probs = dict()
#     for token, count in tokens.items():
#         probs[token] = log(count / self.total)
#     return probs

# def count_ngrams(self, lst, n):
#     """Get n-grams of words (if input is list) or characters (str)."""
#     ngrams = defaultdict(int)
#     for i in range(len(lst) - n + 1):
#         ngram = tuple(lst[i:i + n])
#         ngrams[ngram] += 1
#     return ngrams
