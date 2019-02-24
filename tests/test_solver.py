#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-02-09

Tests for the solver.
"""
import pytest

from cryptogram_solver import defaults
from cryptogram_solver import solver
from cryptogram_solver import utils
from cryptogram_solver.solver import Token


PROJECT_DIR = utils.get_project_dir()
TEST_PATH = PROJECT_DIR / 'models' / 'test'
TEST_VOCAB_SIZE = 9999
N_DOCS = 100


@pytest.fixture
def slv(scope='module'):
    cfg = dict(
        char_ngram_range=defaults.CHAR_NGRAM_RANGE,
        word_ngram_range=defaults.WORD_NGRAM_RANGE,
        vocab_size=TEST_VOCAB_SIZE,
        pseudo_count=defaults.PSEUDO_COUNT,
    )
    slv = solver.Solver(cfg)
    slv.fit(freqs=utils.read_freqs(defaults.FREQS_PATH))
    return slv


@pytest.fixture
def decrypt_kwargs(scope='module'):
    decrypt_kwargs = dict(
        num_iters=defaults.NUM_ITERS,
        log_temp_start=defaults.LOG_TEMP_START,
        log_temp_end=defaults.LOG_TEMP_END,
        lamb_start=defaults.LAMB_START,
        lamb_end=defaults.LAMB_END
    )
    return decrypt_kwargs


@pytest.fixture
def tokenizer(scope='module'):
    tk = solver.Tokenizer(
        char_ngram_range=(1, 2),
        word_ngram_range=(1, 2),
    )
    return tk


def test_tokenizer(tokenizer):
    expected_tokens = {
        Token(ngrams=('a',), kind='word', n=1): 1,
        Token(ngrams=('test',), kind='word', n=1): 1,
        Token(ngrams=('a', 'test'), kind='word', n=2): 1,
        Token(ngrams=('<',), kind='char', n=1): 2,
        Token(ngrams=('a',), kind='char', n=1): 1,
        Token(ngrams=('>',), kind='char', n=1): 2,
        Token(ngrams=('<', 'a'), kind='char', n=2): 1,
        Token(ngrams=('a', '>'), kind='char', n=2): 1,
        Token(ngrams=('t',), kind='char', n=1): 2,
        Token(ngrams=('e',), kind='char', n=1): 1,
        Token(ngrams=('s',), kind='char', n=1): 1,
        Token(ngrams=('<', 't'), kind='char', n=2): 1,
        Token(ngrams=('t', 'e'), kind='char', n=2): 1,
        Token(ngrams=('e', 's'), kind='char', n=2): 1,
        Token(ngrams=('s', 't'), kind='char', n=2): 1,
        Token(ngrams=('t', '>'), kind='char', n=2): 1
    }
    tokens = tokenizer.tokenize('A test!')
    assert tokens == expected_tokens


def test_solver_serialization(slv):
    slv.save(TEST_PATH)
    del slv  # to be explicit
    slv = solver.Solver.load(TEST_PATH)
    assert len(slv.vocab) == TEST_VOCAB_SIZE


def test_solver_simple_cryptogram(slv, decrypt_kwargs):
    encrypted = (
        'KGQU GO QRKK FQ JUWRCN. DFCGSU GC. OPUKK CYU VWGD, WDX '
        'QUUK CYU LGDX. KGTU NFRV KGQU CF CYU QRKKUOC ZFCUDCGWK, '
        'WDX QGEYC QFV NFRV XVUWPO. NFR WVU CYU SUDCUV FQ NFRV '
        'RDGTUVOU, WDX NFR SWD PWMU WDNCYGDE YWZZUD.'
    )
    decrypted_expected = (
        'life is full of beauty. notice it. smell the rain, and '
        'feel the wind. live your life to the fullest potential, '
        'and fight for your dreams. you are the center of your '
        'universe, and you can make anything happen.'
    )
    decrypted = slv.decrypt(encrypted, **decrypt_kwargs)['decrypted']
    assert decrypted_expected == decrypted
