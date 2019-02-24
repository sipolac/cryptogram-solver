#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-02-09

Tests utility functions.
"""
from cryptogram_solver import defaults
from cryptogram_solver import utils


def test_read_freqs():
    n = 3
    freqs = utils.read_freqs(defaults.FREQS_PATH, n)
    assert len(freqs) == n
    assert 'the' in freqs


def test_read_docs():
    expected_text = (
        'After the bullet shells get counted, the blood dries and '
        'the votive candles burn out, people peer do'
    )
    text = list(utils.read_docs(defaults.CORPUS_PATH))[1][:100]
    assert text == expected_text
