#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-02-09

Tests for reading read_data.
"""
from cryptogram_solver import defaults
from cryptogram_solver import read_data


def test_read_freqs():
    freqs = read_data.read_freqs(defaults.FREQS_PATH, 3)
    assert 'the' in freqs


def test_read_docs():
    expected_text = (
        'After the bullet shells get counted, the blood dries and '
        'the votive candles burn out, people peer do'
    )
    text = list(read_data.read_docs(defaults.CORPUS_PATH))[1][:100]
    assert text == expected_text
