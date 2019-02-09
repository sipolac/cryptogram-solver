#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-02-09

Basic test of the solver with a simple cryptogram.

This cryptogram is simple enough that it should be solved almost every time.
"""
from cryptogram_solver import data
from cryptogram_solver import solver


def test_simple_cryptogram():
    encrypted_raw = (
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

    # Define tokenizer and solver.
    tokenizer = solver.Tokenizer(
        char_ngram_range=(2, 3),
        word_ngram_range=(1, 1)
    )
    slv = solver.Solver(tokenizer, vocab_size=10000, pseudo_count=1)

    # Train solver.
    docs = data.get_news_articles()
    slv.fit(docs[:100])

    # Test on text.
    encrypted = solver.clean_text(encrypted_raw)
    decrypted = slv.decrypt(encrypted, 10000)
    assert decrypted_expected == decrypted
