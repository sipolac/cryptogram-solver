#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-01-25

Basic utility functions.
"""
from pathlib import Path
from time import time
import sys


class TicToc:
    def __init__(self):
        self.t0 = None
        self.t1 = None

    def tic(self):
        self.t0 = time()

    def toc(self):
        self.t1 = time()
        return self.t1 - self.t0


def get_project_dir():
    file_stem = 'cryptogram-solver'
    parents = Path(__file__).resolve().parents
    for filepath in parents:
        if filepath.stem == file_stem:
            break
    return filepath
