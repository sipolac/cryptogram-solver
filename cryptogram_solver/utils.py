#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-01-25

Basic utility functions.
"""
from math import exp
from pathlib import Path
from random import uniform


def get_project_dir():
    file_stem = 'cryptogram-solver'
    parents = Path(__file__).resolve().parents
    for filepath in parents:
        if filepath.stem == file_stem:
            break
    return filepath


def impute_defaults(d, default_d):
    """Impute values of dictionary with defaults if key doesn't exist."""
    assert all([k in default_d for k in d])
    for k, v in default_d.items():
        if k not in d:
            d[k] = v
    return d


def linspace(start, stop, num):
    """Return evenly spaced numbers over a specified interval.

    This is a simple version of numpy's `linspace`:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
    """
    step_size = (stop - start) / (num - 1)
    return (start + step_size * i for i in range(num))


def rpoisson(lamb):
    """Generate random number from poisson distribution.

    Don't want to use scipy or numpy as a dependency. Make so that when
    lambda is 0, output is 0.

    https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
    """
    if lamb == 0:
        return 0
    L = exp(-lamb)
    k = 0
    p = 1
    while p > L:
        k += 1
        p *= uniform(0, 1)
    return k - 1
