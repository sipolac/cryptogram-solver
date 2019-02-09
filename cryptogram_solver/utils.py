#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-01-25

Basic utility functions.
"""
from pathlib import Path


def get_project_dir():
    file_stem = 'cryptogram-solver'
    parents = Path(__file__).resolve().parents
    for filepath in parents:
        if filepath.stem == file_stem:
            break
    return filepath


def linspace(start, stop, num):
    """Basic functionality of numpy's `linspace`."""
    step_size = (stop - start) / (num - 1)
    return [start + step_size * i for i in range(num)]
