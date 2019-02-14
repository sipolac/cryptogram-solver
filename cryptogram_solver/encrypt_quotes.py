#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-02-10

Encrypt quotes file.
"""
import random

from cryptogram_solver import solver
from cryptogram_solver import utils


PROJECT_DIR = utils.get_project_dir()
quotes_path = PROJECT_DIR / 'references' / 'quotes.txt'
quotes_encrypted_path = PROJECT_DIR / 'references' / 'quotes_encrypted.txt'


def main():
    random.seed(1234)
    with open(quotes_path) as f:
        quotes = f.read()
    quotes_encrypted = solver.encrypt(quotes)
    with open(quotes_encrypted_path, 'w') as f:
        f.write(quotes_encrypted)


if __name__ == '__main__':
    main()
