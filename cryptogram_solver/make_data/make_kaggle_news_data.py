#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-02-07

Functions for making a corpus .txt file.

Assumes you've downloaded news data from here...

    https://www.kaggle.com/snapcrack/all-the-news

...and saved the zip files here...

    data/raw/corpora/kaggle_news/

Note that the site requires a login for download.
"""
from pathlib import Path
import argparse
import codecs
import csv
import logging
import re
import sys
import zipfile

from tqdm import tqdm

from cryptogram_solver import defaults


# https://www.kaggle.com/snapcrack/all-the-news
KAGGLE_NEWS_RAW_DIR = PROJECT_DIR / 'data/raw/corpora/kaggle_news'


def write_news_articles(dirpath, outfile, n, logger=None):
    logger = logger or logging.getLogger(__name__)

    n_per_file = n / 3 + 1  # approx
    csv.field_size_limit(sys.maxsize)
    docs = list()
    for article_num in range(1, 4):
        filename = f'articles{article_num}.csv'
        logging.info(f'reading {filename}...')
        zipname = dirpath / f'{filename}.zip'
        with zipfile.ZipFile(zipname) as z:
            with z.open(filename, 'r') as f:
                csvfile = csv.reader(codecs.iterdecode(f, 'utf-8'))
                for row, line in tqdm(enumerate(csvfile)):
                    if row == n_per_file:
                        break
                    docs.append(line[-1])  # get content column

    logger.info('substituting all whitespace for space characters...')
    docs = [re.sub(r'\s', ' ', doc) for doc in tqdm(docs)]

    docs = docs[1:]  # remove column header
    docs = docs[:n]  # remove extras from rounding error

    logger.info('writing to file...')
    outfile.parent.mkdir(exist_ok=True, parents=True)
    if outfile.exists():
        outfile.unlink()
    with open(outfile, 'w') as f:
        for doc in docs:
            f.write(f'{doc}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirpath',
                        default=KAGGLE_NEWS_RAW_DIR,
                        type=Path)
    parser.add_argument('--outfile',
                        default=defaults.CORPUS_PATH,
                        type=Path)
    parser.add_argument('-n', '--num_docs', default=10000, type=int)
    args = parser.parse_args()

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level='INFO', format=log_fmt)

    write_news_articles(args.dirpath, args.outfile, args.num_docs)


if __name__ == '__main__':
    main()
