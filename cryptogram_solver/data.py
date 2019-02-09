#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Chris Sipola
Created: 2019-02-07

Functions for getting and manipulating training data.
"""
from pathlib import Path
import codecs
import csv
import sys
import zipfile


def get_news_articles(num_articles=10000):
    articles_per_file = num_articles / 3 + 1  # approx
    csv.field_size_limit(sys.maxsize)
    articles = list()
    for article_num in range(1, 4):
        zipname = Path('..', 'data', 'raw', f'articles{article_num}.csv.zip')
        filename = f'articles{article_num}.csv'
        with zipfile.ZipFile(zipname) as z:
            with z.open(filename, 'r') as f:
                csvfile = csv.reader(codecs.iterdecode(f, 'utf-8'))
                for row, line in enumerate(csvfile):
                    if row == articles_per_file:
                        break
                    articles.append(line[-1])  # get content column

    articles = articles[1:]  # remove column header
    articles = articles[:num_articles]  # remove extras from rounding error
    return articles
