## Overview

The goal of this project is to create a tool that can automatically solve [cryptograms](https://en.wikipedia.org/wiki/Cryptogram), where are puzzles often found in newspapers. The tool should be able to take a piece of text like this...

> OZYEJOK HEZ QAKH DJFIJLFZ YZKIFH AV JFF ZWIGJHRAX RK HEZ JLRFRHT HA QJBZ T AIYKZFV WA HEZ HERXU TAI EJDZ HA WA, MEZX RH AIUEH HA LZ WAXZ, MEZHEZY TAI FRBZ RH AY XAH.

...and return something like this...

> perhaps the most valuable result of all education is the ability to make yourself do the thing you have to do, when it ought to be done, whether you like it or not.


## Approach

It's common and intuitive to use a dictionary-based approach for this task. However, I'll want to use a statistical method for two related reasons:
- I wanted to experiment with a tokenization process that combines word- and character-level n-grams, as written about Facebook AI's paper [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606). This may be an elegant way of avoiding two major limitations of dictionary-based approaches, which are that they rely heavily on the completeness of the dictionary and that they do not set a larger "prior" on more common words.
- I find the optimization problem interesting: there are many local minima where the model can get trapped, and this depends quite a bit on the tokenization scheme. For example, by weighting word-level unigrams very heavily, the model may not want to change any letters in a word that has a non-zero frequency in the training set.
