# cryptogram-solver

## Overview

The goal of this project is to create a tool that can automatically solve [cryptograms](https://en.wikipedia.org/wiki/Cryptogram) of the style that are found in the puzzle section of newspapers. I'll use a statistical method since this I find the optimization problem interesting and would like to learn what set of "hyperparameters" (e.g., regarding the tokenization of the text) work best.

The tool should be able to take a piece of text like this...

OZYEJOK HEZ QAKH DJFIJLFZ YZKIFH AV JFF ZWIGJHRAX RK HEZ JLRFRHT HA QJBZ T AIYKZFV WA HEZ HERXU TAI EJDZ HA WA, MEZX RH AIUEH HA LZ WAXZ, MEZHEZY TAI FRBZ RH AY XAH.

...and return this...

Perhaps the most valuable result of all education is the ability to make yourself do the thing you have to do, when it ought to be done, whether you like it or not.
