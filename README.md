# Task

The goal of this project is to create a tool that can automatically solve [cryptograms](https://en.wikipedia.org/wiki/Cryptogram). That is, the tool take a string like this...

`
"OZYEJOK HEZ QAKH DJFIJLFZ YZKIFH AV JFF ZWIGJHRAX RK HEZ JLRFRHT HA QJBZ T AIYKZFV WA HEZ HERXU TAI EJDZ HA WA, MEZX RH AIUEH HA LZ WAXZ, MEZHEZY TAI FRBZ RH AY XAH."
`

...and return something like this...

`
"perhaps the most valuable result of all education is the ability to make yourself do the thing you have to do, when it ought to be done, whether you like it or not."
`

...by figuring out that _O_ is mapped _p_, _Z_ is mapped to _e_, _Y_ is mapped to _r_, and so on.


# General approach

Although it's perhaps more algorithmically elegant to use a dictionary-based approach (as has been done before), I'll use a statistical method with word- and character-level n-grams. There are two major reasons for this:
- A statistical approach may be an elegant way of avoiding two limitations of dictionary-based approaches, which are that they rely heavily on the completeness of the dictionary and that they don't put a larger "prior" on more common words. (Inspiration: [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606).)
- I find the optimization problem interesting: there are many local minima where the optimizer can get trapped, and this depends quite a bit on the tokenization and scoring schemes. For example, by weighting word-level unigrams very heavily and character-level n-grams lightly, the model may not want to change any letters in a word that the model knows has a non-zero frequency.

Test of syntax highlighting for github.io:
```python
s = "Python syntax highlighting"
print s
```
