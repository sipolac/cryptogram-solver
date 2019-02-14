# cryptogram-solver

Solver for [cryptograms](https://en.wikipedia.org/wiki/Cryptogram) (substitution ciphers).

![](references/demo.gif)


# Arguments

See help: `python solver.py -h`

```
usage: solver.py [-h] [-i NUM_ITERS] [-c CHAR_NGRAM_RANGE CHAR_NGRAM_RANGE]
                 [-w WORD_NGRAM_RANGE WORD_NGRAM_RANGE] [-n N_DOCS]
                 [-b VOCAB_SIZE] [-p PSEUDO_COUNT]
                 [--log_temp_start LOG_TEMP_START]
                 [--log_temp_end LOG_TEMP_END] [--swaps_start SWAPS_START]
                 [--swaps_end SWAPS_END] [-l] [-s] [-v]
                 text

positional arguments:
  text                  text to be decrypted

optional arguments:
  -h, --help            show this help message and exit
  -i NUM_ITERS, --num_iters NUM_ITERS
                        number of iterations during simulated annealing
                        process
  -c CHAR_NGRAM_RANGE CHAR_NGRAM_RANGE, --char_ngram_range CHAR_NGRAM_RANGE CHAR_NGRAM_RANGE
                        range of character n-grams to use in tokenization
  -w WORD_NGRAM_RANGE WORD_NGRAM_RANGE, --word_ngram_range WORD_NGRAM_RANGE WORD_NGRAM_RANGE
                        range of word n-grams to use in tokenization
  -n N_DOCS, --n_docs N_DOCS
                        number of documents used to estimate token frequencies
  -b VOCAB_SIZE, --vocab_size VOCAB_SIZE
                        size of vocabulary to use for scoring
  -p PSEUDO_COUNT, --pseudo_count PSEUDO_COUNT
                        number added to all token frequencies for smoothing
  --log_temp_start LOG_TEMP_START
                        log of initial temperature
  --log_temp_end LOG_TEMP_END
                        log of final temperature
  --swaps_start SWAPS_START
                        number of letter swaps made per iteration in beginning
  --swaps_end SWAPS_END
                        number of letter swaps made per iteration at end
  -l, --load_solver     load pre-fitted solver
  -s, --save_solver     save fitted solver for use later
  -v, --verbose         verbose output for showing solve process
```

# What is a cryptogram?

Let's go with [Wikipedia's definition](https://en.wikipedia.org/wiki/Cryptogram):

> A cryptogram is a type of puzzle that consists of a short piece of encrypted text. Generally the cipher used to encrypt the text is simple enough that the cryptogram can be solved by hand. Frequently used are substitution ciphers where each letter is replaced by a different letter or number. To solve the puzzle, one must recover the original lettering. Though once used in more serious applications, they are now mainly printed for entertainment in newspapers and magazines.

For example, let's say you're given the puzzle below.

> "TNJZTPJVTSP YJX ZTBCP VS NJP VS HSNACPXJVC ITN WSM YIJV IC TX PSV, JPG J XCPXC SW IKNSM YJX AMSBTGCG VS HSPXSEC ITN WSM YIJV IC TX." -SXHJM YTEGC
 
 The goal is to realize that _i_'s were replaced with _T_'s, _m_'s with _N_'s, and so on for all the letters of the alphabet. Once you make all the correct substitutions, you get the following text.

 > "Imagination was given to man to compensate him for what he is not, and a sense of humor was provided to console him for what he is." -Oscar Wilde

By hand, you'd use heuristics to solve cryptograms iteratively. E.g., if you see _ZXCVB'N_, you might guess that _N_ is _t_ or _s_. You might also guess that if _X_ appears a lot in the text, it might be a common letter like _e_.

But having a computer solve this is tricky.  You can't brute-force your way through a cryptogram since there are 26! = 403,291,461,126,605,635,584,000,000 different mappings. (That is, the letter _a_ could map to one of 26 letters, _b_ could map to 25, and so on.) And there isn't a surefire way to tell if you've found the correct mapping.


# Approach

## Tokenizer

I use a tokenizer that can generate both character n-grams and word n-grams. The code example below shows how the tokenizer creates character bigrams and trigrams as well as word unigrams.

```python
>>> from cryptogram_solver import solver
>>> tk = solver.Tokenizer(char_ngram_range=(2, 3), word_ngram_range=(1, 1))
>>> tokens = tk.tokenize('Hello world!')
>>> print(*tokens, sep='\n')
Token(ngrams=('hello',), kind='word', n=1)
Token(ngrams=('world',), kind='word', n=1)
Token(ngrams=('<', 'h'), kind='char', n=2)
Token(ngrams=('h', 'e'), kind='char', n=2)
Token(ngrams=('e', 'l'), kind='char', n=2)
Token(ngrams=('l', 'l'), kind='char', n=2)
Token(ngrams=('l', 'o'), kind='char', n=2)
Token(ngrams=('o', '>'), kind='char', n=2)
Token(ngrams=('<', 'h', 'e'), kind='char', n=3)
Token(ngrams=('h', 'e', 'l'), kind='char', n=3)
Token(ngrams=('e', 'l', 'l'), kind='char', n=3)
Token(ngrams=('l', 'l', 'o'), kind='char', n=3)
Token(ngrams=('l', 'o', '>'), kind='char', n=3)
Token(ngrams=('<', 'w'), kind='char', n=2)
Token(ngrams=('w', 'o'), kind='char', n=2)
Token(ngrams=('o', 'r'), kind='char', n=2)
Token(ngrams=('r', 'l'), kind='char', n=2)
Token(ngrams=('l', 'd'), kind='char', n=2)
Token(ngrams=('d', '>'), kind='char', n=2)
Token(ngrams=('<', 'w', 'o'), kind='char', n=3)
Token(ngrams=('w', 'o', 'r'), kind='char', n=3)
Token(ngrams=('o', 'r', 'l'), kind='char', n=3)
Token(ngrams=('r', 'l', 'd'), kind='char', n=3)
Token(ngrams=('l', 'd', '>'), kind='char', n=3)
```


## Scoring

To compare possible decryptions, I use the negative log likelihood of the token probabilities. Token probabilities are computed by token "type" (word/character and n-gram count combination). For example, to compute the probability of a character bigram, I divide the frequency of that bigram by the total number of character bigrams. This is done when "fitting" the solver to data (see `Solver.fit()`).

You can think of score as error, so a lower score is better.


## Optimization

For the optimization algorithm, I use [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing). The algorithm is run for a pre-defined number of iterations, where in each iteration I swap random letters in the mapping and re-score the text. I use [softmax](https://en.wikipedia.org/wiki/Softmax_function) on the difference of the scores of the current text and new text to determine whether I want to keep the new mapping. Better mappings are more likely to be kept, but I'm *open to accepting worse mappings* for the sake of escaping local minima. Over time, I decrease a softmax parameter called temperature and decrease the number of swaps per iteration so that I'm increasingly likely to accept the mappings that improve the score.

My intuition tells me that character n-grams do the heavy lifting for most of the optimization, while the word n-grams help the algorithm "lock in" on good mappings at the end.

```python
def simulated_annealing(encrypted, num_iters):
    """Python-style pseudo(-ish)code for simulated annealing algorithm."""
    mapping = Mapping()
    best_mapping = mapping
    best_score = score(encrypted)

    for i in range(num_iters):
        temp = temp_list[i]  # defined beforehand
        swaps = swap_list[i]

        new_mapping = mapping.random_swap(swaps)
        new_text = new_mapping.translate(encrypted)
        score = score(new_text)

        score_change = score - best_score

        if exp(-score_change / temp) > uniform(0, 1):
            best_mapping = new_mapping
            best_score = score

        mapping = best_mapping

    decrypted = mapping.translate(encrypted)
    return mapping, decrypted
```

## Alternative approach

I preferred this approach to a dictionary-based approach. In a dictionary-based approach, you use a pre-built dictionary to solve the cryptogram recursively. I decided not to go with this approach for a few reasons:
1. It relies too heavily on the dictionary and doesn't consider information that could be useful for solving the problem (e.g., character n-gram frequencies).
1. It isn't clear to me whether a more comprehensive dictionary is necessarily better for the candidate generation process. that is, a more comprehensive dictionary may let you generate more candidates (so your "[recall](https://en.wikipedia.org/wiki/Precision_and_recall)" is higher), but these candidates are more likely to be of lower quality and may slow down the algorithm with little benefit.
1. It just isn't as theoretically interesting as simulated annealing.

You can try both approaches (statistical and dictionary) on [quipqiup.com](https://quipqiup.com/), created by University of Michigan professor [Edwin Olson](https://april.eecs.umich.edu/people/ebolson/). Olsen's implementations result in solve times that are pretty similar.


# OTHER TODOS
1. Create function for preprocessing data.
1. Use random search to find better set of parameters.
1. See if pre-computing log probabilities results in a significant speedup.
1. See if turning tokens into joined strings speeds things up.
1. Finish this README!
