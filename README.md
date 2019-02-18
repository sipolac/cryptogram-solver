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
                 [--log_temp_end LOG_TEMP_END] [--lamb_start LAMB_START]
                 [--lamb_end LAMB_END] [-l] [-s] [-v]
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
  --lamb_start LAMB_START
                        poisson lambda for number of letter swaps in beginning
  --lamb_end LAMB_END   poisson lambda for number of letter swaps at end
  -l, --load_solver     load pre-fitted solver
  -s, --save_solver     save fitted solver for use later
  -v, --verbose         verbose output for showing solve process
```


# Examples

To fit a solver on 1000 documents (`-n 1000`) with a tokenizer that uses character trigrams (`-c 3 3`) and word unigrams (`-w 1 1`) with a max vocab size of 5000 (`-b 5000`), save it to file (`-s`) (right now just to `data/cache/`) , and solve the encrypted text (represented here as `<ENCRYPTED TEXT> `):

    python solver.py <ENCRYPTED TEXT> -n 1000 -c 3 3 -w 1 1 -b 5000 -s

To load a fitted solver (`-l`) and run for 5000 iterations (`-i 5000`) with a starting lambda (for character swaps; see below) of 1 (`--lamb_start 1`) and verbose output (`-v`):

    python solver.py <ENCRYPTED TEXT> -l -i 5000 --lamb_start 1 -v

The default settings tend to work well most of the time. Usually you only need to specify the encrypted text, saving (`-s`), loading (`-l`) and the number of iterations (`-i`, which you can set to be lower for longer decryptions). At some point I'd like to do some "hyperparameter" optimization to determine better default settings.


# What is a cryptogram?

Let's go with [Wikipedia's definition](https://en.wikipedia.org/wiki/Cryptogram):

> A cryptogram is a type of puzzle that consists of a short piece of encrypted text. Generally the cipher used to encrypt the text is simple enough that the cryptogram can be solved by hand. Frequently used are substitution ciphers where each letter is replaced by a different letter or number. To solve the puzzle, one must recover the original lettering. Though once used in more serious applications, they are now mainly printed for entertainment in newspapers and magazines.

For example, let's say you're given the puzzle below.

> "SNDVSODTSBO LDF VSHYO TB NDO TB EBNRYOFDTY KSN PBX LKDT KY SF OBT, DOC D FYOFY BP KZNBX LDF RXBHSCYC TB EBOFBJY KSN PBX LKDT KY SF." -BFEDX LSJCY
 
 The goal is to realize that _i_'s were replaced with _S_'s, _m_'s with _N_'s, and so on for all the letters of the alphabet. Once you make all the correct substitutions, you get the following text.

 > "Imagination was given to man to compensate him for what he is not, and a sense of humor was provided to console him for what he is." -Oscar Wilde

By hand, you'd use heuristics to solve cryptograms iteratively. E.g., if you see _ZXCVB'N_, you might guess that _N_ is _t_ or _s_. You might also guess that if _X_ appears a lot in the text, it might be a common letter like _e_.

But having a computer solve this is tricky.  You can't brute-force your way through a cryptogram since there are 26! = 403,291,461,126,605,635,584,000,000 different mappings. (That is, the letter _a_ could map to one of 26 letters, _b_ could map to 25, and so on.) And there isn't a surefire way to tell if you've found the correct mapping.


# Method

## Tokenizer

I use a tokenizer that can generate both character n-grams and word n-grams. The code example below shows how the tokenizer creates character bigrams and trigrams as well as word unigrams.

```python
>>> from cryptogram_solver import solver
>>> tk = solver.Tokenizer(char_ngram_range=(2, 3), word_ngram_range=(1, 1))
>>> tokens = tk.tokenize('Hello!')
>>> print(*tokens, sep='\n')
Token(ngrams=('hello',), kind='word', n=1)
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
```


## Scoring

To score decryptions, I use the negative log likelihood of the token probabilities. Token probabilities are computed by token "type" (word/character and n-gram count combination). For example, to compute the probability of a character bigram, I divide the frequency of that bigram by the total number of character bigrams (and not, say, character trigrams or word unigrams). This is done when "fitting" the solver to data (see `Solver.fit()`).

You can think of score as error, so lower is better.


## Optimization

For the optimization algorithm, I use [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing). The algorithm is run for a pre-defined number of iterations, where in each iteration I swap random letters in the mapping and re-score the text. I use softmax on the difference of the scores of the current text and new text to determine whether I want to keep the new mapping. Better mappings are more likely to be kept, but I'm open to accepting worse mappings for the sake of exploration and escaping local minima. Over the course of the optimization, I decrease temperature (exponentially) so that I'm decreasingly likely to accept mappings that hurt the score. I also decrease the number of swaps per iteration (probabilistically, using the poisson process described below) to encourage exploration in the beginning and fine tuning at the end.

My intuition tells me that character n-grams do the heavy lifting for most of the optimization, while the word n-grams help the algorithm "lock in" on good mappings at the end.

```python
def simulated_annealing(encrypted, num_iters):
    """Python-style pseudo(-ish)code for simulated annealing algorithm."""
    mapping = Mapping()
    best_mapping = mapping
    best_score = score(encrypted)

    for i in range(num_iters):
        temp = temp_list[i]  # from scheduler
        num_swaps = swap_list[i]  # from scheduler

        new_mapping = mapping.random_swap(num_swaps)
        new_text = new_mapping.translate(encrypted)
        score = score(new_text)

        score_change = score - best_score

        if exp(-score_change / temp) > uniform(0, 1):  # softmax
            best_mapping = new_mapping
            best_score = score

        mapping = best_mapping

    decrypted = mapping.translate(encrypted)
    return mapping, decrypted
```

To decrease the number of swaps over time, I use a poisson distribution with a lambda parameter that decreases linearly. The scheduler for the number of swaps is

```python
def schedule_swaps(lamb_start, lamb_end, n):
    for l in linspace(lamb_start, lamb_end, n):
        yield poisson(l) + 1 
```

where `lamb_start` is the starting lambda (at the beginning of the optimization), `lamb_end` is the ending lambda, `n` is the number of iterations in the optimization, `linspace` is a function that returns evenly spaced numbers over a specified interval (a basic version of [numpy's implementation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html)), and `poisson` is a function that draws a sample from a poisson distribution given a lambda parameter. Note that a lambda greater than zero gives you *additional* swaps; if `lamb_start` and `lamb_end` are both zero, then in every iteration you swap only once.


## Alternative approach

The main alternative to a statistical approach is a dictionary-based approach, where you solve recursively word-by-word—keeping track of all possibilities—and score using something like negative log likelihood. I didn't go down this route because I just found simulated annealing much more theoretically appealing.

You can try both approaches (statistical and dictionary) on [quipqiup.com](https://quipqiup.com/), created by University of Michigan professor [Edwin Olson](https://april.eecs.umich.edu/people/ebolson/). Olsen's implementations result in solve times that are pretty similar.


# Dependencies

- [tqdm](https://github.com/tqdm/tqdm) for progress bars

# TODOS
1. Create function for preprocessing data.
1. Use random search to find better set of parameters.
1. See if pre-computing log probabilities results in a significant speedup.
1. See if turning tokens into joined strings speeds things up.
1. Finish this README!
