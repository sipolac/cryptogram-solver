# cryptogram-solver

Solver for [cryptograms](https://en.wikipedia.org/wiki/Cryptogram) (substitution ciphers).

The basic syntax is `python solver.py <CRYPTOGRAM TEXT>`, as shown below. This gives both the cipher and the decyphered text. The `-l` argument loads a pre-fitted solver instead of computing token frequencies from raw data again. The cipher is shown twice: first it's ordered by the original letters and second by the encrypted letters.

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

For decryption, I use an optimization technique called [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing). In my particular implementation, I swap random letters in the mapping and re-score the text, where the score is the negative log likelihood of the token frequencies. I use [softmax](https://en.wikipedia.org/wiki/Softmax_function) on the difference of the scores of the current text and new text to determine whether I want to keep the new mapping. Better mappings are more likely to be kept, but I'm *open to accepting worse mappings* for the sake of escaping local minima. Over time, I decrease the "temperature" softmax parameter and decrease the number of swaps per iteration so that I'm increasingly likely to accept the mappings that improve the score.

Here's the Python-style pseudo(-ish)code for simulated annealing algorithm.

```python
def simulated_annealing(encrypted):
    mapping = Mapping()
    best_mapping = mapping
    best_score = score(encrypted)

    for temp, swaps in zip(temp_list, swap_list):

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

The tokenization works as follows.

```python
>>> from cryptogram_solver import solver
>>> tk = solver.Tokenizer((2, 3), (1, 1))
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

I preferred this approach to a dictionary-based approach. In a dictionary-based approach, you use a pre-built dictionary to solve the puzzle recursively. I find that for my tastes, these dictionary-based approaches rely too heavily on the dictionary and don't consider information that could be useful for solving the problem (e.g., character n-gram frequencies). Also, it isn't clear to me whether a more comprehensive dictionary is necessarily better for the candidate generation process; that is, a more comprehensive dictionary may let you generate more candidates (so your "[recall](https://en.wikipedia.org/wiki/Precision_and_recall)" is higher), but these candidates are more likely to be of lower quality and may slow down the algorithm with little benefit.

You can try both approaches (statistical and dictionary) on [quipqiup.com](https://quipqiup.com/), created by University of Michigan professor [Edwin Olson](https://april.eecs.umich.edu/people/ebolson/). Olsen's implementations result in solve times that are pretty similar.

My approach differs from typical statistical approaches in a few ways:
1. I use a tokenizer that's is mix of both words and characters, so scoring is presumably more fluid. (**TODO**: Explain this better, and also how probabilities are computed by token type (n-gram and kind) meaning their types are weighted appropriately. Also explain that this lets you use less data for computing frequencies.)
1. I include an additional problem-specific simulated annealing parameter: the number of swaps in the mapping per iteration. That is, in the beginning of the optimization, you can swap more letters per iteration instead of just two. But as you near the end of the optimization, the number of swaps decreases and you're left only able to swap two letters.
1. I allow the user to compute frequencies from any data source as opposed to using a pre-computed list of bigram frequencies (for example).
1. I let the user define these parameters as they desire.

**TODO: EXPLAIN THIS BETTER**



\[1\] E.g., [Invent with Python, Chapter 18](https://inventwithpython.com/hacking/chapter18.html) or [aquach's cryptogram-solver GitHub project](https://github.com/aquach/cryptogram-solver)

\[2\] E.g., [theikkila's substitution-cipher-SA-solver GitHub project](https://github.com/theikkila/substitution-cipher-SA-solver)


# OTHER TODOS
- Finish this README!
- Create function for preprocessing data.
- Use random search to find better set of parameters.
- See if pre-computing log probabilities results in a significant speedup.
- See if turning tokens into joined strings speeds things up.
