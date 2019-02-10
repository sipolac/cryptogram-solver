# cryptogram-solver

Solver for cryptograms (substitution ciphers).

![](references/demo.gif)

# TODO
- Add two arguments for temperature and one for number of characters (just need to specify the start)
- Finish this README!
  - Show example of tokenizer.
  - Explain simulated annealing.
  - Explain why this is different from other implementations:
    - Tokenizer is mix of both words and chars, so scoring is presumably better. (Explain how probabilities are computed by token type (n-gram and kind) meaning their types are weighted appropriately.)
    - Simulated annealing is controlled by two parameters: temperature and number of swaps.
    - Frequencies can be computed from any data source as opposed to using a pre-computed list of bigram frequencies (for example).
    - All these parameters are user-defined.
  - Explain arguments to `main` solver function.