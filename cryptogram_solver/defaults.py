from cryptogram_solver import utils


PROJECT_DIR = utils.get_project_dir()
CORPUS_PATH = PROJECT_DIR / 'data/processed/corpora/kaggle_news.txt'
# FREQS_PATH = PROJECT_DIR / 'data/raw/freqs/kaggle/unigram_freq.csv'  # https://www.kaggle.com/rtatman/english-word-frequency
FREQS_PATH = PROJECT_DIR / 'data/processed/freqs/kaggle_news.csv'

# Solver defaults.
NUM_ITERS = 10000
CHAR_NGRAM_RANGE = (2, 2)
WORD_NGRAM_RANGE = (1, 1)
N_DOCS = 1000
VOCAB_SIZE = 50000
PSEUDO_COUNT = 1
LOG_TEMP_START = -1
LOG_TEMP_END = -6
LAMB_START = 0.5
LAMB_END = 0
