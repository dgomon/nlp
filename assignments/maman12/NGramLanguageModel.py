import math
import this
from collections import Counter, defaultdict

import nltk
from nltk.corpus import gutenberg

training_data_percentage = 0.9
start_symbol = '<s>'
end_symbol = '</s>'
unknown_symbol = '<UNK>'

class NGramLanguageModel:
    def __init__(self, n):
        self.n: int = n
        self.ngram_probs: dict[tuple[str, ...], dict[str, float]] = {}

        self.tokens: list[str] = []
        self.train_tokens: list[str] = []
        self.test_tokens: list[str] = []

        self.vocabulary: set[str] = set()
        self.history_counts = Counter()
        self.successor_counts = defaultdict(Counter)

    def prepare_data(self, text):
        self.tokens = nltk.tokenize.word_tokenize(text.lower())
        training_text_length = int(len(self.tokens) * training_data_percentage)
        self.train_tokens = self.tokens[:training_text_length]
        self.test_tokens = self.tokens[training_text_length:]

        self.vocabulary = set(self.tokens)
        self.vocabulary.add(unknown_symbol)


    def train(self):
        # count
        for ngram in nltk.ngrams(self.train_tokens, self.n, pad_left=True, pad_right=True,
                                 left_pad_symbol=start_symbol, right_pad_symbol=end_symbol):

            history = ngram[:-1]   # The history string/tuple
            next_word = ngram[-1]  # The word following the history
            self.history_counts[history] += 1
            self.successor_counts[history][next_word] += 1


    def prob(self, word, history, smoothing=None):
        # if history is shorter than n - 1, pad it on the left
        # history = (start_symbol,) * max(0, self.n - len(history)) + history

        if self.history_counts[history] == 0:
            # History not found
            return 0

        if smoothing == 'Laplace':
            return (self.successor_counts[history][word] + 1) / (self.history_counts[history] + len(self.vocabulary))

        # no smoothing
        return self.successor_counts[history][word] / self.history_counts[history]

    def perplexity(self, tokens, smoothing=None):
        prod = 1.0

        num_tokens = len(tokens) - (self.n - 1)
        if num_tokens <= 0:
            return float('inf')

        # skipping first n - 1 tokens such that the length of history of each token in calculation is at least n - 1
        for i in range(self.n - 1, len(tokens)):
            prod *= self.prob(tokens[i], tokens[i - self.n + 1: i], smoothing)
            if prod == 0:
                return float('inf')

        return math.pow(prod, -1 / num_tokens)

if __name__ == '__main__':
    model = NGramLanguageModel(n = 2)

    alice_raw = gutenberg.raw('carroll-alice.txt')
    model.prepare_data(alice_raw)

    model.train()

    result = model.prob("alice", ("[",), smoothing='Laplace')

    print (result)

    perplexity = model.perplexity(("tired", "of", "sitting"))

    pass