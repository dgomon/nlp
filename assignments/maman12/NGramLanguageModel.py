"""Maman 12, Question 5: N-Gram Language Model (starter code).

Provided: constructor, data preparation, and experiment setup.
You should implement: training, probability estimation, perplexity, text generation,
and the requested experiments/analysis.

Install: pip install nltk
Corpus download (once): python -m nltk.downloader gutenberg
"""
import math
import random
from collections import Counter, defaultdict
from typing import Sequence

from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

UNK_TOKEN = "<UNK>"

class NGramLanguageModel:
    """A word-level n-gram language model with optional Laplace smoothing."""

    def __init__(self, n: int):
        if n < 2:
            raise ValueError("n must be at least 2 for this assignment.")

        self.n = n

        self.train_tokens: list[str] = []
        self.test_tokens: list[str] = []
        self.vocab: set[str] = set()

    def prepare_data(self, text: str) -> None:
        """Tokenize the corpus and create a sequential 90/10 train/test split.
        The vocabulary is constructed from training tokens only.
        Unseen test tokens are mapped to <UNK>.
        """
        tokens = word_tokenize(text.lower())
        split_idx = int(0.9 * len(tokens))
        self.train_tokens = tokens[:split_idx]
        self.vocab = set(self.train_tokens)
        self.vocab.add(UNK_TOKEN)

        self.test_tokens = [
            token if token in self.vocab else UNK_TOKEN
            for token in tokens[split_idx:]
        ]

    def train(self) -> None:
        """Count n-grams and the histories that have observed continuations.
        """
        if not self.train_tokens:
            raise ValueError("Call prepare_data(text) before train().")

        self.ngram_counts = Counter()
        self.history_counts = Counter()
        self.next_word_counts = defaultdict(Counter)

        for ngram in ngrams(self.train_tokens, self.n):
            history = ngram[:-1]   # The history string/tuple
            next_word = ngram[-1]  # The word following the history
            self.history_counts[history] += 1
            self.next_word_counts[history][next_word] += 1


    def prob(self, word: str, history: Sequence[str], smoothing=None) -> float:
        """Return P(word | history).
        Args:
            word: Candidate next token.
            history: n-1 preceding tokens.
            smoothing: None (MLE) or 'Laplace' (add-one smoothing).
        """

        if self.history_counts[history] == 0:
            # History not found
            return 0

        if smoothing == 'Laplace':
            return (self.next_word_counts[history][word] + 1) / (self.history_counts[history] + len(self.vocab))

        # no smoothing
        return self.next_word_counts[history][word] / self.history_counts[history]


    def perplexity(self, tokens: Sequence[str], smoothing=None) -> float:
        """Compute perplexity using predictions with full n-1-token histories.
        """
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