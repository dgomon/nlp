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

            self.ngram_counts[ngram] += 1
            self.history_counts[history] += 1
            self.next_word_counts[history][next_word] += 1


    def prob(self, word: str, history: Sequence[str], smoothing=None) -> float:
        """Return P(word | history).
        Args:
            word: Candidate next token.
            history: n-1 preceding tokens.
            smoothing: None (MLE) or 'Laplace' (add-one smoothing).
        """

        history_tuple = tuple(history)

        if self.history_counts[history_tuple] == 0:
            # History not found
            return 0

        if smoothing == 'Laplace':
            return (self.next_word_counts[history_tuple][word] + 1) / (self.history_counts[history_tuple] + len(self.vocab))

        # no smoothing
        return self.next_word_counts[history_tuple][word] / self.history_counts[history_tuple]


    def perplexity(self, tokens: Sequence[str], smoothing=None) -> float:
        """Compute perplexity using predictions with full n-1-token histories.
        """
        num_tokens = len(tokens) - (self.n - 1)
        if num_tokens <= 0:
            return float('inf')

        log_sum = 0.0 # accumulate log(prob) for numerical stability
        # skip first n - 1 tokens such that the length of history of each token in calculation is at least n - 1
        for i in range(self.n - 1, len(tokens)):
            prob = self.prob(tokens[i], tokens[i - self.n + 1: i], smoothing)
            if prob == 0:
                return float('inf')
            log_sum += math.log(prob)

        return math.exp(-log_sum / num_tokens)

    def generate(self,
        history: Sequence[str],
        max_tokens: int,
        strategy: str,
        temperature: float = 1.0,
    ) -> list[str]:
        result = []

        """Generate up to max_tokens new tokens from an initial history.
        Strategies: 'greedy', 'random', 'temperature'.
        Use MLE probabilities (no smoothing) for generation.
        Stop if the current history has no observed continuations.
        Return only the generated tokens, excluding the initial history.
        """
        for i in range (max_tokens):
            next_word_counts = self.next_word_counts[tuple(history)]

            if next_word_counts == {}:
                break

            if strategy == 'random':
                words, weights = zip(*next_word_counts.items())
                next_word = random.choices(words, weights=weights, k=1)[0]
            elif strategy == 'temperature':
                words, counts = zip(*next_word_counts.items())
                weights = [c ** (1.0 / temperature) for c in counts]
                next_word = random.choices(words, weights=weights, k=1)[0]
            else: # defaulting to greedy
                next_word = next_word_counts.most_common(1)[0][0]

            result.append(next_word)
            history = tuple(history[1:]) + (next_word,)

        return result


def main() -> None:
    text = gutenberg.raw("carroll-alice.txt")
    random.seed(42)  # For reproducible text-generation experiments

    models: dict[int, NGramLanguageModel] = {}
    stats = []

    for n in (2, 3, 4):
        model = NGramLanguageModel(n)
        model.prepare_data(text)
        model.train()
        models[n] = model

        stats.append({
            "n": n,
            "train_tokens": len(model.train_tokens),
            "test_tokens": len(model.test_tokens),
            "vocab_size": len(model.vocab),
            "ngram_count": len(model.ngram_counts),
            "train_mle": model.perplexity(model.train_tokens, smoothing=None),
            "test_mle": model.perplexity(model.test_tokens, smoothing=None),
            "train_laplace": model.perplexity(model.train_tokens, smoothing='Laplace'),
            "test_laplace": model.perplexity(model.test_tokens, smoothing='Laplace'),
        })

    header = f"{'n':<4} | {'Train (MLE)':<12} | {'Test (MLE)':<12} | {'Train (Laplace)':<15} | {'Test (Laplace)':<15}"
    print(header)
    print("-" * len(header))

    for s in stats:
        print(
            f"{s['n']:<4} | "
            f"{s['train_mle']:<12.4f} | "
            f"{s['test_mle']:<12.4f} | "
            f"{s['train_laplace']:<15.4f} | "
            f"{s['test_laplace']:<15.4f}"
        )

    # TODO: doc: reason for infinite perplexity: word "speaker" appearing first on line 3219 (from 3600 total)

    generated = models[3].generate(["she", "said"],
                                   max_tokens = 100,
                                   strategy="temperature",
                                   temperature=1.0)

    print(generated)

    # TODO: Choose a fixed, observed initial history. Generate
    # with greedy, random, and temperature sampling for one model.
    # Compare repetitiveness, diversity, and coherence.

    # TODO: Choose one sampling strategy and compare n = 2,3,4.
    # Use the same initial text, choosing its last n-1 tokens for each model.
    # Explain how model order affects the generated text.



# def main() -> None:
#     model = NGramLanguageModel(n = 2)
#
#     alice_raw = gutenberg.raw('carroll-alice.txt')
#     model.prepare_data(alice_raw)
#
#     model.train()
#
#     result = model.prob("alice", ("[",), smoothing='Laplace')
#
#     print (result)
#
#     perplexity = model.perplexity(("tired", "of", "sitting"))
#
#     pass

if __name__ == "__main__":
    main()
