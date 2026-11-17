from collections import Counter, defaultdict

import nltk
from nltk.corpus import gutenberg

training_data_percentage = 0.9
start_symbol = '<s>'
end_symbol = '</s>'
unknown_symbol = '<UNK>'

class NGramLanguageModel:
    def __init__(self, n):
        self.history_length: int = n
        self.ngram_probs: dict[tuple[str, ...], dict[str, float]] = {}

        self.tokens: list[str] = []
        self.train_tokens: list[str] = []
        self.test_tokens: list[str] = []

        self.vocabulary: set[str] = set()
        self.ngram_counts = defaultdict(Counter)

    def prepare_data(self, text):
        self.tokens = nltk.tokenize.word_tokenize(text.lower())
        training_text_length = int(len(self.tokens) * training_data_percentage)
        self.train_tokens = self.tokens[:training_text_length]
        self.test_tokens = self.tokens[training_text_length:]

        self.vocabulary = set(self.tokens)
        self.vocabulary.add(unknown_symbol)


    def train(self):
        # count
        for ngram in nltk.ngrams(self.train_tokens, self.history_length + 1, pad_left=True, pad_right=True,
                                 left_pad_symbol=start_symbol, right_pad_symbol=end_symbol):

            history = ngram[:-1]   # The history string/tuple
            next_word = ngram[-1]  # The word following the history
            self.ngram_counts[history][next_word] += 1

        # normalize into a probability table

        for history, counter in self.ngram_counts.items():
            total_count = sum(counter.values())
            # Map each word to its Maximum Likelihood Estimation (MLE) probability
            self.ngram_probs[history] = {word: count / total_count for word, count in counter.items()}


    def prob(self, word, history, smoothing=None):
        # if history is shorter than n - 1, pad it on the left
        history = (start_symbol,) * max(0, self.history_length - len(history)) + history

        # Retrieve the Counter of following words for this history
        following_counts = self.ngram_counts.get(history)

        if not following_counts:
            # Handle unseen histories (fallback, uniform random, or backoff)
            raise ValueError(f"History {history} not found in model.")

        return self.ngram_probs.get(history).get(word, 0.0)


if __name__ == '__main__':
    model = NGramLanguageModel(n = 2)


    alice_raw = gutenberg.raw('carroll-alice.txt')
    model.prepare_data(alice_raw)

    model.train()

    result = model.prob("alice", ("[",))

    pass