import re
from collections import Counter
from itertools import chain, pairwise
from typing import List, Tuple
import nltk
from nltk.corpus import gutenberg



class BPETokenizer:
    """Byte Pair Encoding (BPE) Tokenizer implementation.

    Maintains internal vocabulary and merge rules learned during training.
    """
    pattern = r"[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9\s]"

    def __init__(self):
        self.vocabulary = None
        self.bpe_ranks: dict[tuple[str, str], int] = None
        self.pre_tokens_counter = None
        self.corpus = None


    def prepare_corpus(self, text: str) -> List[str]:
        pre_tokens = self._pre_tokenize(text)
        self.corpus = Counter(tuple(token) for token in pre_tokens)

    def merge_tuple(self, word_tuple, t_l, t_r, merged_token):
        new_word = []
        i = 0
        n = len(word_tuple)
        while i < n:
          if i < n - 1 and word_tuple[i] == t_l and word_tuple[i + 1] == t_r:
            new_word.append(merged_token)
            i += 2
          else:
            new_word.append(word_tuple[i])
            i += 1
        return tuple(new_word)

    def train_bpe(self, n_merges):
        # V <- all unique characters in C
        self.vocabulary = dict.fromkeys(chain.from_iterable(self.corpus))
        self.bpe_ranks = {}  # (t_l, t_r) -> merge order index

        for step in range(n_merges):
            # t_L, t_R <- Most frequent pair of adjacent tokens in C
            pair = self._get_most_frequent_pair()

            if pair is None:
                break
            t_l, t_r = pair
            t_new = t_l + t_r

            # Save merge rule order
            self.bpe_ranks[pair] = step

            # update the vocabulary
            self.vocabulary[t_new] = None

            # Replace each occurrence of t_L , t_R in C with t_NEW
            new_corpus = {}
            for word, freq in self.corpus.items():
                # fast pre-filter to bypass calling _merge_tuple on words that have no chance of containing the pair
                if t_l in word and t_r in word:
                    new_word = self.merge_tuple(word, t_l, t_r, t_new)
                    new_corpus[new_word] = new_corpus.get(new_word, 0) + freq
                else:
                    new_corpus[word] = freq
            self.corpus = new_corpus

    def encode(self, word_tuple: tuple[str]) -> List[str]:
        result = []
        current_tuple = word_tuple

        # Words of length <= 1 cannot be merged further
        if len(current_tuple) < 2:
            return list(current_tuple)

        while True:
            highest_rank = None
            highest_rank_pair = None

            # find the highest ranking pair (lowest numerical rank)
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                if pair in self.bpe_ranks:
                    rank = self.bpe_ranks[pair]
                    if highest_rank is None or rank < highest_rank:
                        highest_rank = rank
                        highest_rank_pair = pair

            # No further valid merges exist
            if highest_rank_pair is None:
                break

            # Construct the new tuple by merging all occurrences of highest_rank_pair
            first, second = highest_rank_pair
            new_tuple = []
            i = 0
            while i < len(current_tuple):
                if (
                    i < len(current_tuple) - 1
                    and current_tuple[i] == first
                    and current_tuple[i + 1] == second
                ):
                    new_tuple.append(first + second)
                    i += 2
                else:
                    new_tuple.append(current_tuple[i])
                    i += 1

            current_tuple = tuple(new_tuple)

            if len(current_tuple) == 1:
                break

        return result


    def tokenize(self, text: str) -> List[str]:
        # Split the text using the same regex applied during training
        pre_tokens = self._pre_tokenize(text)
        result = []

        for pre_token in pre_tokens:
            # Segment into characters
            word = tuple(pre_token)

            # Run BPE encoder
            encoded_tokens = self.encode(word)

            # Run the BPE merge rules
            result.append(encoded_tokens)

        return result

    def _pre_tokenize(self, text: str) -> List[str]:
        return re.findall(self.pattern, text)

    def _get_most_frequent_pair(self) -> Tuple[str, str]:
        counter: Counter[Tuple[str, str]] = Counter()

        for word in self.corpus:
            counter.update(pairwise(word))

        # Handle edge case where no pairs exist
        top_pair = counter.most_common(1)

        if not top_pair:
            raise ValueError("No adjacent pairs found in corpus.")

        for word in self.corpus:
            # for i in range(word.)
            for i in range(len(word) - 1):
                first = word[i]
                second = word[i + 1]
                counter[(first, second)] += 1

        return top_pair[0][0]



if __name__ == '__main__':
    tokenizer = BPETokenizer()

    # text = "abc abd abd abq bca"
    # tokenizer.prepare_corpus(text)
    # tokenizer.train_bpe(2)
    # tokenizer.tokenize("Alice was 7 years old, wasn't she? Alice was")


    alice_raw = gutenberg.raw('carroll-alice.txt')
    tokens = tokenizer.prepare_corpus(alice_raw)
    tokenizer.train_bpe(300)

    i = 8
