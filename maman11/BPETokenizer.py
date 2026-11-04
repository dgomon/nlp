import re
from collections import Counter
from itertools import chain
from typing import List, Tuple


class BPETokenizer:
    """Byte Pair Encoding (BPE) Tokenizer implementation.

    Maintains internal vocabulary and merge rules learned during training.
    """

    def __init__(self):
        self.vocabulary = None
        self.pre_tokens_counter = None
        self.corpus = None

    def pre_tokenize(self, text: str) -> List[str]:
        pattern = r"[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9\s]"
        return re.findall(pattern, text)

    def most_frequent_pair_of_adjacent_tokens(self) -> Tuple[str, str]:
        counter = Counter()
        for word in self.corpus:
            # for i in range(word.)
            for i in range(len(word) - 1):
                first = word[i]
                second = word[i + 1]
                counter[(first, second)] += 1
        return counter.most_common(1)[0][0]

    def prepare_corpus(self, text):
        pre_tokens = self.pre_tokenize(text)
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

        for _ in range(n_merges):
            # t_L, t_R <- Most frequent pair of adjacent tokens in C
            pair = self.most_frequent_pair_of_adjacent_tokens()

            if pair is None:
                break
            t_l, t_r = pair

            t_new = t_l + t_r

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


if __name__ == '__main__':
    tokenizer = BPETokenizer()
    text = "Alice was 7 years old, wasn't she? Alice was"

    tokens = tokenizer.pre_tokenize(text)
    tokenizer.prepare_corpus(text)
    tokenizer.train_bpe(5)
