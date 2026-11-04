import re
from collections import Counter, OrderedDict
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
        # self.vocabulary = OrderedDict()     # dictionary that remembers insertion order

    def pre_tokenize(self, text: str) -> List[str]:
        """Splits input text into pre-tokens based on assignment specifications:

        - Consecutive sequence of English letters: one pre-token
        - Consecutive sequence of digits: one pre-token
        - Punctuation or special non-whitespace characters: individual
        pre-tokens
        - Whitespace characters are excluded and act as delimiters
        """
        pattern = r"[a-zA-Z]+|[0-9]+|[^a-zA-Z0-9\s]"
        return re.findall(pattern, text)

    def most_frequent_pair_of_adjacent_tokens(self) -> Tuple[str, str]:


    def prepare_corpus(self, text):
        pre_tokens = self.pre_tokenize(text)
        token_counts = Counter(pre_tokens)
        self.corpus = Counter(tuple(token) for token in pre_tokens)

    def train_bpe(self, n_merges):
        # V <- all unique characters in C
        self.vocabulary = dict.fromkeys(chain.from_iterable(self.corpus))

        for i in range(n_merges):
            (t_l, t_r) = self.most_frequent_pair_of_adjacent_tokens()
        pass


if __name__ == '__main__':
    tokenizer = BPETokenizer()
    text = "Alice was 7 years old, wasn't she? Alice was"

    tokens = tokenizer.pre_tokenize(text)
    tokenizer.prepare_corpus(text)
    tokenizer.train_bpe(5)
