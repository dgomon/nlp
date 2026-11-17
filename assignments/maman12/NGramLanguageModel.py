import re
from collections import Counter
from itertools import chain, pairwise

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import gutenberg
from tqdm import tqdm

training_data_percentage = 0.9

class NGramLanguageModel:
    def __init__(self, n):
        self.n = n

    def prepare_data(self, text):
        self.tokenized = nltk.tokenize.word_tokenize(text.lower())
        training_text_length = int(len(self.tokenized) * training_data_percentage)
        self.train_text = self.tokenized[:training_text_length]
        self.test_text = self.tokenized[training_text_length:]
        self.vocabulary = set(self.tokenized)
        self.vocabulary.add("<UNK>")


if __name__ == '__main__':
    model = NGramLanguageModel(10)


    alice_raw = gutenberg.raw('carroll-alice.txt')
    model.prepare_data(alice_raw)

    pass