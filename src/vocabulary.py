"""Vocabulary class for caption tokenization"""
from collections import Counter


class Vocabulary:
    """Vocabulary class for caption tokenization"""

    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        """Build vocabulary from list of sentences"""
        frequencies = Counter()
        idx = 4  # Start after special tokens

        for sentence in sentence_list:
            for word in sentence.lower().split():
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        """Convert text to list of indices"""
        tokenized_text = text.lower().split()
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
