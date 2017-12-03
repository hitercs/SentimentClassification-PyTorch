#-*- encoding: utf-8 -*-
import codecs
import settings
from util import Util

class Vocab(object):
    def __init__(self, vocab_size, vocab_fn):
        self.word2idx = dict()
        self.idx2word = dict()
        self.vocab_size = vocab_size
        self.build_vocab(vocab_fn)
    def build_vocab(self, vocab_fn):
        with codecs.open(vocab_fn, encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
            for line in fp:
                word, idx, _ = line.strip().split()
                Util.add_vocab(self.word2idx, word, int(idx))
                Util.add_vocab(self.idx2word, int(idx), word)
    def get_idx(self, word):
        if not word in self.word2idx:
            return settings.UNK
        if self.word2idx[word] > self.vocab_size - 1:
            return settings.UNK
        return self.word2idx[word]
    def get_word(self, idx):
        if idx > self.vocab_size - 1:
            return "<UNK>"
        return self.idx2word[idx]