#-*- encoding: utf-8 -*-
class Util:
    @staticmethod
    def add_vocab(vocab, word, idx):
        if word in vocab:
            raise AssertionError("{} is duplicate in the vocab".format(word))
        vocab[word] = idx