# encoding: utf-8
import argparse
import os
import codecs
import settings

class Util:
    @staticmethod
    def add_vocab(vocab, key):
        if not key in vocab:
            vocab[key] = 1
        else:
            vocab[key] += 1

class Vocab(object):
    def __init__(self):
        self.word_vocab = dict()
        self.word_vocab["<PAD>"] = 1
        self.word_vocab["<UNK>"] = 1
    def word_freq(self, in_dir):
        with codecs.open(os.path.join(in_dir, "train.txt"), encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
            for line in fp:
                words = line.lower().strip().split(" ||| ")[1].split()
                for w in words:
                    Util.add_vocab(self.word_vocab, w)

    def print_vocab(self, out_dir):
        sorted_vocab = sorted(self.word_vocab.items(), key=lambda x: x[1], reverse=True)
        idx = 2
        with codecs.open(os.path.join(out_dir, "vocab.txt"), encoding='utf-8', mode='w', buffering=settings.write_buffer_size) as re_fp:
            re_fp.write(u"{}\t{}\t{}\n".format("<PAD>", 0, 1))
            re_fp.write(u"{}\t{}\t{}\n".format("<UNK>", 1, 1))
            for x in sorted_vocab:
                if x[1] >= 0 and x[0] != "<UNK>" and x[0] != "<PAD>" and x[0]:
                    re_fp.write(u"{}\t{}\t{}\n".format(x[0], idx, x[1]))
                    idx += 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in-dir", "--in-dir", type=str, default=r"./data/")
    parser.add_argument("-out-dir", "--out-dir", type=str, default=r"./data/")
    args = parser.parse_args()

    v = Vocab()
    v.word_freq(args.in_dir)
    v.print_vocab(args.out_dir)