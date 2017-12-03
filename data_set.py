#-*- encoding:utf-8 -*-
from vocab import Vocab
import os
import settings
import codecs
import numpy as np

class Dataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.vocab = Vocab(settings.vocab_size, os.path.join(self.data_dir, "vocab.txt"))
        self.sentences = []
        self.sen_id = []
        self.labels = []
        self.size = 0
        self.start_idx = 0
        self.load_train_dataset()

        self.dev_data = self.load_test_dataset("dev.txt")
        self.test_data = self.load_test_dataset("test.txt")
    def get_wordembedding(self):
        with codecs.open(os.path.join(self.data_dir, "glove.6B.100d.txt"), encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
            print "start loading word embedding..."
            parameter_word_embedding = np.random.normal(settings.mu, settings.sigma, (settings.vocab_size, settings.word_embedding_size)).astype(np.float32)
            for line in fp:
                words = line.strip().split()
                w = words[0]
                idx = self.vocab.get_idx(w)
                if idx != settings.UNK:
                    vec = np.array([float(x) for x in words[1:]], dtype=np.float32)
                    parameter_word_embedding[idx, :] = vec
        return parameter_word_embedding

    def load_train_dataset(self):
        with codecs.open(os.path.join(self.data_dir, "train.txt"), encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
            for line in fp:
                label, text = line.lower().strip().split(" ||| ")
                self.labels.append(int(label))
                words = text.split()
                self.sentences.append(words)
                self.sen_id.append([self.vocab.get_idx(w) for w in words])
                self.size += 1
        self.max_idx = (self.size / settings.batch_size) * settings.batch_size
        sen_labels_id = zip(self.sentences, self.labels, self.sen_id)
        sorted_sen_labels_id = sorted(sen_labels_id, key=lambda x : len(x[0]))
        self.sentences, self.labels, self.sen_id = zip(*sorted_sen_labels_id)
    def load_test_dataset(self, fn):
        labels = []
        sen_id_pad = []
        sen_mask = []
        with codecs.open(os.path.join(self.data_dir, fn), encoding='utf-8', mode='r', buffering=settings.read_buffer_size) as fp:
            for line in fp:
                label, text = line.lower().strip().split(" ||| ")
                labels.append(int(label))
                words = text.split()
                max_len = min(settings.max_len, len(words))
                sen_id_pad.append(np.array([[self.vocab.get_idx(w) for w in words][0:max_len]]))
                tmp_sen_mask = np.zeros((1, max_len, settings.hidden_size), dtype=np.int64)
                tmp_sen_mask[0, max_len - 1, :] = np.ones(settings.hidden_size)
                sen_mask.append(tmp_sen_mask.tolist())
        return sen_id_pad, sen_mask, labels

    def _batch_padding(self, batch_id):
        """
        
        :param batch_id: [[...], [...], ...] list of list of id 
        :return: 
            batch_id_pad: numpy array (batch_size, max_len)
            batch_mask: numpy array (batch_size, max_len, hidden_size)
        """
        max_len = min(len(batch_id[-1]), settings.max_len)
        batch_id_pad = np.zeros((settings.batch_size, max_len), dtype=np.int64)
        batch_mask = np.zeros((settings.batch_size, max_len, settings.hidden_size), dtype=np.int64)
        i = 0
        for s_id in batch_id:
            l = len(s_id)
            if l > max_len:
                batch_id_pad[i, :] = np.array(s_id[0:max_len])
                batch_mask[i, max_len-1, :] = np.ones(settings.hidden_size)
            else:
                batch_id_pad[i, :] = np.array(s_id + [settings.pad_idx]*(max_len-l))
                batch_mask[i, l-1, :] = np.ones(settings.hidden_size)
            i += 1
        return np.array(batch_id_pad), np.array(batch_mask)
    def get_batch(self):
        batch_id = self.sen_id[self.start_idx:self.start_idx+settings.batch_size]
        batch_labels = self.labels[self.start_idx:self.start_idx+settings.batch_size]
        batch_id_pad, batch_mask = self._batch_padding(batch_id)
        if self.start_idx + settings.batch_size >= self.max_idx:
            self.start_idx = 0
        else:
            self.start_idx += settings.batch_size
        return batch_id_pad, batch_mask.tolist(), batch_labels