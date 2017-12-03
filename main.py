#-*- encoding: utf-8 -*-
import settings
from model import LSTM
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data_set import Dataset
import numpy as np
import os

def main():
    model = LSTM(settings.vocab_size, settings.word_embedding_size,
                 settings.hidden_size, settings.num_layers, settings.out_dim, settings.drop_out)
    '''     pre-train word embedding init    '''
    dataset = Dataset(args.data)
    model.word_embed.weight = nn.Parameter(torch.from_numpy(dataset.get_wordembedding()))
    optimizer = optim.SGD(model.parameters(), lr=settings.lr, weight_decay=1e-5)
    criteria = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(settings.seed)
        model.cuda()
        criteria.cuda()
    best_dev_acc = 0.0
    best_test_acc = 0.0

    for i in xrange(dataset.size/settings.batch_size*settings.max_epochs):
        batch_data = dataset.get_batch()
        loss = train(model, batch_data, optimizer, criteria)
        if (i+1) % settings.validate_freq == 0:
            print "validating..."
            dev_acc = test(model, dataset.dev_data)
            test_acc = test(model, dataset.test_data)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_test_acc = test_acc
                torch.save(model, os.path.join(args.model_dir, "sa_{}.model".format(best_dev_acc)))
            with open(os.path.join(args.model_dir, "log.txt"), "a") as logger:
                logger.write("epoch: {:.2f}, dev acc: {:.2f}, test acc: {:.2f}, " \
                  "batch loss: {:.2f}, best dev acc:{:.2f}, best test acc:{:.2f}\n".format(i*settings.batch_size/float(dataset.size),
                   dev_acc, test_acc, loss.cpu().numpy()[0], best_dev_acc, best_test_acc))
            print "epoch: {:.2f}, dev acc: {:.2f}, test acc: {:.2f}, " \
                  "batch loss: {:.2f}, best dev acc:{:.2f}, best test acc:{:.2f}".format(i*settings.batch_size/float(dataset.size),
                   dev_acc, test_acc, loss.cpu().numpy()[0], best_dev_acc, best_test_acc)

def var_to_cuda(batch_data):
    """
    Transform the input batch to PyTorch variables
    """
    batch_sen_id = Variable(torch.from_numpy(batch_data[0]))
    batch_sen_mask = Variable(torch.ByteTensor(batch_data[1]))
    batch_labels = Variable(torch.LongTensor(batch_data[2]))
    if args.cuda:
        batch_sen_id = batch_sen_id.cuda()
        batch_sen_mask = batch_sen_mask.cuda()
        batch_labels = batch_labels.cuda()

    return batch_sen_id, batch_sen_mask, batch_labels

def train(model, batch_data, optimizer, criteria):
    batch_sen_id, batch_sen_mask, batch_labels = var_to_cuda(batch_data)
    model.zero_grad()
    class_prob = model(batch_sen_id, batch_sen_mask)
    loss = criteria(class_prob, batch_labels)
    loss.backward()
    optimizer.step()
    return loss.data

def test(model, test_data):
    true_predict = 0
    total = len(test_data[0])
    for i in range(total):
        sen_id, sen_mask, sen_label = var_to_cuda((test_data[0][i], test_data[1][i], test_data[2][i]))
        class_prob = model(sen_id, sen_mask)
        prob = class_prob.cpu().data.numpy()
        if np.argmax(prob) == test_data[2][i]:
            true_predict += 1
    acc = true_predict/float(total)
    return acc


if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", "--data", type=str, default=r"./data/")
    parser.add_argument("-model", "--model-dir", type=str, default=r"./model/")
    parser.add_argument("-cuda", "--cuda", type=bool, default=False)
    args = parser.parse_args()
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    main()
