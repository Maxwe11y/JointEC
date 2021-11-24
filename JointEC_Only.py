'''
Author: Li Wei
Email: wei008@e.ntu.edu.sg
'''

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, \
    classification_report, precision_recall_fscore_support

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader_jointEC import IEMOCAPDataset
import torch.nn.functional as F
from tr import TextTransformer, PositionalEncoding, LearnedPositionEncoding
import argparse
import pickle
from prepare_data import *
from torch.nn import utils as nn_utils
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.nn.init import kaiming_uniform_, kaiming_normal_

np.random.seed(1234)
torch.random.manual_seed(1234)
torch.cuda.manual_seed(1234)


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(path, batch_size=1, valid=0.0, num_workers=0, word_idx=None, max_sen_len=None, pin_memory=False):
    trainset = IEMOCAPDataset(path=path, word_idx=word_idx, max_sen_len=max_sen_len)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=path, word_idx=word_idx, max_sen_len=max_sen_len, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch*seq_len
        """

        if type(self.weight) == type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target) \
                   / torch.sum(self.weight[target])

        # loss = self.loss(pred, target)
        return loss


class ECPEC(nn.Module):

    def __init__(self, input_dim, n_class, dropout):
        super(ECPEC, self).__init__()
        self.input_dim = input_dim
        self.n_class = n_class
        self.rep = nn.Linear(input_dim, input_dim)

        self.W = nn.Linear(2 * input_dim, input_dim)
        self.W2 = nn.Linear(6 * input_dim + 2, input_dim)
        self.Wo = nn.Linear(input_dim, self.n_class)

        self.gru = nn.GRU(input_size=input_dim, hidden_size=input_dim, num_layers=1, bidirectional=False)

        self.wgru = nn.LSTM(input_size=input_dim, hidden_size=int(input_dim / 2), num_layers=1, batch_first=True,
                            bidirectional=True)

        self.xatt = nn.Parameter(kaiming_uniform_(torch.zeros((3*input_dim+1, 3*input_dim+1))), requires_grad=True)
        self.xattck = nn.Parameter(kaiming_uniform_(torch.zeros((3*input_dim+1, 3 * input_dim + 1))), requires_grad=True)

        self.fusion2 = nn.Bilinear(2*self.input_dim, 2*self.input_dim, self.input_dim, bias=False)
        self.fusion1 = nn.Linear(2*self.input_dim, self.input_dim)

        self.ac = nn.Sigmoid()
        self.ac_linear = nn.ReLU()
        self.ac_tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.nocuda = not args.cuda

        self.transa = TextTransformer(LearnedPositionEncoding, d_model=4 * self.input_dim, d_out=self.input_dim,
                                      nhead=4, num_encoder_layers=3,
                                      dim_feedforward=512)
        self.W3 = nn.Linear(3 * input_dim + 1, input_dim)
        #self.W3 = nn.Linear(input_dim, input_dim)
        self.cls = nn.Linear(input_dim, self.n_class)
        self.scorer = nn.Linear(2*input_dim, 1)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, text_emo=None, text_cau=None, chunksize=8, t_ratio=1, label_ck=None,
                label3=None, share_rep=False, train=True, word_encode=None, sen_lengths=None, bi_label_emo=None, bi_label_cau=None):
        """
        :param text_emo:-->seq, batch, dim
                text_cau:-->seq, batch, dim
                label_ck:-->seq, num_chunk
        :return:
        """

        # word-level encoding
        word_encode = word_encode.float()
        seq_lengths = sen_lengths.squeeze().cpu()
        pack = nn_utils.rnn.pack_padded_sequence(word_encode, seq_lengths, batch_first=True, enforce_sorted=False)
        hw = torch.zeros((2, word_encode.size(0), int(word_encode.size(2) / 2))).cuda() if cuda else \
            torch.zeros((2, word_encode.size(0), int(word_encode.size(2) / 2)))
        cw = torch.zeros((2, word_encode.size(0), int(word_encode.size(2) / 2))).cuda() if cuda else \
            torch.zeros((2, word_encode.size(0), int(word_encode.size(2) / 2)))
        out, _ = self.wgru(pack, (hw, cw))
        unpacked = nn_utils.rnn.pad_packed_sequence(out, batch_first=True)
        index = torch.LongTensor([seq_lengths[i] + unpacked[0].size(1) * i - 1 for i in range(unpacked[0].size(0))])
        U = unpacked[0].contiguous().view(-1, unpacked[0].size(2))[index].unsqueeze(1)

        emo_mask = bi_label_emo.unsqueeze(1).bool()
        text_emo = torch.masked_select(U, emo_mask).contiguous().view(-1, 1, 100)

        cau_mask = bi_label_cau.unsqueeze(1).bool()
        text_cau = torch.masked_select(U, cau_mask).contiguous().view(-1, 1, 100)

        # the emotion cause chunk pair extraction task -- GRU variant

        label3 = label3.squeeze(1).cuda() if not self.nocuda else label3.squeeze()

        if share_rep:
            text_emo = self.ac(self.rep(text_emo))
            text_cau = self.ac(self.rep(text_cau))

        p_out = []

        # iterate text_emo and pair up all the emotion-chunk and emotion-cause pairs
        pair = torch.zeros((text_emo.size(0)*text_cau.size(0), 3*text_emo.size(-1)+1)).cuda() if cuda else \
            torch.zeros((text_emo.size(0) * text_cau.size(0), 3 * text_emo.size(-1) + 1))
        label_pair = torch.zeros((text_emo.size(0)*text_cau.size(0)), requires_grad=False)
        for j, emo in enumerate(text_emo):
            for i, cau in enumerate(text_cau):
                if i in label3[j]:
                    label_pair[j*text_cau.size(0)+i] = 1
                    #pair[j*text_cau.size(0)+i] = torch.cat((emo, cau), dim=-1).squeeze()
                pair[j * text_cau.size(0) + i] = ECPEC.item_att(emo, cau).squeeze()

        h = self.dropout(self.ac(self.W3(pair)))
        h = self.cls(h)  # .squeeze(1)
        p = torch.log_softmax(h, dim=1)
        p = p.contiguous().view(-1, 2)
        p_out.append(p)

        return p_out[0], label_pair

    @staticmethod
    def item_att(x, y):
        if y.size(0) != 1:
            item1 = torch.cat((x.repeat(y.size(0), 1), y), dim=-1)
        else:
            item1 = torch.cat((x,y), dim=-1)
        item2 = torch.norm(x - y, p=2, dim=-1, keepdim=True)
        item3 = torch.mul(x, y)
        delta = torch.cat((item1, item2, item3), dim=-1)

        return delta

    #TODO
    def ntn(self, x, y):
        #x = x.repeat(y.size(0), 1, 1)
        input = torch.cat((x.repeat(y.size(0), 1, 1), y), dim=2)
        #term2 = self.fusion2(input, input)
        term1 = self.fusion1(input)

        return term1



class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        pass

    def forward(self, *input):
        pass


def train_model(model, loss_function_c, dataloader, epoch, embedding=None, t_ratio=0.5, optimizer=None, train=True):
    losses = []
    # phase2 and phase3 tasks
    # maskec = []
    pred3 = []
    labelp3 = []

    assert not train or optimizer != None
    model.train()
    # count = 0
    total = 0
    postive = 0
    for data in dataloader:

        optimizer.zero_grad()

        text_emo, text_cau, label, label3, textid, sen_len, bi_label_emo, bi_label_cau = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        # total+=text_emo.size(0)*text_cau.size(0)
        # postive+=torch.sum(label3!=-1)
        word_encode = embedding(textid).squeeze()
        log_phase3, label_phase3 = model(text_emo=text_emo, text_cau=text_cau, t_ratio=t_ratio, label_ck=label,
                                                     label3=label3, word_encode=word_encode, sen_lengths=sen_len,
                                                     bi_label_emo=bi_label_emo, bi_label_cau=bi_label_cau)  # batch*seq_len, n_classes

        if args.cuda:
            label_phase3 = label_phase3.long().cuda() if len(label_phase3) != 0 else torch.tensor([])
        else:
            label_phase3 = label_phase3.long() if len(label_phase3) != 0 else torch.tensor([])

        # umask = torch.ones([text_emo.size(0)*text_cau.size(0), 1]).float()

        loss_3 = loss_function_c(log_phase3, label_phase3.view(-1))
        loss = loss_3

        # phase 3
        pred_3 = torch.argmax(log_phase3, 1)  # batch*seq_len
        pred3.append(pred_3.data.cpu().numpy())
        labelp3.append(label_phase3.view(-1).data.cpu().numpy())

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    pred3 = np.concatenate(pred3)
    labelp3 = np.concatenate(labelp3)


    avg_loss = round(np.average(losses), 4)

    #p_p2, r_p2, f_p2 = evaluate(predec, labelec)
    p_p3, r_p3, f_p3 = evaluate(pred3, labelp3)

    return avg_loss, p_p3, r_p3, f_p3


def eval_model(model, loss_function_c, dataloader, epoch, embedding=None, t_ratio=0.0, optimizer=None,
               train=False):
    losses = []
    # phase2 and phase3 tasks
    assert not train or optimizer != None
    model.eval()

    num_proposed_pairs = 0
    num_annotated_pairs = 0
    num_correct_pairs = 0
    pred3 = []
    labelp3 = []
    for data in dataloader:

        text_emo, text_cau, label, label3, bi_label_emo, bi_label_cau, \
        ids_cau, cLabels, phase3_label, textid, sen_len = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        if text_emo.size(0) == 0 or text_cau.size(0) == 0:
            return float('nan'), float('nan'), float('nan'), float('nan')
        else:
            word_encode = embedding(textid).squeeze()
            log_phase3, label_phase3 = model(text_emo=text_emo, text_cau=text_cau, t_ratio=t_ratio, label_ck=label,
                                                         label3=label3, train=False, word_encode=word_encode, sen_lengths=sen_len,
                                                         bi_label_emo=bi_label_emo, bi_label_cau=bi_label_cau)  # batch*seq_len, n_classes

            if args.cuda:
                label_phase3 = label_phase3.long().cuda() if len(label_phase3) != 0 else torch.tensor([])
            else:
                label_phase3 = label_phase3.long() if len(label_phase3) != 0 else torch.tensor([])

            # umask = torch.ones([text_emo.size(0)*text_cau.size(0), 1]).float()

            loss_3 = loss_function_c(log_phase3, label_phase3.view(-1))
            loss = loss_3
            losses.append(loss.item())

            pred_3 = torch.argmax(log_phase3, 1)  # batch*seq_len
            pred3.append(pred_3.data.cpu().numpy())
            labelp3.append(label_phase3.view(-1).data.cpu().numpy())

    pred3 = np.concatenate(pred3)
    labelp3 = np.concatenate(labelp3)
    p_p3, r_p3, f_p3 = evaluate(pred3, labelp3)

    avg_loss = round(np.average(losses), 4)

    return avg_loss, p_p3, r_p3, f_p3


def evaluate(pred, label):
    # calculate precision
    num_proposed_pairs = np.sum(pred)
    res = pred + label
    num_correct_pairs = np.sum(res == 2)
    precision = float(num_correct_pairs) / num_proposed_pairs

    # calculate recall
    # suitable for training phase 3 and phase 2
    num_annotated_pairs = np.sum(label)

    recall = float(num_correct_pairs) / num_annotated_pairs

    # calculate f1
    f1 = 2 * precision * recall / (precision + recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L1',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.30, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=30, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_false', default=True,
                        help='class weight')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    parser.add_argument('--max_sen_len', type=int, default=30,
                        help='max sentence length')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.cuda = True
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    batch_size = args.batch_size
    n_classes = 2
    cuda = args.cuda
    n_epochs = args.epochs
    dropout = args.dropout
    max_sen_len = args.max_sen_len

    D_m = 100

    model = ECPEC(D_m, n_classes, dropout)

    # word2vec loading
    w2v_path = './IEMOCAP_train_raw.txt'
    w2v_file = './glove.6B.100d.txt'
    word_idx_rev, word_id_mapping, word_embedding = load_w2v(D_m, w2v_path, w2v_file)
    word_embedding = torch.from_numpy(word_embedding)
    embedding = torch.nn.Embedding.from_pretrained(word_embedding, freeze=True).cuda() if cuda else \
        torch.nn.Embedding.from_pretrained(word_embedding, freeze=True)

    if cuda:
        model.cuda()

    loss_weights_c = torch.FloatTensor([
        1 / 0.798179,
        1 / 0.201821,
    ])  # emotion-cause pair task

    # loss_weights_c = torch.FloatTensor([
    #     0.527450,
    #     9.607568,
    # ])  # emotion-cause pair task

    if args.class_weight:
        loss_function_c = MaskedNLLLoss(loss_weights_c.cuda() if cuda else loss_weights_c)
    else:
        loss_function_c = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader, test_loader = \
        get_IEMOCAP_loaders(r'./ECPEC_phase_two_gcn_0.4_0.6_relu.pkl',
                            valid=0.0,
                            batch_size=batch_size,
                            num_workers=2,
                            word_idx=word_id_mapping,
                            max_sen_len=max_sen_len)

    best_loss = 100

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, p_p3, r_p3, f_p3 = train_model(model, loss_function_c,
                                                                     train_loader, e, embedding=embedding, optimizer=optimizer, train=True)
        test_loss, precision, recall, fscore = eval_model(model, loss_function_c,
                                                                       test_loader, e, embedding=embedding, t_ratio=0.0)

        if best_loss == 100 or best_loss > test_loss:
            best_loss, best_precision, best_recall, best_fscore = test_loss, precision, recall, fscore

        print(
            'epoch {} train_loss {} '
            'train_precision_p3 {} train_recall_p3 {} train_fscore_p3 {}'
            ' test_loss {} test_precision {} test_recall {} test_fscore {} time {}'. \
                format(e + 1, train_loss, p_p3, r_p3, f_p3, test_loss, precision, recall,
                       fscore, round(time.time() - start_time, 2)))

    print('Test performance..')
    print('Loss {} precision {} recall {} fscore{} '.format(best_loss, best_precision, best_recall, best_fscore))
