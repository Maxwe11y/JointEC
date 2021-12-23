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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader_jointEC_window import IEMOCAPDataset
import torch.nn.functional as F
import argparse
from prepare_data import *
from torch.nn import utils as nn_utils
import sys
import os

np.random.seed(1234)
torch.random.manual_seed(1234)
torch.cuda.manual_seed(1234)


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(path, batch_size=1, valid=0.0, num_workers=0, word_idx=None, max_sen_len=None, ck_size=None, tf=None, pin_memory=False):
    trainset = IEMOCAPDataset(path=path, word_idx=word_idx, max_sen_len=max_sen_len, ck_size=ck_size, tf=tf)
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

    testset = IEMOCAPDataset(path=path, word_idx=word_idx, max_sen_len=max_sen_len, ck_size=ck_size, tf=tf, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


class Logger(object):
    def __init__(self, path, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(os.path.join(path, filename), 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


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

class MaskedNLLLoss2(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss2, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='none')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.contiguous().view(-1, 1) # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            # loss = self.loss(pred*mask_, target)\
            #                 /torch.sum(self.weight[target]*mask_.squeeze())
            loss = torch.sum(self.loss(pred, target)*mask_)/torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class ECPEC(nn.Module):

    def __init__(self, input_dim, pos_dim, n_class, dropout, dropout2, dropout3, ck_size):
        super(ECPEC, self).__init__()
        self.input_dim = input_dim
        self.n_class = n_class
        self.ck_size = ck_size
        self.pos_dim = pos_dim
        self.rep = nn.Linear(input_dim, input_dim)

        self.W = nn.Linear(2 * input_dim, input_dim)
        self.W2 = nn.Linear(3*input_dim+1 + pos_dim, input_dim)
        self.Wo = nn.Linear(input_dim, self.n_class)

        self.wgru = nn.LSTM(input_size=input_dim, hidden_size=int(input_dim / 2), num_layers=1, batch_first=True,
                            bidirectional=True)

        self.s1 = nn.Linear(input_dim, input_dim)
        self.s2 = nn.Linear(input_dim, 1, bias=False)

        #self.ac = nn.Sigmoid()
        #self.ac = nn.ReLU()
        self.ac = nn.LeakyReLU()
        #self.ac_tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)
        self.dropout3 = nn.Dropout(dropout3)
        self.nocuda = not args.cuda
        self.W3 = nn.Linear(3*input_dim + 2 + pos_dim, input_dim) # TODO
        self.cls = nn.Linear(input_dim, self.n_class)
        self.scorer = nn.Linear(input_dim, 1)

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

    def forward(self, label_ck=None, label3=None, share_rep=False, train=True,
                word_encode=None, pos_encode=None, dis_encode=None, sen_lengths=None, bi_label_emo=None, bi_label_cau=None, mask_window=None, ck_mask=None):
        """
        :param text_emo:-->seq, batch, dim
                text_cau:-->seq, batch, dim
                label_ck:-->seq, num_chunk
        :return:
        """

        # word-level encoding
        word_encode = self.dropout3(word_encode.float())
        pos_encode = pos_encode.float()
        dis_encode = dis_encode.float()
        seq_lengths = sen_lengths.squeeze().cpu()
        pack = nn_utils.rnn.pack_padded_sequence(word_encode, seq_lengths, batch_first=True, enforce_sorted=False)
        hw = torch.zeros((2, word_encode.size(0), int(word_encode.size(2) / 2))).cuda() if cuda else \
            torch.zeros((2, word_encode.size(0), int(word_encode.size(2) / 2)))
        cw = torch.zeros((2, word_encode.size(0), int(word_encode.size(2) / 2))).cuda() if cuda else \
            torch.zeros((2, word_encode.size(0), int(word_encode.size(2) / 2)))
        out, _ = self.wgru(pack, (hw, cw))
        unpacked = nn_utils.rnn.pad_packed_sequence(out, batch_first=True)
        # index = torch.LongTensor([seq_lengths[i] + unpacked[0].size(1) * i - 1 for i in range(unpacked[0].size(0))])
        # U = unpacked[0].contiguous().view(-1, unpacked[0].size(2))[index].unsqueeze(1)
        # word attention version
        U = self.att_var(unpacked[0], length=seq_lengths).unsqueeze(1)

        emo_mask = bi_label_emo.unsqueeze(1).bool()
        text_emo = torch.masked_select(U, emo_mask).contiguous().view(-1, 1, word_encode.size(2))

        cau_mask = bi_label_cau.unsqueeze(1).bool()
        text_cau = torch.masked_select(U, cau_mask).contiguous().view(-1, 1, word_encode.size(2))

        label3 = label3.squeeze(1).cuda() if not self.nocuda else label3.squeeze()

        if share_rep:
            text_emo = self.ac(self.rep(text_emo))
            text_cau = self.ac(self.rep(text_cau))
        chunks = torch.split(text_cau, self.ck_size, 0)  # [num_chunk, chunksize,1, dim] --> tuple

        p_out = []
        label3_out = []
        phase2_out = []
        pp_out = []
        mask_window = mask_window.bool()
        # iterate text_emo and pair up all the emotion-chunk and emotion-cause pairs
        for j, emo in enumerate(text_emo):
            chunkEmb = torch.empty([len(chunks), text_emo.size(2)]).cuda() if not self.nocuda else torch.empty([len(chunks), text_emo.size(2)])
            num = 0
            for i, ch in enumerate(chunks):
                scores = F.softmax(torch.matmul(emo, ch.transpose(1, 2)).squeeze(2), dim=0)
                chunkEmb[num] = torch.matmul(scores.transpose(0, 1), ch.permute(1, 0, 2)).squeeze()
                num += 1

            # item attention
            delta_c = ECPEC.item_att(emo, chunkEmb)
            delta = torch.cat((delta_c, pos_encode[j]), dim=-1)

            # classify the emotion-chunk pairs
            hidden = self.dropout2(self.ac(self.W2(delta)))
            out = torch.log_softmax(self.Wo(hidden), dim=1)
            out_ = out.contiguous().view(-1, 2)
            out_feature = torch.argmax(out_, dim=-1)
            phase2_out.append(out_)

            # emotion-cause pair
            chNum = label_ck.size(-1)
            idx = torch.ones(chNum)

            id_chks = torch.nonzero(idx)

            # phase 3 if there is emotion-chunk pair found in phase 2, then phase3 start
            # prepare label
            L_cau = torch.zeros(text_cau.size(0))
            for label in label3[j]:
                if label.item() != -1:
                    L_cau[label.item()] = 1
            chk_L_cau = torch.split(L_cau, self.ck_size, 0)

            chunks_sel = []
            label3_sel = []
            sz_ck = []
            extra_feature = []

            for id in id_chks:
                chunks_sel.append(chunks[id])
                label3_sel.append(chk_L_cau[id])
                sz_ck.append(chunks[id].size(0))
                extra_feature.append(out_feature[id].repeat(chunks[id].size(0), 1))
            extra_feature = torch.cat(extra_feature, dim=0)
            utt_c = torch.cat(chunks_sel, dim=0)


            delta_d = ECPEC.item_att(emo, utt_c.squeeze(1))
            extra_feature = extra_feature.repeat(1, 1)
            delta_ = torch.cat((delta_d, dis_encode[j], extra_feature), dim=-1)

            delta_ = torch.masked_select(delta_, mask_window.bool()[j].transpose(0, 1)).contiguous().view(-1, delta_.size(-1))


            h = self.dropout(self.ac(self.W3(delta_)))
            h_ = self.cls(h)  # .squeeze(1)
            p = torch.log_softmax(h_, dim=1)
            p = p.contiguous().view(-1, 2)
            p_out.append(p)

            label3_sel = torch.cat(label3_sel, dim=0)
            label3_sel = torch.masked_select(label3_sel, mask_window.bool()[j].squeeze(0)).contiguous().view(p.size(0))

            label3_out.extend(label3_sel)
            if not train:
                phase3_out = []
                phase3_out.append(p)
                pp_out.extend(phase3_out)

        if train:
            return phase2_out, p_out, label3_out
        else:
            return phase2_out, pp_out, label3_out

    @staticmethod
    def item_att(x, y):
        item1 = torch.cat((x.repeat(y.size(0), 1), y), dim=-1)
        item2 = torch.norm(x - y, p=2, dim=-1, keepdim=True)
        item3 = torch.mul(x, y)
        delta = torch.cat((item1, item2, item3), dim=-1)
        return delta

    def sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        mask = ~(torch.ones((len(lengths), maxlen), device=lengths.device).cumsum(dim=1).t() > lengths).t()
        mask.type(dtype)
        return mask

    def getmask(self, length, max_seq_len, out_shape):
        '''
        :param length: batch
        '''
        res = self.sequence_mask(length)
        return res.contiguous().view(out_shape)

    def softmax_by_length(self, inputs, length):
        '''
        :param inputs: batch, 1, max_seq_len
        :param length: batch
        :return: batch, dim
        '''
        inputs_ = torch.exp(inputs)
        inputs *= self.getmask(length, inputs_.size(2), inputs_.size())
        _sum = torch.sum(inputs, dim=2, keepdim=True) + 1e-9

        return inputs/_sum

    def att_var(self, inputs, length=None):
        '''

        :param inputs: shape->seq, batch, max_seq_len, dim
        :param length: shape->batch
        :return: shape->batch, dim
        '''
        if self.cuda:
            length = length.cuda()
        max_seq_len, dim = inputs.size(1), inputs.size(2)
        tmp = inputs.contiguous().view(-1, dim)
        utt = self.ac(self.s1(tmp))
        alpha = self.s2(utt).contiguous().view(-1, 1, max_seq_len)
        alpha_ = self.softmax_by_length(alpha, length)

        return torch.matmul(alpha_, inputs).contiguous().view(-1, dim)



class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        pass

    def forward(self, *input):
        pass


def train_model(model, loss_function, loss_function_c, dataloader, epoch, embedding=None, pos_embedding=None, optimizer=None, train=True):
    losses = []
    # phase2 and phase3 tasks
    predec = []
    labelec = []
    # maskec = []
    pred3 = []
    labelp3 = []

    assert not train or optimizer != None
    model.train()
    # count = 0
    # total = 0
    # postive = 0
    for data in dataloader:

        optimizer.zero_grad()

        label, ck_pos_id, pos_id, label3, textid, sen_len, bi_label_emo, bi_label_cau, mask_window, ck_mask = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        # total+=text_emo.size(0)*text_cau.size(0)
        # postive+=torch.sum(label3!=-1)
        word_encode = embedding(textid).squeeze()
        pos_encode = pos_embedding(ck_pos_id).squeeze(1)
        dis_encode = pos_embedding(pos_id).squeeze(1)
        log_phase2, log_phase3, label_phase3 = model(label_ck=label,
                                                     label3=label3, word_encode=word_encode, pos_encode=pos_encode, dis_encode=dis_encode, sen_lengths=sen_len,
                                                     bi_label_emo=bi_label_emo, bi_label_cau=bi_label_cau, mask_window=mask_window, ck_mask=ck_mask)  # batch*seq_len, n_classes

        log_phase2 = torch.cat(log_phase2, dim=0)
        log_phase3 = torch.cat(log_phase3, dim=0) if len(log_phase3) != 0 else []
        if args.cuda:
            label_phase3 = torch.stack(label_phase3).long().cuda() if len(label_phase3) != 0 else torch.tensor([])
        else:
            label_phase3 = torch.stack(label_phase3).long() if len(label_phase3) != 0 else torch.tensor([])

        label_ = label.view(-1)
        # umask = torch.ones([text_emo.size(0)*text_cau.size(0), 1]).float()

        #skip = False if len(log_phase3) != 0 else True
        # if label_.size(0) != ck_mask.view(-1, 1).size(0):
        #     print('error')
        loss_2 = loss_function(log_phase2, label_)
        loss_3 = loss_function_c(log_phase3, label_phase3.view(-1))
        loss = loss_2 + loss_3

        # phase 2
        pred_e = torch.argmax(log_phase2, 1)  # batch*seq_len
        predec.append(pred_e.data.cpu().numpy())
        labelec.append(label_.data.cpu().numpy())
        # maskec.append(umask.view(-1).cpu().numpy())

        # phase 3
        #if not skip:
        pred_3 = torch.argmax(log_phase3, 1)  # batch*seq_len
        pred3.append(pred_3.data.cpu().numpy())
        labelp3.append(label_phase3.view(-1).data.cpu().numpy())
        # else:
        #     pred3.append(torch.zeros_like(label_phase3).numpy())
        #     labelp3.append(label_phase3.view(-1).data.cpu().numpy())
        #     pass

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    if predec != []:
        predec = np.concatenate(predec)
        labelec = np.concatenate(labelec)
        # maskec = np.concatenate(maskec)
        # print(Counter(labels.tolist()))
        #if not skip:
        pred3 = np.concatenate(pred3)
        labelp3 = np.concatenate(labelp3)

    else:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    avg_loss = round(np.average(losses), 4)

    p_p2, r_p2, f_p2 = evaluate(predec, labelec)
    p_p3, r_p3, f_p3 = evaluate(pred3, labelp3)

    # print('total', total)
    # print('postive', postive)

    return avg_loss, p_p2, r_p2, f_p2, p_p3, r_p3, f_p3


def eval_model(model, loss_function, loss_function_c, dataloader, epoch, embedding=None, pos_embedding=None, ck_size=None, optimizer=None,
               train=False):
    losses = []
    losses3 = []

    assert not train or optimizer != None
    model.eval()

    predec = []
    labelec = []
    # maskec = []
    pred3 = []
    labelp3 = []
    for data in dataloader:

        label, ck_pos_id, pos_id, label3, bi_label_emo, bi_label_cau, \
        ids_cau, cLabels, phase3_label, textid, sen_len, mask_window, ck_mask = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        word_encode = embedding(textid).squeeze()
        pos_encode = pos_embedding(ck_pos_id).squeeze(1)
        dis_encode = pos_embedding(pos_id).squeeze(1)
        log_phase2, log_phase3, label_phase3 = model(label_ck=label,
                                                     label3=label3, train=False, word_encode=word_encode, pos_encode=pos_encode, dis_encode=dis_encode, sen_lengths=sen_len,
                                                     bi_label_emo=bi_label_emo, bi_label_cau=bi_label_cau, mask_window=mask_window, ck_mask=ck_mask)  # batch*seq_len, n_classes
        log_phase2 = torch.cat(log_phase2, dim=0)
        log_phase3 = torch.cat(log_phase3, dim=0) if len(log_phase3) != 0 else []
        if args.cuda:
            label_phase3 = torch.stack(label_phase3).long().cuda() if len(label_phase3) != 0 else torch.tensor([])
        else:
            label_phase3 = torch.stack(label_phase3).long() if len(label_phase3) != 0 else torch.tensor([])

        label_ = label.view(-1)
        # umask = torch.ones([text_emo.size(0)*text_cau.size(0), 1]).float()

        #skip = False if len(log_phase3) != 0 else True
        loss_2 = loss_function(log_phase2, label_)
        loss_3 = loss_function_c(log_phase3, label_phase3.view(-1))
        loss = loss_2 + loss_3

        # phase 2
        pred_e = torch.argmax(log_phase2, 1)  # batch*seq_len
        predec.append(pred_e.data.cpu().numpy())
        labelec.append(label_.data.cpu().numpy())
        # maskec.append(umask.view(-1).cpu().numpy())

        # phase 3
        #if not skip:
        pred_3 = torch.argmax(log_phase3, 1)  # batch*seq_len
        pred3.append(pred_3.data.cpu().numpy())
        labelp3.append(label_phase3.view(-1).data.cpu().numpy())
        # else:
        #     pred3.append(torch.zeros_like(label_phase3).numpy())
        #     labelp3.append(label_phase3.view(-1).data.cpu().numpy())
        #     pass

        losses.append(loss.item())
        losses3.append(loss_3.item())

    if predec != []:
        predec = np.concatenate(predec)
        labelec = np.concatenate(labelec)
        # maskec = np.concatenate(maskec)
        # print(Counter(labels.tolist()))
        #if not skip:
        pred3 = np.concatenate(pred3)
        labelp3 = np.concatenate(labelp3)

    else:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    avg_loss = round(np.average(losses), 4)
    avg_loss_3 = round(np.average(losses3), 4)

    #p_p2, r_p2, f_p2 = evaluate(predec, labelec)
    p_p3, r_p3, f_p3 = evaluate(pred3, labelp3)

    # print('total', total)
    # print('postive', postive)

    return avg_loss, avg_loss_3, p_p3, r_p3, f_p3


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
    parser.add_argument('--dropout2', type=float, default=0.3,
                        metavar='dropout2', help='ec chunk dropout rate')
    parser.add_argument('--dropout3', type=float, default=0.5,
                        metavar='dropout3', help='word_encode dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=30, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_false', default=True,
                        help='class weight')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    parser.add_argument('--max_sen_len', type=int, default=20,
                        help='max sentence length')
    parser.add_argument('--chunk_size', type=int, default=2,
                        help='size of the cause chunk')
    parser.add_argument('--tf', type=float, default=0.5,
                        help='teacher forcing rate for training')
    parser.add_argument('--embed_dim', type=int, default=100,
                        help='dimension of pre-trained embeddings')
    parser.add_argument('--pos_dim', type=int, default=100,
                        help='dimension of position embeddings')
    args = parser.parse_args()

    # path = r'/home/maxwe11y/Desktop/weili/phase3/JointEC_gcn_res_sigmoid'
    # if args.tf == 0:
    #     filename = 'key_s20_' + 'true_' + 'ck' + str(args.chunk_size) + str(args.lr) + '_1.txt'
    # elif args.tf == 1:
    #     filename = 'key_s20_' + 'pred_' + 'ck' + str(args.chunk_size) + str(args.lr) + '_1.txt'
    # else:
    #     filename = 'key_s20_' + 'tf_' + 'ck' + str(args.chunk_size) + str(args.lr) + '_1.txt'
    # sys.stdout = Logger(path=path, filename=filename)

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
    ck_size = args.chunk_size
    dropout2 = args.dropout2
    dropout3 = args.dropout3
    tf = args.tf

    D_m = args.embed_dim
    pos_D_m = args.pos_dim

    model = ECPEC(D_m, pos_D_m, n_classes, dropout, dropout2, dropout3, ck_size)

    # word2vec loading
    w2v_path = './key_words.txt'
    w2v_file = './glove.6B.100d.txt'
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(D_m, pos_D_m, w2v_path, w2v_file)
    word_embedding = torch.from_numpy(word_embedding)
    pos_embedding = torch.from_numpy(pos_embedding)
    embedding = torch.nn.Embedding.from_pretrained(word_embedding, freeze=True).cuda() if cuda else \
        torch.nn.Embedding.from_pretrained(word_embedding, freeze=True)
    pos_embedding = torch.nn.Embedding.from_pretrained(pos_embedding, freeze=True).cuda() if cuda else \
        torch.nn.Embedding.from_pretrained(pos_embedding, freeze=True)

    if cuda:
        model.cuda()

    loss_weights = torch.FloatTensor([
        1 / 0.757123,
        1 / 0.242877,
    ])  # emotion-chunk pair task

    loss_weights_c = torch.FloatTensor([
        1 / 0.798179,
        1 / 0.201821,
    ])  # emotion-cause pair task

    if args.class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        loss_function_c = MaskedNLLLoss(loss_weights_c.cuda() if cuda else loss_weights_c)
    else:
        loss_function = MaskedNLLLoss()
        loss_function_c = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader, test_loader = \
        get_IEMOCAP_loaders(r'../ECPEC_phase_two_gcn_0.4_0.7_relu_full.pkl',
                            valid=0.0,
                            batch_size=batch_size,
                            num_workers=2,
                            word_idx=word_id_mapping,
                            max_sen_len=max_sen_len,
                            ck_size=ck_size,
                            tf=tf)

    best_loss = 100

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, p_p2, r_p2, f_p2, p_p3, r_p3, f_p3 = train_model(model, loss_function, loss_function_c,
                                                                     train_loader, e, embedding=embedding, pos_embedding=pos_embedding, optimizer=optimizer, train=True)
        # valid_loss, valid_p_p2, valid_r_p2, valid_f_p2, valid_p_p3, valid_r_p3, valid_f_p3 = train_model(model, loss_function, loss_function_c, valid_loader, e)
        test_loss, test_loss_3, precision, recall, fscore = eval_model(model, loss_function, loss_function_c,
                                                                       test_loader, e, embedding=embedding, pos_embedding=pos_embedding, ck_size=ck_size)

        if best_loss == 100 or best_loss > test_loss_3:
            best_loss, best_precision, best_recall, best_fscore = test_loss_3, precision, recall, fscore

        print(
            'epoch {} train_loss {} train_precision_p2 {} train_recall_p2 {} train_fscore_p2 {} '
            'train_precision_p3 {} train_recall_p3 {} train_fscore_p3 {}'
            ' test_loss {} test_loss_3 {} test_precision {} test_recall {} test_fscore {} time {}'. \
                format(e + 1, train_loss, p_p2, r_p2, f_p2, p_p3, r_p3, f_p3, test_loss, test_loss_3, precision, recall,
                       fscore, round(time.time() - start_time, 2)))

    print('Test performance..')
    print('Loss {} precision {} recall {} fscore{} '.format(best_loss, best_precision, best_recall, best_fscore))
