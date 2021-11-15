#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from torch.nn import utils as nn_utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader_ec import IEMOCAPDataset
import torch.nn.functional as F
#from tr import TextTransformer, PositionalEncoding, LearnedPositionEncoding
import argparse
import pickle
from prepare_data import *

np.random.seed(1234)
torch.random.manual_seed(1234)
torch.cuda.manual_seed(1234)


def init_lstm(input_lstm):
    """
    Initialize lstm

    PyTorch weights parameters:

        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size * hidden_size)`

        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size * hidden_size)`
    """

    # Weights init for forward layer
    for ind in range(0, input_lstm.num_layers):

        ## Gets the weights Tensor from our model, for the input-hidden weights in our current layer
        weight = eval('input_lstm.weight_ih_l' + str(ind))

        # Initialize the sampling range
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))

        # Randomly sample from our samping range using uniform distribution and apply it to our current layer
        #nn.init.uniform(weight, -sampling_range, sampling_range)
        nn.init.orthogonal_(weight)

        # Similar to above but for the hidden-hidden weights of the current layer
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        #sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        #nn.init.uniform(weight, -sampling_range, sampling_range)
        nn.init.orthogonal_(weight)


    # We do the above again, for the backward layer if we are using a bi-directional LSTM (our final model uses this)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -sampling_range, sampling_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -sampling_range, sampling_range)

    # Bias initialization steps

    # We initialize them to zero except for the forget gate bias, which is initialized to 1
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))

            # Initializing to zero
            bias.data.zero_()

            # This is the range of indices for our forget gates for each LSTM cell
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

            #Similar for the hidden-hidden layer
            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        # Similar to above, we do for backward layer if we are using a bi-directional LSTM
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(path, batch_size=1, valid=0.0, num_workers=0, word_idx=None, max_sen_len=30, pin_memory=False):
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

        #loss = self.loss(pred, target)
        return loss


class ECPEC(nn.Module):

    def __init__(self, input_dim, n_class, dropout):
        super(ECPEC, self).__init__()
        self.input_dim = input_dim
        self.n_class = n_class
        self.rep = nn.Linear(input_dim, input_dim)
        #self.ln = nn.LayerNorm([30, input_dim], elementwise_affine=True)

        self.W = nn.Linear(2 * input_dim, input_dim)
        self.W2 = nn.Linear(3*input_dim+1, input_dim)
        self.Wo = nn.Linear(input_dim, self.n_class)

        self.gru = nn.GRU(input_size=input_dim, hidden_size=input_dim, num_layers=1, bidirectional=False)
                        
        #self.wgru = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True, dropout=0.3, bidirectional=False)

        self.wgru_emo = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True, dropout=0.3,
                            bidirectional=False)
        init_lstm(self.wgru_emo)
        self.wgru_cau = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True, dropout=0.3,
                            bidirectional=False)
        init_lstm(self.wgru_cau)

        self.ac = nn.Sigmoid()
        self.ac_linear = nn.ReLU()
        self.ac_tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.nocuda = False

        self.W3 = nn.Linear(4*input_dim+1, input_dim)
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

    def forward(self, ids_emo, ids_cau, x, sen_lengths, chunksize=8, t_ratio=1, label_ck=None,
                label3=None, share_rep=True, train=True):
        """
        :param text_emo:-->seq, batch, dim
                text_cau:-->seq, batch, dim
                label_ck:-->seq, num_chunk
        :return:
        """
        # word-level encoding
        x = x.float()
        seq_lengths = sen_lengths.squeeze()
        word_emo = x[ids_emo.squeeze(1)].squeeze(1)
        word_cau = x[ids_cau.squeeze(1)].squeeze(1)

        seq_emo = seq_lengths[ids_emo.squeeze(1)].squeeze(1)
        seq_cau = seq_lengths[ids_cau.squeeze(1)].squeeze(1)

        # generate sen-level embedding for emotion utterances
        pack_emo = nn_utils.rnn.pack_padded_sequence(word_emo, seq_emo, batch_first=True, enforce_sorted=False)
        #hw = torch.zeros((1, word_emo.size(0), word_emo.size(2))).cuda()
        #cw = torch.zeros((1, word_emo.size(0), word_emo.size(2))).cuda()
        #out_emo, _ = self.wgru_emo(pack_emo, (hw, cw))
        out_emo, _ = self.wgru_emo(pack_emo)
        unpacked_emo = nn_utils.rnn.pad_packed_sequence(out_emo, batch_first=True)
        index_emo = torch.LongTensor([seq_emo[i] + unpacked_emo[0].size(1) * i - 1 for i in range(unpacked_emo[0].size(0))])
        text_emo = unpacked_emo[0].contiguous().view(-1, unpacked_emo[0].size(2))[index_emo].unsqueeze(1)

        # generate sen-level embedding for cause utterances
        pack_cau = nn_utils.rnn.pack_padded_sequence(word_cau, seq_cau, batch_first=True, enforce_sorted=False)
        #hc = torch.zeros((1, word_cau.size(0), word_cau.size(2))).cuda()
        #cc = torch.zeros((1, word_cau.size(0), word_cau.size(2))).cuda()
        #out_cau, _ = self.wgru_cau(pack_cau, (hc, cc))
        out_cau, _ = self.wgru_cau(pack_cau)
        unpacked_cau = nn_utils.rnn.pad_packed_sequence(out_cau, batch_first=True)
        index_cau = torch.LongTensor([seq_cau[i] + unpacked_cau[0].size(1) * i - 1 for i in range(unpacked_cau[0].size(0))])
        text_cau = unpacked_cau[0].contiguous().view(-1, unpacked_cau[0].size(2))[index_cau].unsqueeze(1)


        '''pack = nn_utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        hw = torch.zeros((1, x.size(0), x.size(2))).cuda()
        cw = torch.zeros((1, x.size(0), x.size(2))).cuda()
        out, _ = self.wgru(pack,(hw, cw))
        unpacked = nn_utils.rnn.pad_packed_sequence(out, batch_first=True)
        index = torch.LongTensor([seq_lengths[i]+unpacked[0].size(1)*i-1 for i in range(unpacked[0].size(0))])
        U = unpacked[0].contiguous().view(-1, unpacked[0].size(2))[index]
        text_emo = U[ids_emo.squeeze(1)]
        text_cau = U[ids_cau.squeeze(1)]'''

        # the emotion cause chunk pair extraction task -- GRU variant

        label3 = label3.squeeze(1).cuda()

        if share_rep:
            text_emo = self.ac(self.rep(text_emo))
            text_cau = self.ac(self.rep(text_cau))
        chunks = torch.split(text_cau, chunksize, 0)  # [num_chunk, chunksize,1, dim] --> tuple

        p_out = []
        label3_out = []
        phase2_out = []
        pp_out = []

        # iterate text_emo and pair up all the emotion-chunk and emotion-cause pairs 

        for j, emo in enumerate(text_emo):
            # initialize the emotion-chunk embedding for each text_emo
            chunkEmb = torch.empty([len(chunks), 2*text_emo.size(2)]).cuda()
            chunkEmbedding = torch.empty([len(chunks), text_emo.size(2)]).cuda()
            num = 0

            #for i, ch in enumerate(chunks):
            #    h0 = torch.zeros((1, text_emo.size(1), text_emo.size(2))).cuda()
            #    chEmb, _ = self.gru(ch, h0)
                #chunkEmb[num] = chEmb[-1]
            #    chunkEmb[num] = torch.cat((emo, chEmb[-1]), dim=1)
            #    chunkEmbedding[num] = chEmb[-1]
            #    num +=1


            # employ self attention to generate chunk embedding
            #for i, ch in enumerate(chunks):
            #    ch = self.dropout(ch)
            #    seq_len, batch_size, feature_dim = ch.size()
            #    scores = self.scorer(ch.contiguous().view(-1, feature_dim)).view(batch_size, seq_len)
            #    scores = F.softmax(scores, dim=1).transpose(0, 1)
            #    chEmb = scores.unsqueeze(2).expand_as(ch).mul(ch).sum(0)
            #    chunkEmbedding[num] = chEmb
            #    chunkEmb[num] = torch.cat((emo, chEmb), dim=1)
            #    num += 1
            
            # employ self attention to generate chunk embedding version two
            for i, ch in enumerate(chunks):
                input = torch.cat((emo.repeat(ch.size(0), 1, 1), ch), dim=2)
                ch = self.dropout(ch)
                seq_len, batch_size, feature_dim = input.size()
                scores = self.scorer(input.contiguous().view(-1, feature_dim)).view(batch_size, seq_len)
                scores = F.softmax(scores, dim=1).transpose(0, 1)
                chEmb = scores.unsqueeze(2).expand_as(ch).mul(ch).sum(0)
                chunkEmb[num] = torch.cat((emo, chEmb), dim=1)
                chunkEmbedding[num] = chEmb
                #chEmb = scores.unsqueeze(2).expand_as(input).mul(input).sum(0)
                #chunkEmb[num] = chEmb
                num += 1
            # classify the emotion-chunk pairs
            #chunkEmb = torch.cat((emo.repeat(chunkEmb.size(0), 1), chunkEmb), dim=1)
            hidden = self.dropout(self.ac(self.W(chunkEmb)))
            #hidden = self.dropout(self.ac(self.W2(delta)))
            out = torch.log_softmax(self.Wo(hidden), dim=1)
            out_ = out.contiguous().view(-1, 2)
            phase2_out.append(out_)

            # emotion-cause pair, if t_ratio is 0 then output from pahse2 else real label (teacher forcing)
            thr = torch.rand(1)
            if thr.item() > 1-t_ratio:
                idx = label_ck[j].view(-1)  # [num_chunck]
            else:
                idx = out_.argmax(dim=1)  # [num_chunck]
            id_chks = torch.nonzero(idx)

            # phase 3 if there is emotion-chunk pair found in phase 2, then phase3 start
            if id_chks.size(0) != 0:
                # prepare label
                L_cau = torch.zeros(text_cau.size(0))
                for label in label3[j]:
                    if label.item() != -1:
                        L_cau[label.item()] = 1
                chk_L_cau = torch.split(L_cau, chunksize, 0)

                chunks_sel = []
                label3_sel = []
                sz_ck = []
                chunkEmb_sel = []

                for id in id_chks:
                    chunks_sel.append(chunks[id])
                    chunkEmb_sel.append(chunkEmbedding[id].repeat(chunks[id].size(0), 1, 1))
                    label3_sel.append(chk_L_cau[id])
                    sz_ck.append(chunks[id].size(0))
                # chunks_sel = chunks[id_chks]
                utt_c = torch.cat(chunks_sel, dim=0)
                chunk_c = torch.cat(chunkEmb_sel, dim=0)

                # item attention
                delta = ECPEC.item_att(emo, utt_c.squeeze(1))
                features = torch.cat((delta, chunk_c.squeeze(1)), dim=-1)

                h = self.dropout(self.ac_linear(self.W3(features)))
                h = self.cls(h)#.squeeze(1)
                p = torch.log_softmax(h, dim=1)
                p = p.contiguous().view(-1, 2)
                p_out.append(p)

                label3_out.extend(label3_sel)
                if not train:
                    phase3_out = []
                    l = 0
                    for sz in sz_ck:
                        phase3_out.append(p[l:l+sz])
                        l += sz
                    pp_out.extend(phase3_out)
                    
        if train:
            return phase2_out, p_out, label3_out
        else:
            return phase2_out, pp_out, label3_out

    @ staticmethod
    def item_att(x, y):
        item1 = torch.cat((x.repeat(y.size(0), 1), y), dim=-1)
        item2 = torch.norm(x - y, p=2, dim=-1, keepdim=True)
        item3 = torch.mul(x, y)
        delta = torch.cat((item1, item2, item3), dim=-1)

        return delta


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        pass

    def forward(self, *input):
        pass


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist


class LabelSmoothingLoss(nn.Module):
    """This is label smoothing loss function.
    """

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = smooth_one_hot(target, self.cls, self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def train_model(model, loss_function, loss_function_c, dataloader, epoch, t_ratio=0.5, smth=0, smth3=0.0, embedding=None, optimizer=None, train=True):
    losses = []
    # phase2 and phase3 tasks
    predec = []
    labelec = []
    pred3 = []
    labelp3 = []

    LSloss = LabelSmoothingLoss(2, smoothing=smth)
    assert not train or optimizer != None
    model.train()
    for data in dataloader:

        optimizer.zero_grad()

        ids_emo, ids_cau, label, label3, textid, sen_len = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
            
        word_encode = embedding(textid).squeeze()            
        log_phase2, log_phase3, label_phase3 = model(ids_emo, ids_cau, word_encode, sen_len, chunksize = 8, t_ratio=t_ratio, label_ck=label, label3=label3)  # batch*seq_len, n_classes

        log_phase2 = torch.cat(log_phase2, dim=0)
        log_phase3 = torch.cat(log_phase3, dim=0) if len(log_phase3) !=0 else []
        label_phase3 = torch.cat(label_phase3, dim=0).long().cuda() if len(label_phase3) != 0 else torch.tensor([]).view(-1)

        label_ = label.view(-1)

        skip = False if len(log_phase3) != 0 else True
        if smth>0:

            loss_2 = LSloss(log_phase2, label_)
        else:
            loss_2 = loss_function(log_phase2, label_)
        if not skip:
            loss_3 = LSloss(log_phase3, label_phase3) if smth3>0 else loss_function_c(log_phase3, label_phase3)
            loss = loss_2 + loss_3
        else:
            loss = loss_2

        # phase 2
        pred_e = torch.argmax(log_phase2, 1)  # batch*seq_len
        predec.append(pred_e.data.cpu().numpy())
        labelec.append(label_.data.cpu().numpy())

        # phase 3
        if not skip:
            pred_3 = torch.argmax(log_phase3, 1)  # batch*seq_len
            pred3.append(pred_3.data.cpu().numpy())
            labelp3.append(label_phase3.data.cpu().numpy())
        else:
            pred3.append(torch.zeros_like(label_phase3).numpy())
            labelp3.append(label_phase3.data.cpu().numpy())
            pass

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    if predec != []:
        predec = np.concatenate(predec)
        labelec = np.concatenate(labelec)
        if not skip:
            pred3 = np.concatenate(pred3)
            labelp3 = np.concatenate(labelp3)

    else:
        return float('nan'), float('nan'), float('nan'),float('nan'),float('nan'),float('nan'),float('nan')

    avg_loss = round(np.average(losses), 4)

    p_p2, r_p2, f_p2 = evaluate(predec, labelec)
    p_p3, r_p3, f_p3 = evaluate(pred3, labelp3)

    return avg_loss, p_p2, r_p2, f_p2, p_p3, r_p3, f_p3


def eval_model(model, loss_function, loss_function_c, dataloader, epoch, t_ratio=0.0, smth=0, smth3=0.00, embedding=None, optimizer=None,
                        train=False):
    losses = []
    # phase2 and phase3 tasks
    predec = []

    losses3 = []

    assert not train or optimizer != None
    model.eval()

    num_proposed_pairs = 0
    num_annotated_pairs = 0
    num_correct_pairs = 0
    LSloss = LabelSmoothingLoss(2, smoothing=smth)
    for data in dataloader:

        ids_emo, ids_cau, label, label3, bi_label_emo, \
        ids_cau, cLabels, phase3_label, textid, sen_len = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        #print('ids_emo',ids_emo.size(),ids_cau.size())
        if ids_emo.size(0) ==0 or ids_cau.size(0) ==0:
            a = torch.nonzero(cLabels)
            num_annotated_pairs += a.size(0)
        else:
            
            word_encode = embedding(textid).squeeze()
            log_phase2, log_phase3, label_phase3 = model(ids_emo, ids_cau, word_encode, sen_len, chunksize=8, t_ratio=t_ratio, label_ck=label,
                                                         label3=label3, train=False)  # batch*seq_len, n_classes
            count = 0

            if len(log_phase2) != phase3_label.size(0):

                print('phase3_label',phase3_label.size(),phase3_label)
                print('log_phase2', len(log_phase2))
                print("cLabels", cLabels.size(), cLabels)
                print("bi_label_emo",bi_label_emo)
            for emo_idx, p2 in enumerate(log_phase2):

                ck_ids_cau = torch.split(ids_cau.squeeze(1), 8, dim=0)
                seq = []
                pred_e_ = torch.argmax(p2, 1)
                ck_id = torch.nonzero(pred_e_)

                for id in ck_id:

                    pred_3_ = torch.argmax(log_phase3[count], dim=1)
                    try:
                        ut_id = torch.nonzero(pred_3_)
                    except RuntimeError:
                        print('utid')

                    try:
                        seq.append(ck_ids_cau[id.item()][ut_id])
                    except IndexError or RuntimeError:
                        print('seq')

                    count += 1

                if len(seq) != 0:
                    seq = torch.cat(seq, dim=0)

                    for idx in range(seq.size(0)):

                        if seq[idx].item()+1 in phase3_label[emo_idx]:
                            num_correct_pairs += 1
                    num_proposed_pairs += seq.size(0)


            a = torch.nonzero(cLabels)

            num_annotated_pairs += a.size(0)

            log_phase2 = torch.cat(log_phase2, dim=0)
            log_phase3 = torch.cat(log_phase3, dim=0) if len(log_phase3) != 0 else []
            label_phase3 = torch.cat(label_phase3, dim=0).long().cuda() if len(label_phase3) != 0 else torch.tensor([]).view(-1)

            label_ = label.view(-1)
            # umask = torch.ones([text_emo.size(0)*text_cau.size(0), 1]).float()

            skip = False if len(log_phase3) != 0 else True
            if smth > 0:

                loss_2 = LSloss(log_phase2, label_)
            else:
                loss_2 = loss_function(log_phase2, label_)
            if not skip:
                loss_3 = LSloss(log_phase3, label_phase3) if smth3 > 0 else loss_function_c(log_phase3, label_phase3)
                loss = loss_2 + loss_3
                losses3.append(loss_3.item())
            else:
                loss = loss_2

           # phase 2
            pred_e = torch.argmax(log_phase2, 1)  # batch*seq_len
            predec.append(pred_e.data.cpu().numpy())

            losses.append(loss.item())


    if predec != []:
        try:
            precision = round(num_correct_pairs / float(num_proposed_pairs), 4)
        except ZeroDivisionError:
            precision = float('nan')

        try:
            recall = round(num_correct_pairs / float(num_annotated_pairs), 4)
        except ZeroDivisionError:
            recall = float('nan')

        fscore = round(2 * precision * recall / (precision + recall) if precision+recall !=0 else 0, 4)

    else:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    avg_loss = round(np.average(losses), 4)
    avg_loss_3 = round(np.average(losses3), 4) if len(losses3) !=0 else 100

    return avg_loss, avg_loss_3, precision, recall, fscore



def evaluate(pred, label):

    # calculate precision
    num_proposed_pairs = np.sum(pred)
    res = pred + label
    num_correct_pairs = np.sum(res==2)
    precision = float(num_correct_pairs)/num_proposed_pairs

    # calculate recall
    # suitable for training phase 3 and phase 2
    num_annotated_pairs = np.sum(label)

    recall = float(num_correct_pairs)/num_annotated_pairs

    # calculate f1
    f1 = 2*precision*recall/(precision+recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L1',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=80, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    parser.add_argument('--chunk_size', type=int, default=8,
                        help='the size of chunk') 
    parser.add_argument('--t_ratio', type=float, default=0.2,
                        help='teaching force rate')
    parser.add_argument('--max_sen_len', type=int, default=30, 
                        help='max sentence length')                        
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    batch_size = args.batch_size
    n_classes = 2
    cuda = args.cuda
    n_epochs = args.epochs
    dropout = args.dropout
    chsize = args.chunk_size
    t_ratio = args.t_ratio
    max_sen_len = args.max_sen_len

    D_m = 100

    model = ECPEC(D_m, n_classes, dropout)
    
    # word2vec loading
    w2v_path = './key_words.txt'
    w2v_file = './glove.6B.100d.txt'
    word_idx_rev, word_id_mapping, word_embedding = load_w2v(D_m, w2v_path, w2v_file)
    word_embedding = torch.from_numpy(word_embedding)
    embedding = torch.nn.Embedding.from_pretrained(word_embedding,freeze=True).cuda()

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
        get_IEMOCAP_loaders(r'./ECPEC_phase_two_gcn_0.4_0.6_relu.pkl',
                            valid=0.0,
                            batch_size=batch_size,
                            num_workers=2,
                            word_idx=word_id_mapping,
                            max_sen_len=max_sen_len)  # 'BERT_ECPEC.pkl''./ECPEC_phase_two_xatt.pkl''ECPEC_InterEC.pkl'

    best_loss = 100

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, p_p2, r_p2, f_p2, p_p3, r_p3, f_p3 = train_model(model, loss_function, loss_function_c, train_loader, e, t_ratio=t_ratio, embedding = embedding, optimizer=optimizer, train=True)
        test_loss, test_loss_3, precision, recall, fscore = eval_model(model, loss_function, loss_function_c, test_loader, e, t_ratio=0.0, embedding = embedding)

        if best_loss == 100 or best_loss > test_loss_3:
            best_loss, best_precision, best_recall, best_fscore = test_loss_3, precision, recall, fscore

        print(
            'epoch {} train_loss {} train_precision_p2 {} train_recall_p2 {} train_fscore_p2 {} '
            'train_precision_p3 {} train_recall_p3 {} train_fscore_p3 {}'
            ' test_loss {} test_loss_3 {} test_precision {} test_recall {} test_fscore {} time {}'. \
            format(e + 1, train_loss, p_p2, r_p2, f_p2, p_p3, r_p3, f_p3,test_loss, test_loss_3, precision, recall, fscore, round(time.time() - start_time, 2)))

    print('Test performance..')
    print('Loss {} precision {} recall {} fscore{} '.format(best_loss, best_precision, best_recall, best_fscore))
