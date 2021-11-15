#!/usr/bin/env python
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
from dataloader import IEMOCAPDataset
import torch.nn.functional as F
from tr import TextTransformer, PositionalEncoding, LearnedPositionEncoding
import argparse
import pickle

np.random.seed(1234)
torch.random.manual_seed(1234)
torch.cuda.manual_seed(1234)


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(path, batch_size=1, valid=0.0, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=path)
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

    testset = IEMOCAPDataset(path=path, train=False)
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

        self.W = nn.Linear(2 * input_dim, input_dim)
        self.Wo = nn.Linear(input_dim, self.n_class)

        self.gru = nn.GRU(input_size=input_dim, hidden_size=input_dim, num_layers=1, bidirectional=False)

        self.ac = nn.Sigmoid()
        self.ac_linear = nn.ReLU()
        self.ac_tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.nocuda = False

        self.transa = TextTransformer(LearnedPositionEncoding, d_model=4 * self.input_dim, d_out=self.input_dim,
                                      nhead=4, num_encoder_layers=3,
                                      dim_feedforward=512)
        self.W3 = nn.Linear(3*input_dim+1, input_dim)
        self.cls = nn.Linear(input_dim, self.n_class)

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

    def forward(self, text_emo, text_cau, chunksize=8, t_ratio=1, label_ck=None,
                label3=None, share_rep=True, train=True):
        """
        :param text_emo:-->seq, batch, dim
                text_cau:-->seq, batch, dim
                label_ck:-->seq, num_chunk
        :return:
        """

        # the emotion cause chunk pair extraction task -- GRU variant

        label3 = label3.squeeze(1).cuda()

        if share_rep:
            text_emo = self.ac_linear(self.rep(text_emo))
            text_cau = self.ac_linear(self.rep(text_cau))
        chunks = torch.split(text_cau, chunksize, 0)  # [num_chunk, chunksize,1, dim] --> tuple

        p_out = []
        label3_out = []
        phase2_out = []
        pp_out = []

        # iterate text_emo and pair up all the emotion-chunk and emotion-cause pairs
        for j, emo in enumerate(text_emo):
            # initialize the emotion-chunk embedding for each text_emo
            chunkEmb = torch.empty([len(chunks), 2 * text_emo.size(2)]).cuda()
            num = 0
            # employ GRU to generate chunk embedding chEmb and chunk-emotion embedding
            for i, ch in enumerate(chunks):
                h = torch.zeros((1, text_emo.size(1), text_emo.size(2))).cuda()
                chEmb, hn = self.gru(ch, h)
                chunkEmb[num] = torch.cat((emo, chEmb[-1]), dim=1)
                num +=1

            # employ attention to generate chunk embedding
            #for i, ch in enumerate(chunks):
            #    score = torch.matmul(ch, emo.transpose(0, 1)).squeeze(-1).transpose(0, 1)/10.0
            #    chEmb = torch.matmul(score.unsqueeze(0), ch.permute(1, 0 ,2))
            #    chunkEmb[num] = torch.cat((emo, chEmb[-1]), dim=1)
            #    num += 1

            # classify the emotion-chunk pairs
            hidden = self.dropout(self.ac(self.W(chunkEmb)))
            out = torch.log_softmax(self.Wo(hidden), dim=1)
            out_ = out.contiguous().view(-1, 2)
            phase2_out.append(out_)

            # emotion-cause pair, if t_ratio is 0 then output from pahse2 else real label (teacher forcing)
            thr = torch.rand(1)
            if thr.item() > 1-t_ratio:
                #out_ = [[0, 1] if ck_l.item() == 1 else [1, 0] for ck_l in label_ck[j]]
                idx = label_ck[j].view(-1)  # [num_chunck]
            else:
                idx = out_.argmax(dim=1)  # [num_chunck]
            id_chks = torch.nonzero(idx)
            #dim = text_cau.size(-1)

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

                for id in id_chks:
                    chunks_sel.append(chunks[id])
                    label3_sel.append(chk_L_cau[id])
                    sz_ck.append(chunks[id].size(0))
                # chunks_sel = chunks[id_chks]
                utt_c = torch.cat(chunks_sel, dim=0)
                #pair = torch.empty([utt_c.size(0), 3 * dim])
                #for cau in utt_c:
                item1 = torch.cat((emo.repeat(utt_c.size(0),1).unsqueeze(1), utt_c), dim=-1)
                item2 = torch.norm(emo-utt_c, p=2, dim=-1, keepdim=True)
                item3 = torch.mul(emo, utt_c)
                delta = torch.cat((item1,item2, item3), dim=-1)

                h = self.ac_linear(self.W3(delta))
                h = self.cls(h).squeeze(1)
                p = torch.log_softmax(h, dim=1)
                p = p.contiguous().view(-1, 2)
                p_out.append(p)
                #for item in label3_sel:
                    #label3_out.append(item)
                label3_out.extend(label3_sel)
                if not train:
                    phase3_out = []
                    l = 0
                    for sz in sz_ck:
                        #print(sz_ck)
                        phase3_out.append(p[l:l+sz])
                        #print(phase3_out)
                        l += sz
                    pp_out.extend(phase3_out)
                    #print("id_chks:",id_chks)


        #phase2_out = torch.cat(phase2_out, dim=0)
        #logit_out = torch.cat(p_out, dim=0) if len(p_out) !=0 else []
        #label3_out = torch.cat(label3_out, dim=0).long().cuda() if len(label3_out) !=0 else []
        if train:
            return phase2_out, p_out, label3_out
        else:
            return phase2_out, pp_out, label3_out


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        pass

    def forward(self, *input):
        pass


def train_model(model, loss_function, loss_function_c, dataloader, epoch, t_ratio=1.0, optimizer=None, train=True):
    losses = []
    # phase2 and phase3 tasks
    predec = []
    labelec = []
    #maskec = []
    pred3 = []
    labelp3 = []

    assert not train or optimizer != None
    model.train()
    #count = 0
    for data in dataloader:

        optimizer.zero_grad()

        text_emo, text_cau, label, label3 = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        log_phase2, log_phase3, label_phase3 = model(text_emo, text_cau, t_ratio=t_ratio, label_ck=label, label3=label3)  # batch*seq_len, n_classes

        log_phase2 = torch.cat(log_phase2, dim=0)
        log_phase3 = torch.cat(log_phase3, dim=0) if len(log_phase3) !=0 else []
        label_phase3 = torch.cat(label_phase3, dim=0).long().cuda() if len(label_phase3) !=0 else []

        label_ = label.view(-1)
        # umask = torch.ones([text_emo.size(0)*text_cau.size(0), 1]).float()

        skip = False if len(log_phase3) != 0 else True
        loss_2 = loss_function(log_phase2, label_)
        #n1 = label_.size(0)
        if not skip:
            loss_3 = loss_function_c(log_phase3, label_phase3.view(-1))
            #n2 = log_phase3.size(0)
            loss = 0.2*loss_2 + 0.8*loss_3
        else:
            loss = loss_2

        # phase 2
        pred_e = torch.argmax(log_phase2, 1)  # batch*seq_len
        predec.append(pred_e.data.cpu().numpy())
        labelec.append(label_.data.cpu().numpy())
        # maskec.append(umask.view(-1).cpu().numpy())

        # phase 3
        if not skip:
            pred_3 = torch.argmax(log_phase3, 1)  # batch*seq_len
            pred3.append(pred_3.data.cpu().numpy())
            labelp3.append(label_phase3.view(-1).data.cpu().numpy())
        else:
            pred3.append(torch.zeros_like(label_phase3).numpy())
            labelp3.append(label_phase3.view(-1).data.cpu().numpy())
            pass

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        #count += 1
        #if count==3:
        #   break

    if predec != []:
        predec = np.concatenate(predec)
        labelec = np.concatenate(labelec)
        #maskec = np.concatenate(maskec)
        # print(Counter(labels.tolist()))
        if not skip:
            pred3 = np.concatenate(pred3)
            labelp3 = np.concatenate(labelp3)

    else:
        return float('nan'), float('nan'), float('nan'),float('nan'),float('nan'),float('nan'),float('nan')

    avg_loss = round(np.average(losses), 4)

    p_p2, r_p2, f_p2 = evaluate(predec, labelec)
    p_p3, r_p3, f_p3 = evaluate(pred3, labelp3)

    return avg_loss, p_p2, r_p2, f_p2, p_p3, r_p3, f_p3


def eval_model(model, loss_function, loss_function_c, dataloader, epoch, t_ratio=0.0, optimizer=None,
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
    for data in dataloader:

        text_emo, text_cau, label, label3, bi_label_emo, \
        bi_label_cause, ids_emo, ids_cau, cLabels, phase3_label = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        if text_emo.size(0) ==0 or text_cau.size(0) ==0:
            a = torch.nonzero(cLabels)
            num_annotated_pairs += a.size(0)
        else:

            log_phase2, log_phase3, label_phase3 = model(text_emo, text_cau, t_ratio=t_ratio, label_ck=label,
                                                         label3=label3, train=False)  # batch*seq_len, n_classes
            count = 0
            d_emo = torch.zeros(cLabels.size(0))
            for j in ids_emo:
                d_emo[j.item()] = 1


            if len(log_phase2) != phase3_label.size(0):

                print('phase3_label',phase3_label.size(),phase3_label)
                print('log_phase2', len(log_phase2))
                print("cLabels", cLabels.size(), cLabels)
                print("bi_label_emo",bi_label_emo)
            for emo_idx, p2 in enumerate(log_phase2):

                ck_ids_cau = torch.split(ids_cau.squeeze(1), 8, dim=0)
                seq = []
                # seq_ = torch.LongTensor([0, 0, 0])
                pred_e_ = torch.argmax(p2, 1)
                ck_id = torch.nonzero(pred_e_)

                for id in ck_id:

                    pred_3_ = torch.argmax(log_phase3[count], dim=1)
                    try:
                        ut_id = torch.nonzero(pred_3_)
                    except RuntimeError:
                        print('utid')

                    try:
                        #print(ut_id)
                        #print(ck_ids_cau)
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
            label_phase3 = torch.cat(label_phase3, dim=0).long().cuda() if len(label_phase3) != 0 else []

            label_ = label.view(-1)
            # umask = torch.ones([text_emo.size(0)*text_cau.size(0), 1]).float()

            skip = False if len(log_phase3) != 0 else True
            loss_2 = loss_function(log_phase2, label_)
            #n1 = label_.size(0)
            if not skip:
                loss_3 = loss_function_c(log_phase3, label_phase3.view(-1))
                #n2 = log_phase3.size(0)
                loss = 0.2*loss_2 + 0.8*loss_3
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
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.30, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
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

    D_m = 100

    model = ECPEC(D_m, n_classes, dropout)

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
        get_IEMOCAP_loaders(r'./ECPEC_phase_two_xatt.pkl',
                            valid=0.0,
                            batch_size=batch_size,
                            num_workers=2)

    best_loss = 100

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, p_p2, r_p2, f_p2, p_p3, r_p3, f_p3 = train_model(model, loss_function, loss_function_c, train_loader, e, optimizer=optimizer, train=True)
        #valid_loss, valid_p_p2, valid_r_p2, valid_f_p2, valid_p_p3, valid_r_p3, valid_f_p3 = train_model(model, loss_function, loss_function_c, valid_loader, e)
        test_loss, test_loss_3, precision, recall, fscore = eval_model(model, loss_function, loss_function_c, test_loader, e, t_ratio=0.0)

        if best_loss == 100 or best_loss > test_loss_3:
            best_loss, best_precision, best_recall, best_fscore = test_loss_3, precision, recall, fscore

        print(
            'epoch {} train_loss {} train_precision_p2 {} train_recall_p2 {} train_fscore_p2 {} '
            'train_precision_p3 {} train_recall_p3 {} train_fscore_p3 {}'
            ' test_loss {} test_loss_3 {} test_precision {} test_recall {} test_fscore {} time {}'. \
            format(e + 1, train_loss, p_p2, r_p2, f_p2, p_p3, r_p3, f_p3,test_loss, test_loss_3, precision, recall, fscore, round(time.time() - start_time, 2)))

    print('Test performance..')
    print('Loss {} precision {} recall {} fscore{} '.format(best_loss, best_precision, best_recall, best_fscore))