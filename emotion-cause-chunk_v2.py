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
                label3=None, share_rep=True):
        """
        :param text_emo:-->seq, batch, dim
                text_cau:-->seq, batch, dim
                label_ck:-->seq, num_chunk
        :return:
        """

        # the emotion cause chunk pair extraction task -- GRU variant

        #chunkEmb = torch.empty([text_emo.size(0)*len(chunks), 2*text_emo.size(2)]).cuda()
        label3 = label3.squeeze(1).cuda()

        if share_rep:
            text_emo = self.ac_linear(self.rep(text_emo))
            text_cau = self.ac_linear(self.rep(text_cau))
        chunks = torch.split(text_cau, chunksize, 0)  # [num_chunk, chunksize,1, dim] --> tuple

        p_out = []
        label3_out = []
        phase2_out = []

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
                for id in id_chks:
                    chunks_sel.append(chunks[id])
                    label3_sel.append(chk_L_cau[id])
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

        phase2_out = torch.cat(phase2_out, dim=0)
        logit_out = torch.cat(p_out, dim=0) if len(p_out) !=0 else []
        label3_out = torch.cat(label3_out, dim=0).long().cuda() if len(label3_out) !=0 else []

        return phase2_out, logit_out, label3_out


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        pass

    def forward(self, *input):
        pass


def train_or_eval_model(model, loss_function, loss_function_c, dataloader, epoch, t_ratio=1.0, optimizer=None, train=False):
    losses = []
    # phase2 and phase3 tasks
    predec = []
    labelec = []
    #maskec = []
    pred3 = []
    labelp3 = []


    # store the prediction results of two subtasks
    #ecTask = {}

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        text_emo, text_cau, label, label3 = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        log_phase2, log_phase3, label_phase3 = model(text_emo, text_cau, t_ratio=t_ratio, label_ck=label, label3=label3)  # batch*seq_len, n_classes
        label_ = label.view(-1)
        # umask = torch.ones([text_emo.size(0)*text_cau.size(0), 1]).float()

        skip = False if len(log_phase3) != 0 else True
        loss_2 = loss_function(log_phase2, label_)
        if not skip:
            loss_3 = loss_function_c(log_phase3, label_phase3.view(-1))
            loss = loss_2 + loss_3
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
            pass

        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()
    if predec != []:
        predec = np.concatenate(predec)
        labelec = np.concatenate(labelec)
        #maskec = np.concatenate(maskec)
        # print(Counter(labels.tolist()))
        if not skip:
            pred3 = np.concatenate(pred3)
            labelp3 = np.concatenate(labelp3)


    else:
        return float('nan'), float('nan'), [], [],  float('nan'),float('nan'), [], [],  float('nan')

    avg_loss = round(np.average(losses), 4)
    avg_accuracy_ec = round(accuracy_score(labelec, predec) * 100, 2)
    avg_fscore_ec = round(f1_score(labelec, predec, average='weighted') * 100, 2)
    if not skip:
        avg_accuracy_p3 = round(accuracy_score(labelp3, pred3) * 100, 2)
        avg_fscore_p3 = round(f1_score(labelp3, pred3, average='weighted') * 100, 2)
    else:
        avg_accuracy_p3 = None
        avg_fscore_p3 = None
    return avg_loss, avg_accuracy_ec, labelec, predec, avg_fscore_ec, avg_accuracy_p3, \
           labelp3, pred3, avg_fscore_p3


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

    # params = ([p for p in model.parameters()] + [model.log_var_a] + [model.log_var_b])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader, test_loader = \
        get_IEMOCAP_loaders(r'./ECPEC_phase_two_xatt.pkl',
                            valid=0.0,
                            batch_size=batch_size,
                            num_workers=2)

    best_loss = None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_accuracy_ec, _, _, train_fscore_ec, train_accuracy_p3, _, _, train_fscore_p3 = train_or_eval_model(model, loss_function, loss_function_c, train_loader, e, optimizer=optimizer, train=True)
        valid_loss, valid_accuracy_ec, _, _, valid_fscore_ec, valid_accuracy_p3, _, _, valid_fscore_p3 = train_or_eval_model(model, loss_function, loss_function_c, valid_loader, e)
        test_loss, test_accuracy_ec, test_labelec, test_predec, test_fscore_ec, test_accuracy_p3, \
        test_labelp3, test_pred3, test_fscore_p3 = train_or_eval_model(model, loss_function, loss_function_c, test_loader, e, t_ratio=0.0)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_accuracy_ec, best_labelec, best_predec, best_fscore_ec, best_accuracy_p3, \
            best_labelp3, best_pred3, best_fscore_p3 = \
                test_loss, test_accuracy_ec, test_labelec, test_predec, test_fscore_ec, test_accuracy_p3, \
                test_labelp3, test_pred3, test_fscore_p3

        print(
            'epoch {} train_loss {} train_acc_p2 {} train_fscore_p2 {} train_acc_p3 {} train_fscore_p3 {} valid_loss {} '
            'valid_acc_p2 {} val_fscore_p2 {} valid_acc_p3 {} val_fscore_p3 {} '
            'test_loss {} test_acc_2 {} test_fscore_2 {} test_acc_3 {} test_fscore_3 {} time {}'. \
            format(e + 1, train_loss, train_accuracy_ec, train_fscore_ec, train_accuracy_p3,train_fscore_p3, valid_loss, valid_accuracy_ec,
                   valid_fscore_ec,valid_accuracy_p3,valid_fscore_p3, \
                   test_loss, test_accuracy_ec, test_fscore_ec,test_accuracy_p3,test_fscore_p3, round(time.time() - start_time, 2)))

    print('Test performance..')
    print('Loss {} accuracy {}'.format(best_loss,
                                       round(accuracy_score(best_labelec, best_predec) * 100,
                                             2)))
    print(classification_report(best_labelec, best_predec, digits=4))
    print(confusion_matrix(best_labelec, best_predec))

    print('Loss {} accuracy {}'.format(best_loss,
                                       round(accuracy_score(best_labelp3, best_pred3) * 100,
                                             2)))
    print(classification_report(best_labelp3, best_pred3, digits=4))
    print(confusion_matrix(best_labelp3, best_pred3))


    # f1 = open(r'E:\git_projects\MemNN\IEMOCAP_features_raw.pkl', 'rb')
    """path_out = r'./IEMOCAP_emotion_cause_features.pkl'"""
    #f1 = open(r'C:\Users\liwei\Desktop\IEMOCAP_emotion_cause_features.pkl', 'rb')
    """videoIDs, videoSpeakers, videoLabels, causeLabels, causeLabels2, causeLabels3, videoText, \
    videoAudio, videoVisual, videoSentence, trainVid, \
    testVid = pickle.load(open(path_out, 'rb'), encoding='latin1')
    # print(videoLabels['Ses01F_impro01'],len(videoLabels['Ses01F_impro01']))
    # print(causeLabels['Ses01F_impro01'], len(causeLabels['Ses01F_impro01']))

    path = 'ECPEC_phase_two_xatt.pkl'
    f = open(path, 'wb')
    data = [videoIDs, videoSpeakers, videoLabels, causeLabels, causeLabels2, causeLabels3, best_sa, best_cd, videoText,
            videoAudio, videoVisual, videoSentence, trainVid, testVid]
    pickle.dump(data, f)
    f.close()"""