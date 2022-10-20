#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

# Author: Wei Li
# Email: wei008@e.ntu.edu.sg


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, \
    classification_report, precision_recall_fscore_support

from torch.nn.utils.rnn import pad_sequence
from torch.nn import utils as nn_utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader_gcn import IEMOCAPDataset
import torch.nn.functional as F
import argparse

from GCN_functions import batch_graphify, MaskedEdgeAttention
from torch_geometric.nn import RGCNConv
import pickle
from prepare_data import *
np.random.seed(1234)
torch.random.manual_seed(1234)
torch.cuda.manual_seed(1234)


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

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=64, dropout=0.5,
                 relations=2, no_cuda=False):

        super(GraphNetwork, self).__init__()

        self.conv1 = RGCNConv(num_features, hidden_size, num_relations=relations, num_bases=30)

    def forward(self, x, edge_index, edge_type, edge_norm=None, seq_lengths=None, umask=None):
        out = self.conv1(x, edge_index, edge_type)
        return out


class ECPEC(nn.Module):

    def __init__(self, input_dim, n_class, dropout, acti):
        super(ECPEC, self).__init__()
        self.input_dim = input_dim
        self.n_class = n_class

        self.We = nn.Linear(2 * input_dim, input_dim)
        self.Wc = nn.Linear(2*input_dim, input_dim)

        self.Wsa = nn.Linear(2*input_dim, self.n_class)
        self.Wcd = nn.Linear(2*input_dim, self.n_class)

        self.egru = nn.LSTM(input_size=input_dim, hidden_size=int(input_dim / 2), num_layers=1, bidirectional=True)
        self.cgru = nn.LSTM(input_size=input_dim, hidden_size=int(input_dim / 2), num_layers=1, bidirectional=True)

        self.wgru = nn.LSTM(input_size=input_dim, hidden_size=int(input_dim / 2), num_layers=2, batch_first=True,
                            bidirectional=True)

        if acti=='sigmoid':
            self.ac = nn.Sigmoid()
        elif acti=='relu':
            self.ac = nn.ReLU()
        elif acti=='tanh':
            self.ac = nn.Tanh()
        self.ac_linear = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.dropouta = nn.Dropout(0.3)
        self.nocuda = False
        self.ln = torch.nn.LayerNorm([30,input_dim],elementwise_affine=True)


        self.graph_hidden_size = 200
        self.max_seq_len = 110
        self.relations = 2

        self.att = MaskedEdgeAttention(2 * input_dim, self.max_seq_len, self.nocuda)

        self.graph_net_ = GraphNetwork(2 * input_dim, self.n_class, self.graph_hidden_size, 0.5, self.relations, self.nocuda)
        self.wp = 10
        self.wf = 10


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

    def forward(self, x, sen_lengths, mask):
        """
        :param x:-->seq, batch, dim
        :return:
        """
        
        x = x.float()
        seq_lengths = sen_lengths.squeeze().cpu()
        pack = nn_utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        hw = torch.zeros((4, x.size(0), int(x.size(2) / 2))).cuda()
        cw = torch.zeros((4, x.size(0), int(x.size(2) / 2))).cuda()
        out,_ = self.wgru(pack,(hw, cw))
        unpacked = nn_utils.rnn.pad_packed_sequence(out, batch_first=True)
        index = torch.LongTensor([seq_lengths[i]+unpacked[0].size(1)*i-1 for i in range(unpacked[0].size(0))])
        U = unpacked[0].contiguous().view(-1, unpacked[0].size(2))[index].unsqueeze(1)

        # the conversational sentiment analysis task
        he = torch.zeros((2, U.size(1), int(U.size(2) / 2))).cuda()
        ce = torch.zeros((2, U.size(1), int(U.size(2) / 2))).cuda()
        f1 = U
        f2 = torch.cat((U[0].unsqueeze(0), U[0:-1]), dim=0)

        conf = self.dropout(self.ac(self.We(torch.cat((f2, f1), dim=2))))
        forward, (hne,cne) = self.egru(conf, (he,ce))
        rever_U = self._reverse_seq(U, mask)
        fx1 = rever_U
        fx2 = torch.cat((rever_U[0].unsqueeze(0), rever_U[0:-1]), dim=0)
        conb = self.dropout(self.ac(self.We(torch.cat((fx2, fx1), dim=2))))
        backward, (hne,cne) = self.egru(conb, (he,ce))

        backward = self._reverse_seq(backward, mask)

        ConE = torch.cat((forward, backward), dim=2)

        # the conversational cause utterance detection task
        hc = torch.zeros((2, U.size(1), int(U.size(2) / 2))).cuda()
        cc = torch.zeros((2, U.size(1), int(U.size(2) / 2))).cuda()
        conf = self.dropout(self.ac(self.Wc(torch.cat((f2,f1),dim=2))))
        forward, (hnc, cnc) = self.cgru(conf, (hc,cc))
        rever_U = self._reverse_seq(U, mask)
        fx1 = rever_U
        fx2 = torch.cat((rever_U[0].unsqueeze(0), rever_U[0:-1]), dim=0)
        conb = self.dropout(self.ac(self.Wc(torch.cat((fx2,fx1),dim=2))))
        backward, (hne, cnc) = self.cgru(conb, (hc,cc))
        backward = self._reverse_seq(backward, mask)
        ConC = torch.cat((forward, backward), dim=2)

        # information share through GCN
        infeature_ = torch.cat((ConE, ConC), dim=0)  # 2*node, batch, dim
        maskgcn_ = torch.cat((mask, mask), dim=1)  # batch, 2*seq
        adj_ = torch.ones(infeature_.size(1), infeature_.size(1)).cuda()
        adj_[0: ConE.size(0), 0: ConE.size(0)] = 0
        adj_[ConE.size(0):infeature_.size(1), ConE.size(0):infeature_.size(1)] = 0  # 2*seq, 2*seq
        types_ = torch.zeros(infeature_.size(0), infeature_.size(0)).cuda()
        types_[0: infeature_.size(0) // 2, infeature_.size(0) // 2: infeature_.size(0)] = 1

        vertex_, edge_, edge_type_ = batch_graphify(infeature_, maskgcn_, infeature_.size(0), self.wp,
                                                    self.wf, infeature_.size(0) // 2, att_model=self.att, type=types_)
        gcnf_ = self.graph_net_(vertex_, edge_, edge_type_.long())
        ConE_, ConC_ = torch.chunk(gcnf_, 2, dim=0)
        emoe = ConE_.unsqueeze(1)
        emoc = ConC_.unsqueeze(1)

        # conversational sentiment analysis task
        #feature_sa = self.transa(emoe)
        results_sa = torch.log_softmax(self.Wsa(emoe.squeeze(1)), dim=1)
        results_sa = results_sa.contiguous().view(-1, self.n_class)

        # the conversational cause utterance detection task
        #feature_cd = self.trancd(emoc)
        results_cd = torch.log_softmax(self.Wcd(emoc.squeeze(1)), dim=1)
        results_cd = results_cd.contiguous().view(-1, self.n_class)

        return results_sa, results_cd


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        pass

    def forward(self, *input):
        pass


def train_or_eval_model(model, loss_function, loss_function_cd, dataloader, epoch, prop, embedding, optimizer=None, train=False):
    losses = []
    # conversational sentiment analysis
    predse = []
    labelse = []
    maskse = []

    # cause utterance detection
    predsc = []
    labelsc = []
    masksc = []

    # store the prediction results of two subtasks
    saTask = {}
    cdTask = {}

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label_e, label_c, causeLabel, textid, sen_len = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]   
        word_encode = embedding(textid).squeeze()  # seq, batch, sen_len, dim
        log_sa, log_cd = model(word_encode, sen_len, umask)  # batch*seq_len, n_classes
        labelse_ = label_e.view(-1)  # batch*seq_len
        labelsc_ = label_c.view(-1)  # batch*seq_len

        loss_sa = loss_function(log_sa, labelse_, umask)
        loss_cd = loss_function_cd(log_cd, labelsc_, umask)
        loss = prop*loss_sa + (1-prop)*loss_cd

        # conversational sentiment analysis
        pred_e = torch.argmax(log_sa, 1)  # batch*seq_len
        predse.append(pred_e.data.cpu().numpy())
        labelse.append(labelse_.data.cpu().numpy())
        maskse.append(umask.view(-1).cpu().numpy())

        # emotion utterance detection
        pred_c = torch.argmax(log_cd, 1)  # batch*seq_len
        predsc.append(pred_c.data.cpu().numpy())
        labelsc.append(labelsc_.data.cpu().numpy())
        masksc.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * maskse[-1].sum())
        saTask[data[-1][0]] = list(pred_e.data.cpu().numpy())
        cdTask[data[-1][0]] = list(pred_c.data.cpu().numpy())

        if train:
            loss.backward()
            optimizer.step()
    if predse != []:
        predse = np.concatenate(predse)
        labelse = np.concatenate(labelse)
        maskse = np.concatenate(maskse)
        # print(Counter(labels.tolist()))

        # cause utterance detection
        predsc = np.concatenate(predsc)
        labelsc = np.concatenate(labelsc)
        masksc = np.concatenate(masksc)

    else:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), [], [], [], float('nan'), float('nan'), float('nan'), float('nan')

    avg_loss = round(np.sum(losses) / np.sum(maskse), 4)
    avg_accuracy_e = round(accuracy_score(labelse, predse, sample_weight=maskse) * 100, 2)
    avg_fscore_e = round(f1_score(labelse, predse, sample_weight=maskse, average='weighted') * 100, 2)

    # cause utterance detection
    avg_accuracy_c = round(accuracy_score(labelsc, predsc, sample_weight=masksc) * 100, 2)
    avg_fscore_c = round(f1_score(labelsc, predsc, sample_weight=masksc, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy_e, avg_accuracy_c, labelse, predse, labelsc, predsc, maskse, avg_fscore_e, avg_fscore_c, saTask, cdTask


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2',
                        help='L1 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.20, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=40, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False,
                        help='class weight')
    parser.add_argument('--activation', type=str, default='sigmoid',
                        help='activation function')
    parser.add_argument('--proportion', type=float, default=0.1, help='Proportion of Loss')
    parser.add_argument('--max_sen_len', type=int, default=20, help='max sentence length')
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
    prop = args.proportion
    acti = args.activation
    max_sen_len = args.max_sen_len

    D_m = 100

    model = ECPEC(D_m, n_classes, dropout, acti)
    
    # word2vec loading
    w2v_path = './key_words.txt'
    w2v_file = './glove.6B.100d.txt'
    word_idx_rev, word_id_mapping, word_embedding = load_w2v(D_m, w2v_path, w2v_file)
    word_embedding = torch.from_numpy(word_embedding)
    embedding = torch.nn.Embedding.from_pretrained(word_embedding, freeze=True).cuda()

    if cuda:
        model.cuda()

    loss_weights = torch.FloatTensor([
        1 / 0.227883,
        1 / 0.772117,
    ])

    loss_weights_c = torch.FloatTensor([
        1 / 0.420654,
        1 / 0.579346,
    ])

    if args.class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    if args.class_weight:
        loss_function_cd = MaskedNLLLoss(loss_weights_c.cuda() if cuda else loss_weights)
    else:
        loss_function_cd = MaskedNLLLoss()

    # params = ([p for p in model.parameters()] + [model.log_var_a] + [model.log_var_b])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader, test_loader = \
        get_IEMOCAP_loaders(r'./IEMOCAP_emotion_cause_features.pkl',
                            valid=0.0,
                            batch_size=batch_size,
                            num_workers=2,
                            word_idx=word_id_mapping,
                            max_sen_len=max_sen_len)

    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, train_acc_c, _, _, _, _, _, train_fscore, train_fscore_c, sa_pred, cd_pred = train_or_eval_model(model,
                                                                                                              loss_function,
                                                                                                              loss_function_cd,
                                                                                                              train_loader,
                                                                                                              e,
                                                                                                              prop,
                                                                                                              embedding,
                                                                                                              optimizer,
                                                                                                              True)
        valid_loss, valid_acc, valid_acc_c, _, _, _, _, _, val_fscore, val_fscore_c, _, _ = train_or_eval_model(model,
                                                                                                          loss_function,
                                                                                                          loss_function_cd,
                                                                                                          valid_loader,
                                                                                                          e,
                                                                                                          prop,
                                                                                                          embedding)
        test_loss, test_acc, test_acc_c, test_labele, test_prede, test_labelc, test_predc, test_mask, test_fscore, test_fscore_c, saTask, cdTask = train_or_eval_model(
            model, loss_function, loss_function_cd, test_loader, e, prop, embedding)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_labele, best_prede, best_labelc, best_predc, best_mask, saTrain, cdTrain, best_sa, best_cd = \
                test_loss, test_labele, test_prede, test_labelc, test_predc, test_mask, sa_pred, cd_pred, saTask, cdTask

        print('epoch {} train_loss {} train_acc {} train_acc_c {} train_fscore{} train_fscore_c {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_acc_c {} test_fscore {} test_fscore_c {} time {}'. \
            format(e + 1, train_loss, train_acc, train_acc_c, train_fscore, train_fscore_c, valid_loss, valid_acc, val_fscore, \
                   test_loss, test_acc, test_acc_c, test_fscore, test_fscore_c, round(time.time() - start_time, 2)))

    print('Test performance..', '\n')
    print('Loss {} accuracy {}'.format(best_loss,
                                       round(accuracy_score(best_labele, best_prede, sample_weight=best_mask) * 100,
                                             2)),'\n')
    print(classification_report(best_labele, best_prede, sample_weight=best_mask, digits=4),'\n')
    print(confusion_matrix(best_labele, best_prede, sample_weight=best_mask),'\n')

    print('Loss {} accuracy {}'.format(best_loss,
                                       round(accuracy_score(best_labelc, best_predc, sample_weight=best_mask) * 100,
                                             2)),'\n')
    print(classification_report(best_labelc, best_predc, sample_weight=best_mask, digits=4),'\n')
    print(confusion_matrix(best_labelc, best_predc, sample_weight=best_mask),'\n')


    # This code block is used to generate data with predicted emotion & cause labels included for the step 2 Joint-EC model
    # f1 = open(r'E:\git_projects\MemNN\IEMOCAP_features_raw.pkl', 'rb')
    path_out = r'./IEMOCAP_emotion_cause_features.pkl'
    videoIDs, videoSpeakers, videoLabels, causeLabels, causeLabels2, causeLabels3, videoText, \
    videoAudio, videoVisual, videoSentence, trainVid, \
    testVid = pickle.load(open(path_out, 'rb'), encoding='latin1')
    
    # print(videoLabels['Ses01F_impro01'],len(videoLabels['Ses01F_impro01']))
    # print(causeLabels['Ses01F_impro01'], len(causeLabels['Ses01F_impro01']))

    path = './pkl_gcn_full/ECPEC_phase_two_gcn_' + str(args.dropout) + '_' + str(args.proportion) + '_' + str(
        args.activation) + '.pkl'
    f = open(path, 'wb')
    data = [videoIDs, videoSpeakers, videoLabels, causeLabels, causeLabels2, causeLabels3, saTrain, cdTrain, best_sa,
            best_cd, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid]
    pickle.dump(data, f)
    f.close()
