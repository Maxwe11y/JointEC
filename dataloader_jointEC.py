'''
Author: Li Wei
Email: wei008@e.ntu.edu.sg
'''

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd


class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.causeLabels, self.causeLabels2, self.causeLabels3, self.pred_sa, self.pred_cd, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

        self.train = train

    def __getitem__(self, index):
        vid = self.keys[index]
        causeLabels = torch.stack((torch.LongTensor(self.causeLabels[vid]),torch.LongTensor(self.causeLabels2[vid]),
                                 torch.LongTensor(self.causeLabels3[vid])), 0)  # stack the cause three cause labels

        if self.train:
            bi_label_emo = torch.LongTensor([0 if label == 2 else 1 for label in self.videoLabels[vid]])  # generate emotion label

            bi_label_cause = torch.LongTensor([1 if i in causeLabels else 0 for i in range(1, len(self.videoLabels[vid])+1)])  # generate cause label TODO

        else:
            bi_label_emo = torch.LongTensor(self.pred_sa[vid])  # generate emotion label
            bi_label_cause = torch.LongTensor(self.pred_cd[vid])  # generate cause label
            #bi_label_emo = torch.LongTensor([0 if label == 2 else 1 for label in self.videoLabels[vid]])  # generate emotion label
            #bi_label_cause = torch.LongTensor([1 if i in causeLabels else 0 for i in range(1, len(self.videoLabels[vid])+1)])  # generate cause label
            #bi_label_emo_real = torch.LongTensor(
            #    [0 if label == 2 else 1 for label in self.videoLabels[vid]])  # generate emotion label
            #bi_label_cause_real = torch.LongTensor([1 if i in causeLabels else 0 for i in
            #                                        range(1, len(self.videoLabels[vid]) + 1)])  # generate cause label



        text = torch.FloatTensor(self.videoText[vid])  # text embedding of a given conversation
        emo_mask = bi_label_emo.unsqueeze(1).bool()   # boolean value of emotion label
        text_emo = torch.masked_select(text, emo_mask).contiguous().view(-1, 100)  # extracted text embedding of emotion utterance
        cau_mask = bi_label_cause.unsqueeze(1).bool()  # boolean value of cause label
        text_cau = torch.masked_select(text, cau_mask).contiguous().view(-1, 100)  # extracted text embedding of cause utterance
        ids_cau = torch.nonzero(bi_label_cause)  # the id of cause utterances in a conversation  # cauid_to_convid
        ids_emo = torch.nonzero(bi_label_emo)   # the id of emotion utterances in a conversation  # cauid_to_convid
        chks_id = torch.split(ids_cau, 8, 0)   # split cause utterance into chunks
        chck_label = [[0 for ch_id in chks_id] for idx in ids_emo]  # initialize chck label for each emotion utterance  # TODO
        convid_to_cauid = torch.LongTensor([torch.sum(bi_label_cause[:id]).item() if l.item() ==1 else 0 for id, l in enumerate(bi_label_cause)])  # map absolute cause label and relative cause label
        cLabels = causeLabels.permute(1, 0)  # swap the first and second dimension of causeLabels
        phase3_label = torch.masked_select(cLabels, emo_mask).view(-1,3) # use emo_mask to remove non-emotion utterances in cLabels
        p3_label = torch.LongTensor([[convid_to_cauid[l.item()-1] if l.item()!=0 else -1 for l in line] for line in phase3_label])  # labels of emotion-cause pair
        for i, id_emo in enumerate(ids_emo):
            for cause_id in causeLabels[:, id_emo.item()]:
                for j, ch_id in enumerate(chks_id):
                    if cause_id-1 in ch_id:
                        chck_label[i][j] = 1
                        continue

        chck_label = torch.LongTensor(chck_label)
        # chunks = [ch.permute(1,0) for ch in chunks]


        if self.train:
            return  text_emo,\
                    text_cau,\
                    chck_label,\
                    p3_label,\
                    vid
        else:
            #print('text', text.size())
            #print('text_emo',text_emo.size())
            #print('beforepadding',bi_label_emo.size(),bi_label_emo)
            #print('cLabels',cLabels)
            #print('afterpadding', pad_sequence([bi_label_emo]).size())
            return text_emo,\
                    text_cau,\
                    chck_label,\
                    p3_label,\
                    bi_label_emo,\
                    ids_cau, \
                    cLabels, \
                    phase3_label, \
                    vid
                    #ids_emo,\bi_label_cause, \



    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)


        if self.train:
            return [pad_sequence(dat[i]) if i<4 else dat[i].tolist() for i in dat]
        else:
            #print(dat)
            res = [pad_sequence(dat[i]) if i<8 else dat[i].tolist() for i in dat]
            #print('afterpadding', res[4].size(),res[4])
            return res