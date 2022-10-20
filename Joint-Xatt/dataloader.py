# Author: Wei Li
# Email: wei008@e.ntu.edu.sg

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
import string


class IEMOCAPDataset(Dataset):

    def __init__(self, path, word_idx=None, max_sen_len=None, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.causeLabels, self.causeLabels2, self.causeLabels3, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        #print(self.videoSentence)

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)
        
        self.max_sen_len = max_sen_len
        
        self.word_idx = word_idx

    def __getitem__(self, index):
        vid = self.keys[index]
        causeLabels = torch.stack((torch.LongTensor(self.causeLabels[vid]),torch.LongTensor(self.causeLabels2[vid]),
                                 torch.LongTensor(self.causeLabels3[vid])), 0)
        bi_label_emo = torch.LongTensor([0 if label == 2 else 1 for label in self.videoLabels[vid]])
        bi_label_cause = torch.LongTensor([1 if i in causeLabels else 0 for i in range(1, len(self.videoLabels[vid])+1)])
        
        # generate word-level embedding
        n_cut = 0
        text = self.videoSentence[vid]
        x_tmp = np.zeros((len(text), self.max_sen_len),dtype=np.int32)
        sen_len = np.zeros(len(text),dtype=np.int32)
        for i in range(len(text)):
            words = text[i].strip()
            words = words.lower().replace("'s", "").replace("'", "")
            for s in range(len(string.punctuation)):
                words = words.replace(string.punctuation[s], " ")
            words = words.split()
            sen_len[i] = min(len(words), self.max_sen_len)
            if len(words)==0:
                print(text[i],words)
            for j, word in enumerate(words):
                if j >= self.max_sen_len:
                    n_cut += 1
                    break
                if word not in self.word_idx:
                    x_tmp[i][j] = 0
                else:
                    x_tmp[i][j] = int(self.word_idx[word])
        textid = torch.from_numpy(x_tmp).long()
        sen_len = torch.from_numpy(sen_len)
        
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1, 0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               bi_label_emo,\
               bi_label_cause,\
               causeLabels,\
               textid,\
               sen_len,\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<10 else dat[i].tolist() for i in dat]
