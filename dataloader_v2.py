import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd


class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.causeLabels, self.causeLabels2, self.causeLabels3, pred_sa, pred_cd, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        causeLabels = torch.stack((torch.LongTensor(self.causeLabels[vid]),torch.LongTensor(self.causeLabels2[vid]),
                                 torch.LongTensor(self.causeLabels3[vid])), 0)  # stack the cause three cause labels


        bi_label_emo = torch.LongTensor([0 if label == 2 else 1 for label in self.videoLabels[vid]])  # generate emotion label

        bi_label_cause = torch.LongTensor([1 if i in causeLabels else 0 for i in range(1, len(self.videoLabels[vid])+1)])  # generate cause label
        #cauid = torch.LongTensor([i if i+1 in causeLabels else -1 for i in range(len(self.videoLabels[vid]))])

        text = torch.FloatTensor(self.videoText[vid])  # text embedding of a given conversation
        emo_mask = bi_label_emo.unsqueeze(1).bool()   # boolean value of emotion label
        text_emo = torch.masked_select(text, emo_mask).contiguous().view(-1, 100)  # extracted text embedding of emotion utterance
        cau_mask = bi_label_cause.unsqueeze(1).bool()  # boolean value of cause label
        text_cau = torch.masked_select(text, cau_mask).contiguous().view(-1, 100)  # extracted text embedding of cause utterance
        #convid_to_cauid = torch.masked_select(cauid, cau_mask)
        ids_cau = torch.nonzero(bi_label_cause)  # the id of cause utterances in a conversation
        ids_emo = torch.nonzero(bi_label_emo)   # the id of emotion utterances in a conversation
        # chunks = list(torch.split(text_cau, 8, 0))
        chks_id = torch.split(ids_cau, 8, 0)   # split cause utterance into chunks
        #chck_label = #[[1 if id.item() in for id in chck for chck in chcks_id] for i in range(text_emo.size(0))]
        #chck_label = [ [ for cause_id in causeLabels[:,idx.item()]] for idx in ids_emo]
        chck_label = [[0 for ch_id in chks_id] for idx in ids_emo]  # initialize chck label for each emotion utterance
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


        return  text_emo,\
                text_cau,\
                chck_label,\
                p3_label,\
                vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i]) if i<4 else dat[i].tolist() for i in dat]