'''
Author: Li Wei
Email: wei008@e.ntu.edu.sg
'''

import os
import numpy as np
import json
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
import pickle
import pandas as pd

from scipy.sparse import linalg, dok_matrix, coo_matrix, csr_matrix
import numpy as np
import matplotlib.pyplot as plt

def test_csr(testdata, N, W):
    indices = [x for _ in range(W-1) for x in range(N**2)]
    ptrs = [N*(i) for i in range(N*(W-1))]
    ptrs.append(len(indices))

    data = []
    # loop along the first axis
    for i in range(W-1):
        vec = testdata[:,i].squeeze()

        # repeat vector N times
        for i in range(N):
            data.extend(vec)

    Hshape = ((N*(W-1), N**2))
    H = csr_matrix((data, indices, ptrs), shape=Hshape)
    return H


class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.causeLabels, self.causeLabels2, self.causeLabels3, self.sa_Tr, self.cd_Ta, self.pred_sa, self.pred_cd, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

        self.train = train
path = r'../ECPEC_phase_two_gcn_0.4_0.7_relu_full.pkl'
testset = IEMOCAPDataset(path=path, train=False)



path = '/home/maxwe11y/Desktop/weili/phase3/case_study/'
path2 = '/home/maxwe11y/Desktop/weili/phase3/case_study/JointEC/'
file_pred = 'predicted_dialogue_1_0.5.json'
file_truth = 'ground_truth_dialogue_1_0.5.json'

file_pred_EC = 'predicted_dialogue_jointEC_1.0.json'
file_truth_EC = 'ground_truth_dialogue_jointEC_1.0.json'

with open(os.path.join(path, file_pred), 'r') as f:
    pred = json.load(f)
f.close()

with open(os.path.join(path, file_truth), 'r') as g:
    ground = json.load(g)
g.close()

with open(os.path.join(path2, file_pred_EC), 'r') as ff:
    pred_EC = json.load(ff)
ff.close()

with open(os.path.join(path2, file_truth_EC), 'r') as gg:
    ground_EC = json.load(gg)
gg.close()

count_pair = {}
count=0
for i in pred.keys():
    dialog = pred[i]
    label = ground[i]

    dialog_EC = pred_EC[i]
    label_EC = ground_EC[i]

    dialog_ = np.array(dialog)
    label_ = np.array(label)

    dialog_EC_ = np.array(dialog_EC)
    label_EC_ = np.array(label_EC)

    count+=sum([sum(i) for i in label])

    x1 = sum([sum(i) for i in dialog])
    x2 = sum([sum(i) for i in label])
    x3 = len((dialog_ - label_).nonzero()[0])
    x4 = ((x1+x2)-x3)/2

    x5 = sum([sum(i) for i in dialog_EC])
    x6 = sum([sum(i) for i in label_EC])
    x7 = len((dialog_EC_ - label_EC_).nonzero()[0])
    x8 = ((x5+x6)-x7)/2

    x9 = len(testset.videoSentence[i])

    x10 = x4 - x8

    count_pair[i] = (x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)

#sorted_res = sorted(count_pair.items(), key=lambda k: k[1][2])
sorted_res = list(count_pair.items())
sorted_res.sort(key=lambda k: k[1][2])

# TODO visualize res
# res = pd.DataFrame(count_pair, index=['Proposed_pairs', 'labelled_pairs', 'Difference', 'Correct_pairs', 'Proposed_pairs', 'labelled_pairs', 'Difference', 'Correct_pairs','Text_length', 'Gap'])
# res.to_excel(os.path.join(path, 'statistics' + '.xlsx'), sheet_name='statistics')

#id = 'Ses05F_impro01'
#id = 'Ses05F_impro08'  # good for photoid = 'Ses05M_impro06'  # good for photo
path_figure = '/home/maxwe11y/Desktop/weili/phase3/case_study/figure/'
for id in pred.keys():
    text = testset.videoSentence[id]
    proposed = np.array(pred[id])
    compress_pro = csr_matrix(proposed)
    true_label = np.array(ground[id])
    compress_label = csr_matrix(true_label)

    proposed_EC = np.array(pred_EC[id])
    compress_pro_EC = csr_matrix(proposed_EC)
    true_label_EC = np.array(ground_EC[id])
    compress_label_EC = csr_matrix(true_label_EC)

    # TODO visualize the predicted result
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('Joint-ECW')
    plt.imshow(compress_pro.toarray(), cmap=plt.get_cmap('ocean_r'))
    #plt.savefig("vis.png", dpi=240)
    #plt.show()
    plt.subplot(2, 2, 3)
    plt.title('Ground Truth of Joint-ECW')
    plt.imshow(compress_label.toarray(), cmap=plt.get_cmap('ocean_r'))

    plt.subplot(2, 2, 2)
    plt.title('Joint-EC')
    plt.imshow(compress_pro_EC.toarray(), cmap=plt.get_cmap('ocean_r'))
    #plt.show()

    plt.subplot(2, 2, 4)
    plt.title('Ground Truth of Joint-EC')
    plt.imshow(compress_label_EC.toarray(), cmap=plt.get_cmap('ocean_r'))

    plt.tight_layout(pad=1.08)
    plt.savefig(os.path.join(path_figure, id+".png"), dpi=240)

    plt.close()
    #plt.show()



    #print('error!')
#
#
#     case_data = {}
#     case_data['text'] = text
#     case_data['emotion'] = testset.videoLabels[id]
#
#
#     case_data['predicted'] = [(np.array(i)+1).tolist() for i in compress_pro.tolil().rows] # compress_pro.tolil().rows
#     case_data['pred_label'] = [(np.array(i)+1).tolist() for i in compress_label.tolil().rows] #compress_label.tolil().rows
#     case_data['predicted_EC'] = [(np.array(i)+1).tolist() for i in compress_pro_EC.tolil().rows] # compress_pro_EC.tolil().rows
#     case_data['pred_label_EC'] = [(np.array(i)+1).tolist() for i in compress_label_EC.tolil().rows] # compress_label_EC.tolil().rows
#     case_data['cause_1'] = testset.causeLabels[id]
#     case_data['cause_2'] = testset.causeLabels2[id]
#     case_data['cause_3'] = testset.causeLabels3[id]
#     case_data['speaker'] = testset.videoSpeakers[id]
#
#     case_pd = pd.DataFrame(case_data)
#     path = '/home/maxwe11y/Desktop/weili/phase3/case_study/case/'
#     case_pd.to_excel(os.path.join(path,'case_study' + id + '.xlsx'), sheet_name=id)

print('finished!')