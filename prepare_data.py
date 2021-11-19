# encoding:utf-8
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb, time
import string

def print_time():
    print ('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

def load_w2v(embedding_dim, train_file_path, embedding_path):
    print('\nload embedding...')
    print(embedding_path)

    words = ['happy', 'sadness', 'anger', 'neutral', 'excited', 'frustrated']
    inputFile1 = open(train_file_path, 'r')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        #emotion, clause = line[1], line[-1]
        if line[2] == "":
            clause = line[3]
        else:
            clause = line[2]
        clause = clause.lower().replace("'s", "").replace("'", "")
        for i in range(len(string.punctuation)):
            clause = clause.replace(string.punctuation[i], " ")
        words.extend( clause.split())
    words = set(words)
    word_idx = dict((c, k + 1) for k, c in enumerate(words))
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))
    word_idx['unk'] = 0
    word_idx_rev[0] = 'unk'
    w2v = {}
    inputFile2 = open(embedding_path, 'r', encoding='latin')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    #embedding_pos = [list(np.zeros(embedding_dim_pos))]
    #embedding_pos.extend( [list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)] )

    embedding = np.array(embedding)
    
    print("embedding.shape: {}:".format(embedding.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding