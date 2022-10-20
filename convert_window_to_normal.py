'''
Author: Li Wei
Email: wei008@e.ntu.edu.sg
'''

def converts(precision, recall, count):
    annotated = count

    correct = recall*annotated

    new_recall = correct/1915

    fscore = 2*precision*new_recall/(precision+new_recall)

    return precision, new_recall, fscore

def converts_batch(precision, recall, count):

    f = []

    for i, j in zip(precision, recall):

        annotated = count

        correct = j*annotated

        new_recall = correct/1915

        fscore = 2*i*new_recall/(i+new_recall)

        f.append(fscore)

    return f

precision = 0.3901
recall = 0.5963
count = 1682

p, r, f = converts(precision, recall, count)

print(p, r, f)
