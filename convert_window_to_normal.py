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


# precision_tf = [0.3603, 0.3526, 0.3593, 0.3596, 0.3572, 0.3887, 0.3634, 0.3769, 0.3863, 0.3812, 0.3762]
# recall_tf = [0.6415, 0.6688, 0.6576, 0.6576, 0.6605, 0.5898, 0.6088, 0.61, 0.5767, 0.6011, 0.569]
# count = 1682
#
# precision = [0.348, 0.3807, 0.3541, 0.3454, 0.3188, 0.3063, 0.3172, 0.3223, 0.3302, 0.324]
# recall = [0.628, 0.5898, 0.5832, 0.5619, 0.5082, 0.461, 0.4556, 0.4427, 0.4667, 0.4353]
#
# f = converts_batch(precision, recall, count)

precision = 0.4927
recall = 0.6792
count = 1682

p, r, f = converts(precision, recall, count)

print(p, r, f)