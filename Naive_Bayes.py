import numpy as np
from collections import Counter, defaultdict
import pandas as pd
from collections import defaultdict
import re
import operator



def frequency(l):
    no_of_examples = len(l)
    p = dict(Counter(l))
    for key in p.keys():
        p[key] = p[key] / float(no_of_examples)
    return p


def classifier(train_review, Attitude, test_review):
    classes = np.unique(Attitude)
    rows, cols = np.shape(train_review)
    likelihood = {}
    for cls in classes:
        likelihood[cls] = defaultdict(list)

    probabilities = frequency(Attitude)

    for cls in classes:
        row_indices = np.where(Attitude == cls)[0]
        subset = train_review[row_indices, :]
        r, c = np.shape(subset)
        for j in range(0, c):
            likelihood[cls][j] += list(subset[:, j])

    for cls in classes:
        for j in range(0, cols):
            likelihood[cls][j] = frequency(likelihood[cls][j])

    prediction = {}
    for cls in classes:
        probability = probabilities[cls]
        for i in range(0, len(test_review)):
            relative_values = likelihood[cls][i]
            if test_review[i] in relative_values.keys():
                probability *= relative_values[test_review[i]]
            else:
                probability *= 0
            prediction[cls] = probability
    print(prediction)
    print('Attitude:',max(prediction.items(), key=operator.itemgetter(1))[0])


train_review = np.asarray(pd.read_csv('C:/Users/Naive_bayes/AD_Train.csv', sep=','));

Attitude = np.asarray((1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
));

test_review = np.asarray(pd.read_csv('C:/Users/Naive_bayes/AD_test.csv', sep=','))
for i in range(len(test_review)):
    test = test_review[i]
    classifier(train_review, Attitude, test)
