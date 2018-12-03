#!/usr/bin/env python3
import pdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from gaussian import gaussian


def load_x(path):
    X = []
    with open(path) as f:
        lines = f.readlines()
        for l in lines:
            values = l.split(',')
            X.append([
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3])
            ])
    return X


def load_y(path):
    with open(path) as f:
        return [line.rstrip() for line in f.readlines()]


X_train = np.array(load_x('../data/train_states.txt'))
y_train = np.array(load_y('../data/train_labels.txt'))
X_test = np.array(load_x('../data/test_states.txt'))
y_test = np.array(load_y('../data/test_labels.txt'))


# build features
lane_width = 4.0
def make_d_relative(X):
    for sample in X:
        sample[1] = sample[1] % lane_width

left = []
keep = []
right = []


for i in range(len(y_train)):
    print(X_train[i])
    if y_train[i] == 'left':
        left.append(X_train[i])
    if y_train[i] == 'keep':
        keep.append(X_train[i])
    if y_train[i] == 'right':
        right.append(X_train[i])

# make_d_relative(left)
# make_d_relative(keep)
# make_d_relative(right)

left = np.array(left)
keep = np.array(keep)
right = np.array(right)

# build model
mus_left = left.sum(0) / len(left)
mus_keep = keep.sum(0) / len(keep)
mus_right = right.sum(0) / len(right)

plt.scatter(left[:, 0], left[:, 1], c='blue')
plt.scatter(keep[:, 0], keep[:, 1], c='black')
plt.scatter(right[:, 0], right[:, 1], c='red')
# plt.show()
plt.scatter(mus_left[0], mus_left[1], s=200, c='blue')
plt.scatter(mus_keep[0], mus_keep[1], s=200, c='black')
plt.scatter(mus_right[0], mus_right[1], s=200, c='red')
plt.show()


stds_left = (((left - mus_left) ** 2).sum(0) / len(left)) ** (1/2)
stds_keep = (((keep - mus_keep) ** 2).sum(0) / len(keep)) ** (1/2)
stds_right = (((right - mus_right) ** 2).sum(0) / len(right)) ** (1/2)

# predict
# make_d_relative(X_test)
y_pred = []

import pdb; pdb.set_trace()
for sample in X_test:  # per data point
    p_left = 1
    p_keep = 1
    p_right = 1
    for i in range(len(sample)):  # per feature
        p_left = p_left * gaussian(sample[i], mus_left[i], stds_left[i])
        p_keep = p_keep * gaussian(sample[i], mus_keep[i], stds_keep[i])
        p_right = p_right * gaussian(sample[i], mus_right[i], stds_right[i])

    prob = [p_left, p_keep, p_right]
    labels = ['left', 'keep', 'right']

    y = labels[prob.index(max(prob))]
    y_pred.append(y)


y_pred = np.array(y_pred)
# # using sklearn
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train).predict(X_test)

errors = y_pred[y_pred != y_test]
accuracy = ((len(y_pred) - (len(errors)))/len(y_pred)) * 100
print("Accuracy: %d%%" % accuracy)
