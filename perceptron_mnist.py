# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:41:31 2020

@author: Sai
"""

import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from scipy.io import loadmat
import multiprocessing
from joblib import Parallel, delayed

# Read MNIST data
data = loadmat('mnist_digits.mat')
X = data['X']
Y = data['Y']
del data

# "Lift" the features
X = np.concatenate((X, np.ones([X.shape[0], 1])), axis = 1)

# Train - test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
num_train = X_train.shape[0]
num_test = X_test.shape[0]
num_digits = 10


# Perceptron V0
def perceptron_v0(max_iters, pred_every, X_train, X_test, Y_train, Y_test):
    acc = []
    w = np.zeros([num_digits, X_train.shape[1]])
    iters = 0
    while iters < max_iters:
        # Train
        for i in range(len(Y_train)):
            iters += 1
            for int_class in range(num_digits):
                Y_mult = 1 if Y_train[i] == int_class else -1
                if Y_mult * np.dot(w[int_class], X_train[i]) <= 0:
                    w[int_class] += Y_mult * X_train[i]

            if iters % pred_every == 0:
                # Predict
                Y_pred = np.zeros(Y_test.shape)
                for k in range(len(Y_test)):
                    preds = np.zeros(num_digits)
                    for int_class in range(num_digits):
                        preds[int_class] = np.dot(w[int_class], X_test[k])
                    Y_pred[k] = np.argmax(preds)

                # Test
                acc.append(zero_one_loss(Y_test, Y_pred))

    return acc

# Perceptron V1
def perceptron_v1(max_iters, pred_every, X_train, X_test, Y_train, Y_test):
    acc = []
    w = np.zeros([num_digits, X_train.shape[1]])

    untrained = list(range(num_digits))
    iters = 0
    while iters < max_iters:
        iters += 1
        # Train
        for int_class in untrained:
            Y_mult = [1 if Y_train[j] == int_class else -1 for j in range(len(Y_train))]
            margins = Y_mult * (X_train @ w[int_class].T)
#            margins = np.zeros(len(Y_train))
#            for i in range(len(Y_train)):
#                Y_mult = 1 if Y_train[i] == int_class else -1
#                margins[i] = Y_mult * np.dot(w[int_class], X_train[i])

            min_idx = np.argmin(margins)
            Y_mult = 1 if Y_train[min_idx] == int_class else -1
            if Y_mult * np.dot(w[int_class], X_train[min_idx]) <= 0:
                w[int_class] += Y_mult * X_train[min_idx]
            else:
                untrained.remove(int_class)

        if iters % pred_every == 0:
            # Predict
            Y_pred = np.zeros(Y_test.shape)
            for k in range(len(Y_test)):
                preds = np.zeros(num_digits)
                for int_class in range(num_digits):
                    preds[int_class] = np.dot(w[int_class], X_test[k])
                Y_pred[k] = np.argmax(preds)

            # Test
            acc.append(zero_one_loss(Y_test, Y_pred))

    return acc


# Perceptron V2
def perceptron_v2(max_iters, pred_every, X_train, X_test, Y_train, Y_test):
    acc = []
    w = np.zeros([num_digits, max_iters, X_train.shape[1]])
    c = np.zeros([num_digits, max_iters])
    for j in range(num_digits):
        c[j][0] = 1
    k = np.zeros(num_digits, dtype = 'uint32')

    iters = 0
    while iters < max_iters:
        # Train
        for i in range(len(Y_train)):
            iters += 1
            for int_class in range(num_digits):
                Y_mult = 1 if Y_train[i] == int_class else -1
                if Y_mult * np.dot(w[int_class][k[int_class]], X_train[i]) <= 0:
                    w[int_class][k[int_class] + 1] = w[int_class][k[int_class]] + Y_mult * X_train[i]
                    c[int_class][k[int_class] + 1] = 1
                    k[int_class] += 1
                else:
                    c[int_class][k[int_class]] += 1


            if iters % pred_every == 0:
                # Predict
                Y_pred = np.zeros(len(Y_test))
                for i in range(len(Y_test)):
                    preds = np.zeros(num_digits)
                    for int_class in range(num_digits):
                        preds[int_class] = np.dot(c[int_class][:k[int_class]], w[int_class][:k[int_class]] @ X_test[i])
                    Y_pred[i] = np.argmax(preds)

                # Test
                acc.append(zero_one_loss(Y_test, Y_pred))

    return acc

#

max_iters = 8000
pred_every = 40

#acc_v0 = perceptron_v0(max_iters, pred_every, X_train, X_test, Y_train, Y_test)
#print('v0 done')
#acc_v1 = perceptron_v1(max_iters, pred_every, X_train, X_test, Y_train, Y_test)
#print('v1 done')
#acc_v2 = perceptron_v2(max_iters, pred_every, X_train, X_test, Y_train, Y_test)
#print('v2 done')
#
#plt.figure()
#plt.plot(pred_every * np.array(range(len(acc_v0))), acc_v0, '.-')
#plt.plot(pred_every * np.array(range(len(acc_v1))), acc_v1, '.-')
#plt.plot(pred_every * np.array(range(len(acc_v2))), acc_v2, '.-')
#plt.legend(['v0', 'v1', 'v2'])
#plt.xlabel('Iteration'); plt.ylabel('Ratio misclassified');
#plt.show()