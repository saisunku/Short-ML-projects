# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:50:33 2020

@author: Sai
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle, copy, queue

# Unpickle and load
with open('email_data_train_test_split_v1.pickle', 'rb') as f:
    words_dict = pickle.load(f, encoding = 'latin-1')
    ham_emails = pickle.load(f, encoding = 'latin-1')
    spam_emails = pickle.load(f, encoding = 'latin-1')
    ham_vectors_train = pickle.load(f, encoding = 'latin-1')
    ham_vectors_test = pickle.load(f, encoding = 'latin-1')
    spam_vectors_train = pickle.load(f, encoding = 'latin-1')
    spam_vectors_test = pickle.load(f, encoding = 'latin-1')

vectors_train = ham_vectors_train + spam_vectors_train
labels_train = [0]*len(ham_vectors_train) + [1]*len(spam_vectors_train)


# Need to pick a few features for decision tree - pick the ones with the highest inter-class variance


def build_counts_dict(vectors_train, words_dict):
    # Build an array that contains the count rates for each word
    # Each element in the array corresponds to a word and contains a dictionary which has the count rates for that word
    # Similar to the function used in the naive Bayes model but there is no maximum count here
    counts_dict_arr = [0]*len(words_dict)

    for word in words_dict:
        idx = words_dict[word]

        counts_arr = []
        counts_dict = {}
        for email in vectors_train:
            counts_arr.append(email[idx])

        for count_val in set(counts_arr):
            counts_dict[count_val] = counts_arr.count(count_val)

        counts_dict_arr[idx] = counts_dict

    return counts_dict_arr

ham_counts_dict_arr = build_counts_dict(ham_vectors_train, words_dict)
spam_counts_dict_arr = build_counts_dict(spam_vectors_train, words_dict)


def get_variance(word, ham_counts_dict_arr, spam_counts_dict_arr, pseudocount):
    # Returns the inter-class variance for a given feature
    idx = words_dict[word]

    ham_counts = 0
    ham_emails_with_word = 0
    for count in ham_counts_dict_arr[idx]:
        ham_counts += (count + pseudocount) * ham_counts_dict_arr[idx][count]
        ham_emails_with_word += ham_counts_dict_arr[idx][count]
    ham_mean = ham_counts / ham_emails_with_word

    spam_counts = 0
    spam_emails_with_word = 0
    for count in spam_counts_dict_arr[idx]:
        spam_counts += (count + pseudocount) * spam_counts_dict_arr[idx][count]
        spam_emails_with_word += spam_counts_dict_arr[idx][count]
    spam_mean = spam_counts / spam_emails_with_word

    return abs(ham_mean - spam_mean) / (ham_mean + spam_mean)

# Make a list of variances
variances = [0]*len(words_dict)
for word in words_dict:
    idx = words_dict[word]
    variances[idx] = get_variance(word, ham_counts_dict_arr, spam_counts_dict_arr, pseudocount = 1)

# Print some values
print('Max variance for a feature is %.3f and the word is %s' %(max(variances), list(words_dict.keys())[list(words_dict.values()).index(variances.index(max(variances)))]))
# Histogram of variances
plt.figure()
plt.hist(variances, log=True)
plt.xlabel('Variance'); plt.ylabel('Counts')
plt.show()


# Pick features based on a variance threshold
variance_thresh = 0.005
features_for_tree = [i for i in range(len(variances)) if variances[i] > variance_thresh]

# Returns the uncertainty used to decide the features of the decision tree
def uncert(idxs):
    # Use Shannon entropy for now
    if len(idxs) == 0:
        return 0

    labels = [labels_train[idx] for idx in idxs]
    assert labels.count(0) + labels.count(1) == len(labels)   # Make sure there are only 0s and 1s
    p_ham = labels.count(0) / len(labels)
    p_spam = labels.count(1) / len(labels)

    entropy_ham = p_ham * np.log(1 / p_ham) if p_ham > 0 else 0
    entropy_spam = p_spam * np.log(1 / p_spam) if p_spam > 0 else 0
    return entropy_ham + entropy_spam


def find_best_thresh(idxs, features):
    # Checks all the features and thresholds to find the feature, threshold combination
    # that maximizes the reduction in uncertainty
    if len(idxs) == 0:
        return 0, 0, [], []

    max_change_uncert = float('-inf')
    uncert_start = uncert(idxs)

    for feature in features:
        max_feature_val = max(list(ham_counts_dict_arr[feature].keys()) +  list(spam_counts_dict_arr[feature].keys()))
        for thresh in range(max_feature_val):   # Only consider integer thresholds - valid because our data is all integers
            left_idxs, right_idxs = [], []
            for idx in idxs:
                if vectors_train[idx][feature] <= thresh:
                    left_idxs.append(idx)
                else:
                    right_idxs.append(idx)

            change_uncert = uncert_start - (len(left_idxs) / len(idxs)) * uncert(left_idxs) \
                - (len(right_idxs) / len(idxs)) * uncert(right_idxs)
            if change_uncert > max_change_uncert:
                max_change_uncert = change_uncert
                opt_thresh = thresh
                opt_feature = feature
                opt_left_idxs = left_idxs
                opt_right_idxs = right_idxs

    print(opt_feature, opt_thresh)
    return opt_feature, opt_thresh, opt_left_idxs, opt_right_idxs


# Decision tree node class
class decision_tree_node():
    def __init__(self, opt_feature, opt_thresh, idxs, left_node = None, right_node = None):
        self.opt_feature = opt_feature
        self.opt_thresh = opt_thresh
        self.idxs = idxs

        self.left = left_node
        self.right = right_node


# Train the decision tree
root_node = decision_tree_node(None, None, [i for i in range(len(vectors_train))])
max_nodes = 2**12

# Process the nodes in a BFS fashion - so use a queue
nodes_to_process = queue.Queue()
nodes_to_process.put(root_node)

for num_node in range(max_nodes):
    cur_node = nodes_to_process.get()

    # Find the best split for the current node and add the optimal feature and threshold values to the node
    opt_feature, opt_thresh, left_idxs, right_idxs = \
        find_best_thresh(cur_node.idxs, features_for_tree)
    cur_node.opt_feature = opt_feature
    cur_node.opt_thresh = opt_thresh

    # Make left and right nodes and connect them to the current node
    left_node = decision_tree_node(None, None, left_idxs)
    right_node = decision_tree_node(None, None, right_idxs)

    cur_node.left = left_node
    cur_node.right = right_node

    nodes_to_process.put(left_node)
    nodes_to_process.put(right_node)


# Classify
def classify(test_vector, root_node):
    cur_node = root_node
    while len(cur_node.idxs) > 1 and cur_node.left is not None:
        if test_vector[cur_node.opt_feature] <= cur_node.opt_thresh:
            cur_node = cur_node.left
        else:
            cur_node = cur_node.right

    labels = [labels_train[idx] for idx in cur_node.idxs]

    return 'ham' if labels.count(0) > labels.count(1) else 'spam'


# Classify all emails in the test set and keep track of accuracy
def classify_all(ham_vectors_test, spam_vectors_test):
    ham_correct, ham_incorrect = 0, 0
    for ham_test in ham_vectors_test:
        ham_or_spam = classify(ham_test, root_node)
        if ham_or_spam == 'ham':
            ham_correct += 1
        else:
            ham_incorrect += 1
    ham_correct_pct = 100 * ham_correct / len(ham_vectors_test)
    ham_incorrect_pct = 100 * ham_incorrect / len(ham_vectors_test)
    print('ham test set: correctly classified %.2f, incorrectly classified %.2f' %(ham_correct_pct, ham_incorrect_pct))

    spam_correct, spam_incorrect = 0, 0
    for spam_test in spam_vectors_test:
        ham_or_spam = classify(spam_test, root_node)
        if ham_or_spam == 'spam':
            spam_correct += 1
        else:
            spam_incorrect += 1
    spam_correct_pct = 100 * spam_correct / len(spam_vectors_test)
    spam_incorrect_pct = 100 * spam_incorrect / len(spam_vectors_test)
    print('spam test set: correctly classified %.2f, incorrectly classified %.2f' %(spam_correct_pct, spam_incorrect_pct))

    overall_acc = 100 * (ham_correct + spam_correct) / (len(ham_vectors_test) + len(spam_vectors_test))
    print('overall accuracy rate: %.2f' %(overall_acc))

    return overall_acc, ham_correct_pct, spam_correct_pct


print('Test set')
classify_all(ham_vectors_test, spam_vectors_test)

print('Training set')
classify_all(ham_vectors_train, spam_vectors_train)