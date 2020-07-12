# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:09:29 2020

@author: Sai
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pickle, random, os, gc
import nltk

# Word stemmer
sno = nltk.stem.SnowballStemmer('english')

# Parameters
ham_folder = r'C:\Users\Sai\Documents\SpiderOak Hive\Career\CS4771\Homeworks\HW1\email_data\ham'
spam_folder = r'C:\Users\Sai\Documents\SpiderOak Hive\Career\CS4771\Homeworks\HW1\email_data\spam'
test_split = 0.2

def tokenize(folder):
    # Returns a list of lists which contains the tokens for each email
    files = list(os.walk(folder))[0][2]

    emails = []
    for fname in files:
        filename = os.path.join(folder, fname)

        tokens = []
        with open(filename, encoding = 'latin-1') as f:
            for line in f:
                tkns = nltk.word_tokenize(line)
                for tkn in tkns:
                    # Don't include tokens that are not alphanumeric or contain just numbers or have a single character
                    if tkn.isalnum() and not tkn.isnumeric() and len(tkn) > 1:
                        tokens.append(sno.stem(tkn))

        emails.append(tokens)
    return emails


def process_input(ham_folder, spam_folder):
    # Tokenize all emails
    print('Tokenizing...')
    ham_emails = tokenize(ham_folder)
    spam_emails = tokenize(spam_folder)


    # Build dictionary of words
    print('Building dictionary...')
    words_dict = {}
    idx = 0
    for ham_email in ham_emails:
        for word in ham_email:
            if word not in words_dict:
                words_dict[word] = idx
                idx += 1
    for spam_email in spam_emails:
        for word in spam_email:
            if word not in words_dict:
                words_dict[word] = idx
                idx += 1


    # Build the bag of words vectors
    # Do a train-test split by using random.randint
    print('Building ham vectors...')
    ham_vectors_train = []
    ham_vectors_test = []

    for ham_email in ham_emails:
        ham_vector = [0]*len(words_dict)
        for word in words_dict:
            ham_vector[words_dict[word]] = ham_email.count(word)

        if random.randint(0, int(1 / test_split - 1)) == 0:
            ham_vectors_test.append(ham_vector)
        else:
            ham_vectors_train.append(ham_vector)
        print('.', end = '')
    print(' ')


    print('Building spam vectors...')
    spam_vectors_train = []
    spam_vectors_test = []

    for spam_email in spam_emails:
        spam_vector = [0]*len(words_dict)
        for word in words_dict:
            spam_vector[words_dict[word]] = spam_email.count(word)

        if random.randint(0, int(1 / test_split - 1)) == 0:
            spam_vectors_test.append(spam_vector)
        else:
            spam_vectors_train.append(spam_vector)
        print('.', end = '')
    print(' ')

    print('Done!')
    return words_dict, ham_emails, spam_emails, ham_vectors_train, ham_vectors_test, spam_vectors_train, spam_vectors_test


words_dict, ham_emails, spam_emails, ham_vectors_train, ham_vectors_test, spam_vectors_train, spam_vectors_test = process_input(ham_folder, spam_folder)

print('Starting pickling...')
with open('email_data_train_test_split.pickle', 'wb') as f:
    pickle.dump(words_dict, f)
    pickle.dump(ham_emails, f)
    pickle.dump(spam_emails, f)
    pickle.dump(ham_vectors_train, f)
    pickle.dump(ham_vectors_test, f)
    pickle.dump(spam_vectors_train, f)
    pickle.dump(spam_vectors_test, f)
print('Pickling done!')