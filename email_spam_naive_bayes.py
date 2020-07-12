# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 09:50:44 2020

@author: Sai
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Unpickle and load
with open('email_data_train_test_split.pickle', 'rb') as f:
    words_dict = pickle.load(f, encoding = 'latin-1')
    ham_emails = pickle.load(f, encoding = 'latin-1')
    spam_emails = pickle.load(f, encoding = 'latin-1')
    ham_vectors_train = pickle.load(f, encoding = 'latin-1')
    ham_vectors_test = pickle.load(f, encoding = 'latin-1')
    spam_vectors_train = pickle.load(f, encoding = 'latin-1')
    spam_vectors_test = pickle.load(f, encoding = 'latin-1')

max_counts = 20     # Maximum counts we keep track of. Any word that appears more than max_counts is counted as appearing as max_counts
pseudocount = 1     # For additive smoothing https://en.wikipedia.org/wiki/Additive_smoothing

def build_counts_dict(vectors_train, words_dict):
    # Build an array that contains the count rates for each word
    # Each element in the array corresponds to a word and contains a dictionary which has the count rates for that word
    counts_dict_arr = [0]*len(words_dict)

    for word in words_dict:
        idx = words_dict[word]

        counts_arr = []
        counts_dict = {}
        for email in vectors_train:
            counts_arr.append(email[idx])

        for count_val in set(counts_arr):
            counts_dict[count_val] = counts_arr.count(count_val)

        # Add all the counts above the max_counts and set the value to the max_counts
        counts_dict_keys = list(counts_dict.keys())
        num_counts_over_max = 0
        for key in counts_dict_keys:
            if key >= max_counts:
                num_counts_over_max += counts_dict[key]
                counts_dict.pop(key)
        counts_dict[max_counts] = num_counts_over_max

        counts_dict_arr[idx] = counts_dict

    return counts_dict_arr

ham_counts_dict_arr = build_counts_dict(ham_vectors_train, words_dict)
spam_counts_dict_arr = build_counts_dict(spam_vectors_train, words_dict)

# Make sure all words have been processed
assert not 0 in ham_counts_dict_arr
assert not 0 in spam_counts_dict_arr


# Plot some hammy and spammy words
hammy_words = ['hr', 'error']
spammy_words = ['money', 'bank']

fig, (ax1, ax2) = plt.subplots(1, 2)
colors = ['b', 'g', 'r', 'k']
for idx, word in enumerate(hammy_words):
    ham_x = list(ham_counts_dict_arr[words_dict[word]].keys())
    ham_y = [ham_counts_dict_arr[words_dict[word]][x] for x in ham_x[1:]]
    spam_x = list(spam_counts_dict_arr[words_dict[word]].keys())
    spam_y = [spam_counts_dict_arr[words_dict[word]][x] for x in spam_x[1:]]
    ax1.plot(ham_x[1:], np.log(ham_y), '*-', label = word + ' ham', color = colors[idx])
    ax1.plot(spam_x[1:], np.log(spam_y), '.--', label = word + ' spam', color = colors[idx])
ax1.set_title('Hammy words')
ax1.set_ylabel('Log counts of emails')
ax1.set_xlabel('Frequency in one email')
ax1.legend()

for idx, word in enumerate(spammy_words):
    ham_x = list(ham_counts_dict_arr[words_dict[word]].keys())
    ham_y = [ham_counts_dict_arr[words_dict[word]][x] for x in ham_x[1:]]
    spam_x = list(spam_counts_dict_arr[words_dict[word]].keys())
    spam_y = [spam_counts_dict_arr[words_dict[word]][x] for x in spam_x[1:]]
    ax2.plot(ham_x[1:], np.log(ham_y), '*-', label = word + ' ham', color = colors[idx])
    ax2.plot(spam_x[1:], np.log(spam_y), '.--', label = word + ' spam', color = colors[idx])
ax2.set_title('Spammy words')
ax2.set_ylabel('Log counts of emails')
ax2.set_xlabel('Frequency in one email')
ax2.legend()


# Naive Bayes classifier
def classify(test_vector, ham_counts_dict_arr, spam_counts_dict_arr, prior_ham, prior_spam, pseudocount):
    # Classifies an email and returns either 'ham' or 'spam'

    def get_freqs(test_vector, counts_dict_arr):
        # Helper function that returns the frequency of the count rate
        freqs = [0]*len(test_vector)
        for idx, word_count in enumerate(test_vector):
            if word_count > 0:  # Only consider words that appear in the email at least once for classification purposes
                if word_count in counts_dict_arr[idx]:
                    # At least one training email had exactly the counts of the test email
                    freqs[idx] = counts_dict_arr[idx][word_count] + pseudocount
                elif word_count > max_counts:
                    # Number of counts in the test email is greater than max_counts, so we use the result from max_counts
                    freqs[idx] = counts_dict_arr[idx][max_counts] + pseudocount
                else:
                    # No training email had exactly the counts in the test email, so we just use the pseudocount
                    freqs[idx] = pseudocount
        return freqs

    ham_freqs = get_freqs(test_vector, ham_counts_dict_arr)
    spam_freqs = get_freqs(test_vector, spam_counts_dict_arr)

    # Use log-probabilities because the numbers get very small
    p_ham = prior_ham
    p_spam = prior_spam

    for ham_freq in ham_freqs:
        if ham_freq > 0:
            p_ham_word = ham_freq/(len(ham_vectors_train) + max_counts*pseudocount)
            p_ham += np.log(p_ham_word)
    for spam_freq in spam_freqs:
        if spam_freq > 0:
            p_spam_word = spam_freq/(len(spam_vectors_train) + max_counts*pseudocount)
            p_spam += np.log(p_spam_word)

    return 'ham' if p_ham > p_spam else 'spam'


# Start classifying
def classify_all(ham_vectors_test, spam_vectors_test, ham_prior, spam_prior, pseudocount):
    ham_correct, ham_incorrect = 0, 0
    for ham_test in ham_vectors_test:
        ham_or_spam = classify(ham_test, ham_counts_dict_arr, spam_counts_dict_arr, ham_prior, spam_prior, pseudocount = 1)
        if ham_or_spam == 'ham':
            ham_correct += 1
        else:
            ham_incorrect += 1
    ham_correct_pct = 100 * ham_correct / len(ham_vectors_test)
    ham_incorrect_pct = 100 * ham_incorrect / len(ham_vectors_test)
    print('ham test set: correctly classified %.2f, incorrectly classified %.2f' %(ham_correct_pct, ham_incorrect_pct))

    spam_correct, spam_incorrect = 0, 0
    for spam_test in spam_vectors_test:
        ham_or_spam = classify(spam_test, ham_counts_dict_arr, spam_counts_dict_arr, ham_prior, spam_prior, pseudocount = 1)
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


# First choice of priors: ratio of the number of ham and spam emails
# This assumes that we know the percentage of ham and spam emails in the test set
# but we're drawing a random email everytime
ham_prior = np.log(len(ham_vectors_test) / (len(ham_vectors_test) + len(spam_vectors_test)))
spam_prior = np.log(len(spam_vectors_test) / (len(ham_vectors_test) + len(spam_vectors_test)))
print('Priors assuming random draw from test set')
classify_all(ham_vectors_test, spam_vectors_test, ham_prior, spam_prior, pseudocount = 1)

# Second choice: equal priors
print('Equal priors')
classify_all(ham_vectors_test, spam_vectors_test, 0, 0, pseudocount = 1)


# Third choice: skewed towards spam - real world scenario
print('Skewed towards spam')
classify_all(ham_vectors_test, spam_vectors_test, 0, 1, pseudocount = 1)


# See what happens with the test set
print('Train set with equal priors and no pseudocount')
classify_all(ham_vectors_train, spam_vectors_train, 0, 0, pseudocount = 0)