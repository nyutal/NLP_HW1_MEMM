from __future__ import print_function
from builtins import range
from future.utils import iteritems
import numpy as np
import logging
from consts import CONST

logging.basicConfig(filename='hw1.log', filemode='w', level=logging.DEBUG)

train_file_name = 'train_small.wtag'
# train_file_name = 'train.wtag'

# put something into class?
class MyClass(object):
    epsilon = 10 ** (-4)

    def __init__(self, nid):
        self.enabled = True

    def __lt__(self, other):
        if type(self) is str(type(other)):
            return self.nid < other.nid
        else:
            return str(type(self)) < str(type(other))

    def reset(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


def main():
    """Compare runs for various test graph initializations"""

    sentences = []
    sentences_w = []
    sentences_t = []
    features = {}

    v_index = prepare_features(sentences, sentences_t, sentences_w, features)

    weights = np.random.rand(v_index, 1) # make weights vector the same length as the feature vector

    # v dot f of all sentences:
    for s_words, s_tags in zip(sentences_w, sentences_t):
        v_dot_f = calc_v_dot_f(features, s_tags, s_words, weights)
        print(v_dot_f)


    logging.info('Done!')
    print("Done!")


def calc_v_dot_f(features, s_tags, s_words, weights):
    v_dot_f = 0
    # trigrams:
    for i in range(2,len(s_tags)): # start from word not *, look at the stop sign.
        t_hash = hash((s_tags[i-2], s_tags[i-1], s_tags[i]))
        v_dot_f += weights[features[t_hash]]
    # bigrams:
    for i in range(2,len(s_tags)): # start from word not *, look at the stop sign.
        t_hash = hash((s_tags[i-1], s_tags[i]))
        v_dot_f += weights[features[t_hash]]
    # word-tag (emission):
    for i in range(2,len(s_tags)-1): # start from word not *, do not look at the stop sign.
        t_hash = hash((s_words[i], s_tags[i]))
        v_dot_f += weights[features[t_hash]]
    return v_dot_f


def prepare_features(sentences, sentences_t, sentences_w, features):
    lines = [line.rstrip('\n') for line in open(train_file_name)]
    for line in lines:
        w = ['*_*', '*_*']  # start
        w.extend(line.split(" "))
        if w[-1] == '._.':
            del w[-1]  # remove 'period' from the end of the sentence
        w.append('SSS_SSS')  # stop
        sentences.append(w)
    for sentence in sentences:
        w = []
        tag = []
        for word in sentence:
            a, b = word.split("_")
            w.append(a)
            tag.append(b)
        sentences_w.append(w)
        sentences_t.append(tag)
    print(sentences_w[0]) # just debug
    print(sentences_t[0]) # just debug

    v_index = 0

    # F103: add a feature for each trigram seen in the training data
    for sentence_t in sentences_t:
        for i in range(2,len(sentence_t)): # start from word not *, look at the stop sign
            t_hash = hash((sentence_t[i-2], sentence_t[i-1], sentence_t[i]))
            if t_hash not in features:
                features[t_hash] = v_index
                v_index += 1
    print(v_index)

    # F104: add a feature for each bigram seen in the training data
    for sentence_t in sentences_t:
        for i in range(2,len(sentence_t)): # start from word not *, look at the stop sign
            t_hash = hash((sentence_t[i-1], sentence_t[i]))
            if t_hash not in features:
                features[t_hash] = v_index
                v_index += 1
    print(v_index)

    # F100: word-tag pairs (emission)
    for sentence_w, sentence_t in zip(sentences_w, sentences_t):
        for i in range(2, len(sentence_t)-1): # start from word not *, do not look at the stop sign.
            t_hash = hash((sentence_w[i], sentence_t[i]))
            if t_hash not in features:
                features[t_hash] = v_index
                v_index += 1
    print(v_index)
    return v_index

"""Run main"""
if __name__ == '__main__':
    main()
