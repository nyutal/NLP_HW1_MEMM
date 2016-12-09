from __future__ import print_function
from builtins import range
from future.utils import iteritems
import numpy as np
import logging
from consts import CONST

logging.basicConfig(filename='hw1.log', filemode='w', level=logging.DEBUG)

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

    trigrams_offset, bigrams_offset, unigrams_offset, v_index = \
        prepare_features(sentences, sentences_t, sentences_w, features)

    weights = np.random.rand(v_index, 1) # make weights vector the same length as the feature vector

    # v dot f of one example sentence:
    s_words = sentences_w[0]
    s_tags = sentences_t[0]
    v_dot_f = 0
    # trigrams:
    for i in range(len(s_tags) - 2):# TODO: do we need to look at the stop sign?
        t_hash = hash((s_tags[i], s_tags[i + 1], s_tags[i + 2])) # TODO: make all index backards (n,n-1,n-2)
        # print((s_tags[i], s_tags[i + 1], s_tags[i + 2]), features[t_hash])
        v_dot_f += weights[features[t_hash]]
    # bigrams:
    for i in range(len(s_tags) - 1):# TODO: do we need to look at the stop sign?
        t_hash = hash((s_tags[i], s_tags[i + 1])) # TODO: make all index backards (n,n-1,n-2)
        # print((s_tags[i], s_tags[i + 1]), features[t_hash])
        v_dot_f += weights[features[t_hash]]
    # unigrams:
    for i in range(len(s_tags)):# TODO: do we need to look at the stop sign?
        t_hash = hash((s_words[i], s_tags[i])) # TODO: make all index backards (n,n-1,n-2)
        # print((s_tags[i], w_tags[i]), features[t_hash])
        v_dot_f += weights[features[t_hash]]
    print(v_dot_f)


    logging.info('Done!')
    print("Done!")


def prepare_features(sentences, sentences_t, sentences_w, features):
    lines = [line.rstrip('\n') for line in open('train_small.wtag')]
    # lines = [line.rstrip('\n') for line in open('train.wtag')]
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
    print(sentences_w[0])
    print(sentences_t[0])

    # add a feature for each trigram seen in the training data
    v_index = 0
    trigrams_offset = 0
    for sentence_t in sentences_t:
        for i in range(len(sentence_t) - 2):# TODO: do we need to look at the stop sign?
            t_hash = hash((sentence_t[i], sentence_t[i + 1], sentence_t[i + 2]))  # TODO: make all index backards (n,n-1,n-2)
            if t_hash not in features:
                features[t_hash] = v_index
                v_index += 1
    bigrams_offset = v_index
    print(bigrams_offset)
    # add a feature for each bigram seen in the training data
    for sentence_t in sentences_t:
        for i in range(len(sentence_t) - 1):# TODO: do we need to look at the stop sign?
            t_hash = hash((sentence_t[i], sentence_t[i + 1]))
            if t_hash not in features:
                features[t_hash] = v_index
                v_index += 1
    unigrams_offset = v_index
    print(unigrams_offset)
    for sentence_w, sentence_t in zip(sentences_w, sentences_t):
        for i in range(len(sentence_t)):# TODO: do we need to look at the stop sign?
            t_hash = hash((sentence_w[i], sentence_t[i]))
            if t_hash not in features:
                features[t_hash] = v_index
                v_index += 1
    print(v_index)
    return trigrams_offset, bigrams_offset, unigrams_offset, v_index

"""Run main"""
if __name__ == '__main__':
    main()
