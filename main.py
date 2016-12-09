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
    word_count = {}
    tag_count = {}

    lines = [line.rstrip('\n') for line in open('train_small.wtag')]
    # lines = [line.rstrip('\n') for line in open('train.wtag')]

    for line in lines:
        w = ['*_*', '*_*'] # start
        w.extend(line.split(" "))
        if w[-1] == '._.':
            del w[-1] # remove 'period' from the end of the sentence
        w.append('SSS_SSS') # stop
        sentences.append(w)

    for sentence in sentences:
        w = []
        tag = []
        for word in sentence:
            a,b = word.split("_")
            w.append(a)
            tag.append(b)
        sentences_w.append(w)
        sentences_t.append(tag)

    print(sentences_w[0])
    print(sentences_t[0])

    # add a feature for each trigram seen in the training data
    features = {}
    v_index = 0
    bigrams_offset = 0
    for sentence_t in sentences_t:
        for i in range(len(sentence_t)-2):
            t_hash = hash((sentence_t[i], sentence_t[i+1], sentence_t[i+2])) #TODO: make all index backards (n,n-1,n-2)
            if t_hash not in features:
                features[t_hash] = v_index
                v_index += 1
    bigrams_offset = v_index
    print(bigrams_offset)

    # add a feature for each bigram seen in the training data
    for sentence_t in sentences_t:
        for i in range(len(sentence_t)-1):
            t_hash = hash((sentence_t[i], sentence_t[i + 1]))
            if t_hash not in features:
                features[t_hash] = v_index
                v_index += 1
    unigrams_offset = v_index
    print(unigrams_offset)

    for sentence_w, sentence_t in zip(sentences_w, sentences_t):
        for i in range(len(sentence_t)-1):
            t_hash = hash((sentence_w[i], sentence_t[i]))
            if t_hash not in features:
                features[t_hash] = v_index
                v_index += 1
    print(v_index)

    # for line in sentences:
    #     for word in line:
    #         w,tag = word.split("_")
    #         lower_w = str(w).lower()
    #         if lower_w in word_count:
    #             word_count[lower_w] += 1
    #         else:
    #             word_count[lower_w] = 1
    #         if tag in tag_count:
    #             tag_count[tag] += 1
    #         else:
    #             tag_count[tag] = 1
            # print(lower_w,tag)

    # print()
    # for tri in trigrams:
    #     print(tri)

    # capital_word = 'Word'
    # print('capital_word?: ', capital_word.islower())

    # print('number of uniqe words: ' , len(word_count))
    # print(tag_count)
    # print('number of uniqe tags: ' , len(tag_count))

    # for w in tag_count:
    #     print('the tag ', w, ' appears: ', tag_count[w], ' times')
    logging.info('Done!')
    print("Done!")

"""Run main"""
if __name__ == '__main__':
    main()
