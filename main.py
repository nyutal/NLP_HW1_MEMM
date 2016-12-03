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
    word_count = {}
    tag_count = {}

    # lines = [line.rstrip('\n') for line in open('train_small.wtag')]
    lines = [line.rstrip('\n') for line in open('train.wtag')]

    for line in lines:
        w = line.split(" ")
        if w[-1] == '._.':
            del w[-1]
        sentences.append(w)

    for line in sentences:
        print(line)
        for word in line:
            w,tag = word.split("_")
            lower_w = str(w).lower()
            if lower_w in word_count:
                word_count[lower_w] += 1
            else:
                word_count[lower_w] = 1
            if tag in tag_count:
                tag_count[tag] += 1
            else:
                tag_count[tag] = 1
            # print(lower_w,tag)

    print()
    print('the word \'the\' appears: ', word_count['the'], ' times')
    print('number of uniqe words: ' , len(word_count))
    print(tag_count)
    print('number of uniqe tags: ' , len(tag_count))



    logging.info('Done!')
    print("Done!")



"""Run main"""
if __name__ == '__main__':
    main()
