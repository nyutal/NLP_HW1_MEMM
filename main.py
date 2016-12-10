from __future__ import print_function
from builtins import range
import scipy as sp
import scipy.optimize
import numpy as np
import logging
from consts import CONST

logging.basicConfig(filename='hw1.log', filemode='w', level=logging.DEBUG)


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
    tags = set()

    v_index = prepare_features(sentences, sentences_t, sentences_w, features, tags)

    #weights = np.random.rand(v_index) / 1000 # make weights vector the same length as the feature vector
    # L = calc_L(weights, features, sentences_t, sentences_w, tags, v_index)
    # Lprime = calc_Lprime(weights, features, sentences_t, sentences_w, tags, v_index)
    # print(L, Lprime)

    x1, f1, d1 = sp.optimize.fmin_l_bfgs_b(calc_L,
                                           x0=np.zeros(v_index),
                                           args=(features, sentences_t, sentences_w, tags, v_index),
                                           fprime=calc_Lprime,
                                           pgtol=1e-3, disp=True)
    print('x1:', x1)
    print('f1:', f1)
    print('d1:', d1)

    # regularizer_Lprime = CONST.reg_lambda * weights[0] #TODO: [k]
    # print('regularizer_Lprime:', regularizer_Lprime)

    logging.info('Done!')
    print("Done!")


def calc_L(weights, features, sentences_t, sentences_w, tags, v_index):
    # v dot f of all sentences:
    s1 = sum([calc_v_dot_f(features, s_tags, s_words, weights)
              for s_words, s_tags in zip(sentences_w, sentences_t)])
    # print('s1:', s1)

    s2 = sum([calc_v_dot_f_for_tag_in_tags(features, s_tags, s_words, weights, tags)
              for s_words, s_tags in zip(sentences_w, sentences_t)])
    # print('s2:', s2)

    regularizer_L = (CONST.reg_lambda/2) * (np.linalg.norm(weights))**2
    # print('regularizer_L:', regularizer_L)

    return float(s1 - s2 - regularizer_L)

def calc_Lprime(weights, features, sentences_t, sentences_w, tags, v_index):
    # # v dot f of all sentences:
    # s1 = sum([calc_v_dot_f(features, s_tags, s_words, weights)
    #           for s_words, s_tags in zip(sentences_w, sentences_t)])
    # # print('s1:', s1)
    #
    # s2 = sum([calc_v_dot_f_for_tag_in_tags(features, s_tags, s_words, weights, tags)
    #           for s_words, s_tags in zip(sentences_w, sentences_t)])
    # # print('s2:', s2)
    #
    # regularizer_L = (CONST.reg_lambda/2) * (np.linalg.norm(weights))**2
    # # print('regularizer_L:', regularizer_L)

    # return s1 - s2 - regularizer_L
    return np.random.rand(v_index)

# TODO: should this be for one parameter at the time??? Fk???
# TODO: use one line sums: J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])
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

# TODO: is Xi a sentence or a "history" (1 word with its priors)?
def calc_v_dot_f_for_tag_in_tags(features, s_tags, s_words, weights, tags):
    v_dot_f = 0
    t = CONST.epsilon
    for tag in tags:
        # trigrams:
        for i in range(2,len(s_tags)): # start from word not *, look at the stop sign.
            t_hash = hash((s_tags[i-2], s_tags[i-1], tag))
            if t_hash in features:
                v_dot_f += weights[features[t_hash]]
        # bigrams:
        for i in range(2,len(s_tags)): # start from word not *, look at the stop sign.
            t_hash = hash((s_tags[i-1], tag))
            if t_hash in features:
                v_dot_f += weights[features[t_hash]]
        # word-tag (emission):
        for i in range(2,len(s_words)-1): # start from word not *, do not look at the stop sign.
            t_hash = hash((s_words[i], tag))
            if t_hash in features:
                v_dot_f += weights[features[t_hash]]
        t += np.exp(v_dot_f)
    return np.log(t)


def prepare_features(sentences, sentences_t, sentences_w, features, tags):
    lines = [line.rstrip('\n') for line in open(CONST.train_file_name)]
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
            tags.add(b) # create a set of all tags
        sentences_w.append(w)
        sentences_t.append(tag)
    tags.remove("SSS")
    tags.remove("*")

    print(sentences_w[0]) # just debug
    print(sentences_t[0]) # just debug
    print('all tags(', len(tags), '):', tags)
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
