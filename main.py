from __future__ import print_function
from builtins import range
import scipy as sp
import scipy.optimize
import numpy as np
import logging
from consts import CONST
from featureFunc import *
import time
from viterbi import *
from memmChecker import *
from sentenceParser import *

logging.basicConfig(filename='hw1.log', filemode='w', level=logging.DEBUG)


def main():
    """Compare runs for various test graph initializations"""

    np.seterr(all='raise')

    fv = FeatureVec()
    fv.addFeatureGen(F100())
    fv.addFeatureGen(F101_2())
    fv.addFeatureGen(F101_3())
    fv.addFeatureGen(F102_2())
    fv.addFeatureGen(F102_3())
    fv.addFeatureGen(F103())
    fv.addFeatureGen(F104())
    fv.addFeatureGen(F105())

    parser = SentenceParser()
    testCorpus = parser.parseTagedFile(CONST.train_file_name, 1000)
    fv.generateFeatures(testCorpus)


    validateCorpus = parser.parseTagedFile("test.Wtag", 1)
    print(validateCorpus.getTags().issubset(testCorpus.getTags()))

    print('start optimization', time.asctime())
    x1, f1, d1 = sp.optimize.fmin_l_bfgs_b(calc_L,
                                           x0=np.full(fv.getSize(), CONST.epsilon),
                                           args=(fv,),
                                           fprime=calc_Lprime,  # m=256, maxfun=8, maxiter=8,
                                           disp=True)#, factr=CONST.accuracy['high'])

    print('x1:', x1)
    print('f1:', f1)
    print('d1:', d1)

    x1 = x1 * 10 ** 12  # in order to eliminate underflow

    fv.setWeights(x1)

    checker = MemmChecker()

    checker.check(fv, validateCorpus)



    fp = open('test.txt', 'w')
    for i in x1:
        fp.write("%s\n" % i)
    logging.info('Done!')
    print("Done!")


def calc_L(weights, fv):
    #     print('start L')
    c = 0
    s1 = 0.0
    s2 = 0.0

    sentences_w = fv.corpus.getSentencesW()
    sentences_t = fv.corpus.getSentencesT()
    tags = fv.corpus.getTags()
    fgArr = fv.fgArr

    for w, t in zip(sentences_w, sentences_t):
        for i in range(2, len(t)):
            c += 1
            if c % 10000 == 0: print('L sample ', c, time.asctime())
            tagPreLogExp = {}
            for tag in tags:
                tagPreLogExp[tag] = 0.0
            for fg in fgArr:
                idx = fg.getFeatureIdx(w, t[i], t[i - 1], t[i - 2], i)
                if idx != -1:
                    s1 += weights[idx]
                for tag in tags:
                    idx = fg.getFeatureIdx(w, tag, t[i - 1], t[i - 2], i)
                    if idx != -1:
                        try:
                            prevlogregex = tagPreLogExp[tag]
                            tagPreLogExp[tag] += weights[idx]
                        except RuntimeError:
                            print('oops', weights[idx], prevlogregex)
                            exit()
            tagValsMul100 = 100.0 * np.asarray(list(tagPreLogExp.values()))
            loged = sp.misc.logsumexp(tagValsMul100)
            s2 += loged - np.log(100)

    regularizer_L = (CONST.reg_lambda / 2) * (np.linalg.norm(weights)) ** 2
    retVal = -float(s1 - s2 - regularizer_L)
    print('finish L', str(retVal), time.asctime())
    return retVal


# v dot f of all sentences: one parameter at the time:
def calc_Lprime(weights, fv):
    sentences_w = fv.corpus.getSentencesW()
    sentences_t = fv.corpus.getSentencesT()
    tags = fv.corpus.getTags()
    fgArr = fv.fgArr

    c = 0
    empirical = np.zeros(fv.getSize())
    # empirical = np.zeros(fv.getSize())
    for k in range(fv.getSize()):
        empirical[k] = fv.featureIdx2Fg[k].getCountsByIdx(k)

    expected = np.zeros(fv.getSize())
    for w, t in zip(sentences_w, sentences_t):
        for i in range(2, len(t)):
            c += 1
            if c % 10000 == 0: print('LPrime sample ', c, time.asctime())
            tagsCalc = {}
            denominator = 0.0

            for tag in tags:
                tagsCalc[tag] = 0.0
                for fg in fgArr:
                    idx = fg.getFeatureIdx(w, tag, t[i - 1], t[i - 2], i)
                    if idx != -1:
                        tagsCalc[tag] += weights[idx]
                tagsCalc[tag] = np.exp(tagsCalc[tag])
                denominator += tagsCalc[tag]
            for tag in tags:
                for fg in fgArr:
                    k = fg.getFeatureIdx(w, tag, t[i - 1], t[i - 2], i)
                    if k != -1:
                        expected[k] += tagsCalc[tag] / denominator

    regulaized_LP = CONST.reg_lambda * weights

    lprimeVec = empirical - expected - regulaized_LP
    retVal = -lprimeVec
    print('finished LPrime', time.asctime())
    return retVal


# def prepare_features(sentences, sentences_t, sentences_w, tags, fv, fgArr):
#
#     parser = SentenceParser()
#     parser.parseTagedFile(sentences, sentences_t, sentences_w, tags,CONST.train_file_name)
#
#     for w, t in zip(sentences_w, sentences_t):
#         for i in range(2, len(t)):
#             for fg in fgArr:
#                 fg.addFeature(fv, w, t[i], t[i-1], t[i-2], i)
#     print('fv contains ', fv.getSize(), ' features')

"""Run main"""
if __name__ == '__main__':
    main()
