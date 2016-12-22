from __future__ import print_function
from builtins import range
import scipy as sp
import scipy.optimize
import numpy as np
import logging
from consts import CONST
from featureFunc import *
import time
from memmChecker import *
from sentenceParser import *
import multiprocessing as mp

logging.basicConfig(filename='hw1.log', filemode='w', level=logging.DEBUG)


def main():
    """Compare runs for various test graph initializations"""

    np.seterr(all='raise')

    train = False

    fv = FeatureVec()
    fv.addFeatureGen(F100())
    fv.addFeatureGen(F101_2())
    fv.addFeatureGen(F101_3())
    fv.addFeatureGen(F101_4())
    fv.addFeatureGen(F102_2())
    fv.addFeatureGen(F102_3())
    fv.addFeatureGen(F102_4())
    fv.addFeatureGen(F103())
    fv.addFeatureGen(F104())
    fv.addFeatureGen(F105())
    fv.addFeatureGen(FCapital())

    parser = SentenceParser()
    trainCorpus = parser.parseTagedFile(CONST.train_file_name)
    fv.generateFeatures(trainCorpus)

    # trainC2 = parser.parseTagedFile(CONST.train_file_name, 20)

    validateCorpus = parser.parseTagedFile(CONST.test_file_name)
    print(validateCorpus.getTags().issubset(trainCorpus.getTags()))

    if train:
        print('start optimization', time.asctime())
        x1, f1, d1 = sp.optimize.fmin_l_bfgs_b(calc_L,
                                               x0=np.ones(fv.getSize()),
                                               args=(fv,),
                                               # maxfun=8
                                               m=60,
                                               maxiter=30,
                                               disp=True)#, factr=CONST.accuracy['high'])

        # x1 = x1 * 10 ** 15  # in order to eliminate underflow
        print('x1:', x1)
        print('f1:', f1)
        print('d1:', d1)
    else:
        x1 = np.asarray(list(map(float,[line.strip() for line in open('results/test32.txt')])))
    print('x1:', x1)
    fv.setWeights(x1)

    checker = MemmChecker()

    # checker.check(fv, trainC2)

    checker.check(fv, validateCorpus)

    fp = open('test.txt', 'w')
    for i in x1:
        fp.write("%s\n" % i)
    logging.info('Done!')
    print("Done!")


def calc_L(weights, fv):
    sentences_w = fv.corpus.getSentencesW()
    sentences_t = fv.corpus.getSentencesT()

    #function  calculation vartiables
    funcReg = (CONST.reg_lambda / 2) * (np.linalg.norm(weights)) ** 2

    #gradiant calculation variables
    empirical = fv.getEmpirical()

    gradReg = CONST.reg_lambda * weights


    # #empirical calculation (for each k eiddccfiunujvnggdvbifrkjidglfvrrkdedjlbcugdr
    # for k in range(fv.getSize()):
    #     empirical[k] = fv.featureIdx2Fg[k].getCountsByIdx(k)
    w = []
    t = []
    j = []
    num_of_proc = 7
    sen_per_proc = int(len(fv.corpus.getSentences()) / num_of_proc)
    for i in range(num_of_proc):
        w.append(sentences_w[sen_per_proc*i : sen_per_proc*(i+1)])
        t.append(sentences_t[sen_per_proc*i : sen_per_proc*(i+1)])

    for i in range(num_of_proc):
        j.append((fv, t[i], w[i], weights))

    with mp.Pool() as pool:
        ret = pool.map(par_calc, j)
    s1, s2, expected = [sum(x) for x in zip(*ret)]

    gradVec = empirical - expected - gradReg
    g = -gradVec #maximize function

    funcRes = s1 - s2 - funcReg
    f = -funcRes #maximize function

    print('finish L', str(f), time.asctime())
    file_name = 'results/test' + str(fv.getIter()) + '.txt'
    fp = open(file_name, 'w')
    for i in weights:
        fp.write("%s\n" % i)
    return (f, g)


def par_calc(params):
    fv, sentences_t, sentences_w, weights = params
    tags = fv.corpus.getTags()
    fgArr = fv.fgArr
    s1 = 0.0
    s2 = 0.0
    expected = np.zeros(fv.getSize())
    for w, t in zip(sentences_w, sentences_t):
        for i in range(2, len(t)):
            # c += 1
            # if c % 10000 == 0: print('L sample ', c, time.asctime())

            s2TagPreLogExp = {}
            expectedTagsCalc = {}
            expectedDenominator = 0.0

            for tag in tags:

                # s2 prellog  zerois
                s2TagPreLogExp[tag] = 0.0

                # expected have constant coeficcient per sample and tag, calculate it here ( exp(v*f(t) / sum tags(exp())
                expectedTagsCalc[tag] = 0.0
                for fg in fgArr:
                    idx = fg.getFeatureIdx(w, tag, t[i - 1], t[i - 2], i)
                    if idx != -1:
                        expectedTagsCalc[tag] += weights[idx]
                expectedTagsCalc[tag] = np.exp(expectedTagsCalc[tag])
                expectedDenominator += expectedTagsCalc[tag]

            for fg in fgArr:

                # s1 is simply the sum of active features weights
                idx = fg.getFeatureIdx(w, t[i], t[i - 1], t[i - 2], i)
                if idx != -1:
                    s1 += weights[idx]

                for tag in tags:
                    jdx = fg.getFeatureIdx(w, tag, t[i - 1], t[i - 2], i)
                    if jdx != -1:
                        # expected calculation
                        expected[jdx] += expectedTagsCalc[tag] / expectedDenominator

                        # s2 calculation
                        s2TagPreLogExp[tag] += weights[jdx]

            # s2 cqlculqtion
            tagValsMul100 = np.asarray(list(s2TagPreLogExp.values()))
            loged = sp.misc.logsumexp(tagValsMul100)
            s2 += loged
    return s1, s2, expected


"""Run main"""
if __name__ == '__main__':
    main()
