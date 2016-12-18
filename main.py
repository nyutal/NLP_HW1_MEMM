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
from SentenceParser import *

logging.basicConfig(filename='hw1.log', filemode='w', level=logging.DEBUG)


def main():
    """Compare runs for various test graph initializations"""

    np.seterr(all='raise')

    sentences = []
    sentences_w = []
    sentences_t = []
    tags = set()

    fgArr = []
    fgArr.append(F100())
    fgArr.append(F101_2())
    fgArr.append(F101_3())
    fgArr.append(F102_2())
    fgArr.append(F102_3())
    fgArr.append(F103())
    fgArr.append(F104())
    fgArr.append(F105())
    fv = FeatureVec(fgArr)

    # print('Accuracy =',CONST.epsilon * CONST.accuracy['low'])

    prepare_features(sentences, sentences_t, sentences_w, tags, fv, fgArr)

    vSentences = []
    vSentences_w = []
    vSentences_t = []
    vTags = set()
    parser = SentenceParser()
    parser.parseTagedFile(vSentences, vSentences_t, vSentences_w, vTags, "test.Wtag")
    print(vTags.issubset(tags))

    print('start optimization', time.asctime())
    x1, f1, d1 = sp.optimize.fmin_l_bfgs_b(calc_L,
                                           x0=np.full(fv.getSize(), CONST.epsilon),
                                           args=(fgArr, sentences_t, sentences_w, tags, fv),
                                           fprime=calc_Lprime, m=256, maxfun=8, maxiter=8, disp=True, factr=CONST.accuracy['high'])



    print('x1:', x1)
    print('f1:', f1)
    print('d1:', d1)

    x1 = x1 * 10**12 # in order to eliminate underflow

    fv.setWeights(x1)



    v = Viterbi(tags, fv, x1)


    totalTags = 0
    totalErrors = 0
    for i in range(len(vSentences)):
        print('sample ', str(i), ':')
        tags = v.solve(vSentences_w[i])
        if len(tags) != len(vSentences_w[i])-2:
            print(len(tags))
            print(tags)
            print(len(vSentences_w[i]))
            print(vSentences_w[i])
            exit()
        for j in range(2, len(vSentences_t[i])-1):
            totalTags += 1
            if ( vSentences_t[i][j] != tags[j-2] ):
                totalErrors += 1
                print('Error:' , vSentences_t[i][j],tags[j-2])

    print('TotalTags: ', totalTags, 'Total Errors: ', totalErrors, 'Precision: ', float(totalTags-totalErrors)/totalTags)



    fp = open('test.txt', 'w')
    for i in x1:
        fp.write("%s\n" % i)
    logging.info('Done!')
    print("Done!")


def calc_L(weights, fgArr, sentences_t, sentences_w, tags, fv):
#     print('start L')
    c = 0
    s1 = 0.0
    s2 = 0.0
    for w, t in zip(sentences_w, sentences_t):
        for i in range(2, len(t)):
            c += 1
            if c % 10000 == 0: print('L sample ', c, time.asctime())
            tagPreLogExp = {}
            for tag in tags:
                tagPreLogExp[tag] = 0.0
            for fg in fgArr:
                idx = fg.getFeatureIdx(w, t[i], t[i-1], t[i-2], i)
                if idx != -1:
                    s1 += weights[idx] 
                for tag in tags:
                    idx = fg.getFeatureIdx(w, tag, t[i-1], t[i-2], i)
                    if idx != -1:
                        try:
                            prevlogregex = tagPreLogExp[tag]
                            tagPreLogExp[tag] += weights[idx]
                        except RuntimeError:
                            print('oops', weights[idx], prevlogregex)
                            exit()
            # if prelogexp < (10**-10): prelogexp = (10**-10) # remove zeros from log
            # preLog = 0.0
            # for tag in tags:
            #     print('tag', tag, tagPreLogExp[tag])
            #     preLog += np.exp(tagPreLogExp[tag])
            # s2 += np.log(preLog)
            # print(tagVals)
            tagValsMul100 = 100.0 * np.asarray(list(tagPreLogExp.values()))
            loged = sp.misc.logsumexp(tagValsMul100)
            s2 += loged - np.log(100)

    
    regularizer_L = (CONST.reg_lambda/2) * (np.linalg.norm(weights))**2
    retVal = -float(s1 - s2 - regularizer_L)
    print('finish L', str(retVal), time.asctime())
    return retVal

# v dot f of all sentences: one parameter at the time:
def calc_Lprime(weights, fgArr, sentences_t, sentences_w, tags, fv):
    c = 0
    empirical = np.zeros(fv.getSize())
    # empirical = np.zeros(fv.getSize())
    for k in range(fv.getSize()):
        empirical[k] = fv.featureIdx2Fg[k].getCountsByIdx(k)

    expected = np.zeros(fv.getSize())
    for w, t in zip(sentences_w, sentences_t):
        for i in range(2, len(t)):
            c += 1
            if c % 10000 == 0 : print('LPrime sample ', c, time.asctime())
            tagsCalc = {}
            denominator = 0.0

            # for fg in fgArr:
            #     idx = fg.getFeatureIdx(w, t[i], t[i - 1], t[i - 2], i)
            #     if idx != -1:
            #         empirical[idx] += 1
            # print('empiricalTest: ', idx, tag, w[i], t[i-1])

            for tag in tags:
                tagsCalc[tag] = 0.0
                for fg in fgArr:
                    idx = fg.getFeatureIdx(w, tag, t[i-1], t[i-2], i)
                    if idx != -1:
                        tagsCalc[tag] += weights[idx]
                tagsCalc[tag] = np.exp(tagsCalc[tag])
                denominator += tagsCalc[tag]
            for tag in tags:
                for fg in fgArr:
                    k = fg.getFeatureIdx(w, tag, t[i-1], t[i-2], i)
                    if k != -1:
                        expected[k] += tagsCalc[tag] / denominator

    regulaized_LP = CONST.reg_lambda * weights

    lprimeVec = empirical - expected - regulaized_LP
    retVal = -lprimeVec
    print('finished LPrime', time.asctime())
    return retVal

def prepare_features(sentences, sentences_t, sentences_w, tags, fv, fgArr):

    parser = SentenceParser()
    parser.parseTagedFile(sentences, sentences_t, sentences_w, tags,CONST.train_file_name)

    for w, t in zip(sentences_w, sentences_t):
        for i in range(2, len(t)):
            for fg in fgArr:
                fg.addFeature(fv, w, t[i], t[i-1], t[i-2], i)
    print('fv contains ', fv.getSize(), ' features')
    
"""Run main"""
if __name__ == '__main__':
    main()
