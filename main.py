from __future__ import print_function
from builtins import range
import scipy as sp
import scipy.optimize
import numpy as np
import logging
from consts import CONST
from featureFunc import *

logging.basicConfig(filename='hw1.log', filemode='w', level=logging.DEBUG)


def main():
    """Compare runs for various test graph initializations"""

    sentences = []
    sentences_w = []
    sentences_t = []
    features = {}
    tags = set()
    
    fv = FeatureVec()
    fgArr = []
    fgArr.append(F100())
    fgArr.append(F103())
    fgArr.append(F104())

    prepare_features(sentences, sentences_t, sentences_w, features, tags, fv, fgArr)

    x1, f1, d1 = sp.optimize.fmin_l_bfgs_b(calc_L,
                                           x0=np.zeros(fv.getSize()),
                                           args=(fgArr, sentences_t, sentences_w, tags, fv),
                                           fprime=calc_Lprime, m=100,
                                           pgtol=1e-3, disp=True)
    print('x1:', x1)
    print('f1:', f1)
    print('d1:', d1)

    logging.info('Done!')
    print("Done!")


def calc_L(weights, fgArr, sentences_t, sentences_w, tags, fv):
    print('start L')
    c = 0
    s1 = 0.0
    s2 = 0.0
    for w, t in zip(sentences_w, sentences_t):
        for i in range(2, len(t)):
            c += 1
#             if ( c % 1000 == 0 ): print('sample ' + str(c))
            prelogexp = 0.0
            for fg in fgArr:
                idx = fg.getFeatureIdx(w, t[i], t[i-1], t[i-2], i)
                if ( idx != -1 ):
                    s1 += weights[idx] 
                for tag in tags:
                    idx = fg.getFeatureIdx(w, tag, t[i-1], t[i-2], i)
                    if ( idx != -1 ):
                        prelogexp += weights[idx]
            s2 += np.log(np.exp(prelogexp))
    
#     s1 = calc_v_dot_f(fgArr, sentences_t, sentences_w, weights)
#     s2 = calc_v_dot_f_for_tag_in_tags(fgArr, sentences_t, sentences_w, weights, tags)
    regularizer_L = (CONST.reg_lambda/2) * (np.linalg.norm(weights))**2
    retVal = -float(s1 - s2 - regularizer_L)
    print('finish L ' + str(retVal))
    return retVal

# v dot f of all sentences: one parameter at the time:
def calc_Lprime(weights, fgArr, sentences_t, sentences_w, tags, fv):
    c = 0
    print('start LPrime')
    print('start LPrime-Empirical')
    empirical = np.zeros(fv.getSize())
    for k in range(fv.getSize()):
        empirical[k] = fv.featureIdx2Fg[k].getCountsByIdx(k)
    print('finish LPrime-Empirical ')
    print(empirical)
    
    
    expected = np.zeros(fv.getSize())
    for w, t in zip(sentences_w, sentences_t):
        for i in range(2, len(t)):
            c += 1
            if ( c % 10000 == 0 ): print('LPrime sample ' + str(c))
            tagsCalc = {}
            denominator = 0.0
            for tag in tags:
                tagsCalc[tag] = 0.0
                for fg in fgArr:
                    idx = fg.getFeatureIdx(w, tag, t[i-1], t[i-2], i)
                    if ( idx != -1 ):
                        tagsCalc[tag] += weights[idx]
                np.exp(tagsCalc[tag])
                denominator += tagsCalc[tag]
            for k in range(fv.getSize()):
                tag = fv.featureIdx2Tag[k]
                if ( fv.featureIdx2Fg[k].calc(k, w, tag, t[i-1], t[i-2], i)):
                    expected[k] += tagsCalc[tag] / denominator
#                 for itag, tag in enumerate(tags):
#                     if ( fv.featureIdx2Fg[k].calc(k, w, tag, t[i-1], t[i-2], i)):
#                         expected[k] += tagsCalc[tag] / denominator
            
    regulaized = CONST.reg_lambda * weights
    
    lprimeVec = empirical - expected - regulaized
    retVal = -lprimeVec
    print('finish LPrime')
    print(retVal)
    return retVal

def prepare_features(sentences, sentences_t, sentences_w, features, tags, fv, fgArr):
    lines = [line.rstrip('\n') for line in open(CONST.train_file_name)]
    for line in lines:
        w = ['*_*', '*_*']  # start
        w.extend(line.split(" "))
#         if w[-1] == '._.':
#             del w[-1]  # remove 'period' from the end of the sentence
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

    print('all tags(', len(tags), '):', tags)

    for w, t in zip(sentences_w, sentences_t):
        for i in range(2, len(t)):
            for fg in fgArr:
                fg.addFeature(fv, w, t[i], t[i-1], t[i-2], i)
  
    
"""Run main"""
if __name__ == '__main__':
    main()
