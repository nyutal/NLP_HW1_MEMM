from consts import CONST
import numpy as np

class Viterbi(object):
    def __init__(self, model):
        self.preSk = ['*']
        self.postSk = ['SSS']
        self.middleSk = model.corpus.getTags()
        if '*' in self.middleSk: self.middleSk.remove('*')
        if 'SSS' in self.middleSk: self.middleSk.remove('SSS')
        self.model = model

    def solve(self, sentence, addPrefixSuffix=False):
        if addPrefixSuffix: fullSentence = ['*', '*'] + sentence + ['SSS']
        else: fullSentence = sentence
        l = len(fullSentence)
        pi = {(1, '*', '*'): 1.0}
        bp = {}
        for k in range(2, l):
            for u in self.getSk(k-1, l):
                for v in self.getSk(k, l):
                    piMax = float("-inf")
                    b = None
                    for t in self.getSk(k-2, l):
                        curr = pi[(k-1), t, u] * self.model.getWeightForHistory(v, u, t, fullSentence, k)
                        if piMax < curr:
                            piMax = curr
                            b = t
                    pi[(k, u, v)] = piMax
                    bp[(k, u, v)] = b
                    # print(k, u, v, b, piMax)

        piMax = float("-inf")
        uMax = None
        vMax = None
        for u in self.getSk(l - 2, l):
            for v in self.getSk(l-1, l):
                if pi[l-1, u, v ] > piMax:
                    piMax = pi[l-1, u, v ]
                    uMax = u
                    vMax = v
        # print(l-1, uMax, vMax, )
        reversedPath = [vMax, uMax]
        # for b in bp:
        #     print(b)
        for k in range(l-3, 1, -1):
            # print("adding to v, u ", str(k), reversedPath[-1], reversedPath[-2], bp[(k+2,reversedPath[-1], reversedPath[-2])])
            reversedPath.append(bp[(k+2,reversedPath[-1], reversedPath[-2])])
        # print(reversedPath)
        return list(reversed(reversedPath))

    def fullSolve(self, sentence, addPrefixSuffix=False):
        if addPrefixSuffix: fullSentence = ['*', '*'] + sentence + ['SSS']
        else: fullSentence = sentence
        l = len(fullSentence)
        pi = {(1, '*', '*'): 1.0}
        bp = {}
        for k in range(2, l):
            tuvExpFv = {}
            sumTuExpFv = {}
            for t in self.getSk(k - 2, l):
                for u in self.getSk(k-1, l):
                    sumTuExpFv[t, u] = 0.0
                    for v in self.getSk(k, l):
                        w = self.model.getWeightForHistory(v, u, t, fullSentence, k)
                        tuvExpFv[t, u, v] = np.exp(w)
                        sumTuExpFv[t, u] += tuvExpFv[t, u, v]

            for u in self.getSk(k-1, l):
                for v in self.getSk(k, l):
                    piMax = float("-inf")
                    b = None
                    for t in self.getSk(k-2, l):
                        try:
                            curr = pi[(k-1), t, u] * ( tuvExpFv[t,u,v] / sumTuExpFv[t, u] )
                        except FloatingPointError:
                            curr = 0.0

                        if piMax < curr:
                            piMax = curr
                            b = t
                    pi[(k, u, v)] = piMax
                    bp[(k, u, v)] = b
                    # print(k, u, v, b, piMax)

        piMax = float("-inf")
        uMax = None
        vMax = None
        for u in self.getSk(l - 2, l):
            for v in self.getSk(l-1, l):
                if pi[l-1, u, v ] > piMax:
                    piMax = pi[l-1, u, v ]
                    uMax = u
                    vMax = v
        reversedPath = [vMax, uMax]
        for k in range(l-3, 1, -1):
            # print("adding to v, u ", str(k), reversedPath[-1], reversedPath[-2], bp[(k+2,reversedPath[-1], reversedPath[-2])])
            reversedPath.append(bp[(k+2,reversedPath[-1], reversedPath[-2])])
        return list(reversed(reversedPath))

    def getSk(self, k, l):
        if k < 2: return self.preSk
        elif k < l-1: return self.middleSk
        return self.postSk