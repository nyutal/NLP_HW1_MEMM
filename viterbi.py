import numpy as np


class Viterbi(object):

    def __init__(self, tags, fv, weights):
        self.preSk = ['*']
        self.postSk = ['SSS']
        self.middleSk = tags
        self.fv = fv
        self.weights = weights

    def solve(self, sentence, addPrefix=False):
        fullSentence = []
        if addPrefix: fullSentence = ['*', '*'] + sentence + ['SSS']
        else: fullSentence = sentence + ['SSS']
        l = len(fullSentence)
        tags = ['*', '*']
        pi = {}
        pi[(1, '*', '*')] = 1.0
        bp = {}
        for k in range(2, l):
            for u in self.getSk(k-1):
                for v in self.getSk(k):
                    piMax = float("-inf")
                    b = None
                    for t in self.getSk(k-2):
                        curr = pi[(k-1), t, u] * self.fv.getQ(v, u, t, fullSentence, k)
                        if piMax < curr:
                            piMax = curr
                            b = t
                    pi[(k, u, v)] = piMax
                    bp[(k, u, v)] = b
                    # print(k, u, v, b, piMax)

        piMax = float("-inf")
        uMax = None
        vMax = None
        for u in self.getSk(l - 2):
            for v in self.getSk(l-1):
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
        print(reversedPath)
        return reversedPath.reverse()

    def getSk(self, k):
        if k < 2: return self.preSk
        return self.middleSk
