from sentenceParser import *
import numpy as np

class FeatureVec(object):
    
    def __init__(self, fgArr = []):
        self.featureVecSize = -1 
        self.featureIdx2Fg = {}
        self.featureIdx2Tag = {}
        self.fgArr = fgArr
        self.weights = []
        self.corpus = None
        self.empirical = None

    def addFeatureGen(self, fg):
        self.fgArr.append(fg)
        
    def getSize(self):
        return self.featureVecSize + 1

    def setWeights(self, w):
        self.weights = w

    def getWeightForHistory(self, t, t_minus_1, t_minus_2, words, i):
        wRes = 0.0
        for fg in self.fgArr:
            k = fg.getFeatureIdx(words, t, t_minus_1, t_minus_2, i)
            if k != -1:
                wRes += self.weights[k]
        return wRes

    def generateFeatures(self, corpus):
        self.corpus = corpus
        for w, t in zip(corpus.sentences_w, corpus.sentences_t):
            for i in range(2, len(t)):
                for fg in self.fgArr:
                    fg.addFeature(self, w, t[i], t[i - 1], t[i - 2], i)
        print('fv contains ', self.getSize(), ' features')

    def getEmpirical(self):
        if self.empirical is None:
            self.empirical = np.zeros(self.getSize())
            for k in range(self.getSize()):
                self.empirical[k] = self.featureIdx2Fg[k].getCountsByIdx(k)
        return self.empirical



class FeatureGenerator(object):
    
    def __init__(self):
        self.hash2FeatureIdx = {}
        self.hash2Count = {}
        self.featureIdx2hash = {}
        
    def addFeature(self, featureVec, words, t, t_minus_1, t_minus_2, i):
        h , valid = self.getHashAndValid(words, t, t_minus_1, t_minus_2, i)
        if not valid: return
        if h in self.hash2FeatureIdx: 
            # print('already exists ' + t + ' ' + words[i] + ' at index ' + str(self.getFeatureIdx(words, t, t_minus_1, t_minus_2, i)))
            self.hash2Count[h] += 1
        else:
            featureVec.featureVecSize += 1
            # print('adding feature ' + t + ' ' + words[i] + ' at index ' + str(featureVec.featureVecSize))
            self.hash2FeatureIdx[h] = featureVec.featureVecSize
            self.featureIdx2hash[featureVec.featureVecSize] = h
            featureVec.featureIdx2Fg[featureVec.featureVecSize] = self #in order to access specific feature
            featureVec.featureIdx2Tag[featureVec.featureVecSize] = t
            self.hash2Count[h] = 1
            
    def getFeatureIdx(self, words, t, t_minus_1, t_minus_2, i):
        h, valid = self.getHashAndValid(words, t, t_minus_1, t_minus_2, i)
        if not valid: return -1
        if h in self.hash2FeatureIdx:
            # print(type(self).__name__, " fetched")
            return self.hash2FeatureIdx[h]
        return -1
    
    def getWeight(self, weightsVec, words, t, t_minus_1, t_minus_2, i):
        h, valid = self.getHashAndValid(words, t, t_minus_1, t_minus_2, i)
        if not valid: return 0
        if h in self.hash2FeatureIdx:
            return weightsVec[self.hash2FeatureIdx[h]]
        return 0
    
    def getCounts(self, words, t, t_minus_1, t_minus_2, i):
        h, valid = self.getHashAndValid(words, t, t_minus_1, t_minus_2, i)
        if not valid: return 0
        if h in self.hash2Count:
            return self.hash2Count[h]
        return 0
    
    def getCountsByIdx(self, featureVecIdx):
        if featureVecIdx not in self.featureIdx2hash: exit('error')
        h = self.featureIdx2hash[featureVecIdx]
        return self.hash2Count[h]
        
    
    def getHashAndValid(self,words, t, t_minus_1, t_minus_2, i):
        raise Exception('you should implement hash')
    
    def calc(self, featureIdx, words, t, t_minus_1, t_minus_2, i):
#         print(featureIdx)
#         print(self.getHashAndValid(words, t, t_minus_1, t_minus_2, i))
#         print(self.featureIdx2hash[featureIdx])
        h, valid = self.getHashAndValid(words, t, t_minus_1, t_minus_2, i)
        return ( h == self.featureIdx2hash[featureIdx] )


class F100(FeatureGenerator): 
    
    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        return (words[i], t), True

class F101_2(FeatureGenerator):

    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        if len(words[i]) < 2: return None, False
        return (words[i][-2:], t), True


class F101_3(FeatureGenerator):

    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        if len(words[i]) < 3: return None, False
        return (words[i][-3:], t), True

class F101_4(FeatureGenerator):

    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        if len(words[i]) < 4: return None, False
        return (words[i][-4:], t), True

class F102_2(FeatureGenerator):

    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        if len(words[i]) < 2: return None, False
        return (words[i][0:2], t), True

class F102_3(FeatureGenerator):

    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        if len(words[i]) < 3: return None, False
        return (words[i][0:3], t), True

class F102_4(FeatureGenerator):

    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        if len(words[i]) < 4: return None, False
        return (words[i][0:4], t), True

class F103(FeatureGenerator):
    
    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        return (t, t_minus_1, t_minus_2), True

class F104(FeatureGenerator):
    
    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        return (t, t_minus_1), True


class F105(FeatureGenerator):
    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        return (t), True
