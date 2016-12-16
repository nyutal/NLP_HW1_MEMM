
class FeatureVec(object):
    
    def __init__(self):
        self.featureVecSize = -1 
        self.featureIdx2Fg = {}
        self.featureIdx2Tag = {} 
        
    def getSize(self):
        return self.featureVecSize + 1   

class FeatureGenerator(object):
    
    def __init__(self):
        self.hash2FeatureIdx = {}
        self.hash2Count = {}
        self.featureIdx2hash = {}
        
    def addFeature(self, featureVec, words, t, t_minus_1, t_minus_2, i):
        h , valid = self.getHashAndValid(words, t, t_minus_1, t_minus_2, i)
        if not valid: return
        if h in self.hash2FeatureIdx: 
#             print('already exists ' + t + ' ' + words[i] + ' at index ' + str(self.getFeatureIdx(words, t, t_minus_1, t_minus_2, i)))
            self.hash2Count[h] += 1
        else:
            featureVec.featureVecSize += 1
            print('adding feature ' + t + ' ' + words[i] + ' at index ' + str(featureVec.featureVecSize))
            self.hash2FeatureIdx[h] = featureVec.featureVecSize
            self.featureIdx2hash[featureVec.featureVecSize] = h
            featureVec.featureIdx2Fg[featureVec.featureVecSize] = self #in order to access specific feature
            featureVec.featureIdx2Tag[featureVec.featureVecSize] = t
            self.hash2Count[h] = 1
            
    def getFeatureIdx(self, words, t, t_minus_1, t_minus_2, i):
        h, valid = self.getHashAndValid(words, t, t_minus_1, t_minus_2, i)
        if h in self.hash2FeatureIdx:
            return self.hash2FeatureIdx[h]
        return -1
    
    def getWeight(self, weightsVec, words, t, t_minus_1, t_minus_2, i):
        h, valid = self.getHashAndValid(words, t, t_minus_1, t_minus_2, i)
        if h in self.hash2FeatureIdx:
            return weightsVec[self.hash2FeatureIdx[h]]
        return 0
    
    def getCounts(self, words, t, t_minus_1, t_minus_2, i):
        h, valid = self.getHashAndValid(words, t, t_minus_1, t_minus_2, i)
        if h in self.hash2Count:
            return self.hash2Count[h]
        return 0
    
    def getCountsByIdx(self, featureVecIdx):
        h = self.featureIdx2hash[featureVecIdx]
        return self.hash2Count[h]
        
    
    def getHashAndValid(self,words, t, t_minus_1, t_minus_2, i):
        raise Exception('you should implement hash')
    
    def calc(self, featureIdx, words, t, t_minus_1, t_minus_2, i):
        return ( self.getHashAndValid(words, t, t_minus_1, t_minus_2, i) == self.featureIdx2hash[featureIdx] )


class F100(FeatureGenerator): 
    
    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        return (words[i], t), True
    
class F103(FeatureGenerator):
    
    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        return (t, t_minus_1, t_minus_2), True

class F104(FeatureGenerator):
    
    def getHashAndValid(self, words, t, t_minus_1, t_minus_2, i):
        return (t, t_minus_1), True
    