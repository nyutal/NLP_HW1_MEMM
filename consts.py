# DEFINES
import numpy as np
class CONST(object):
    reg_lambda = 8.0
    epsilon = np.finfo(float).eps
    accuracy = {'low':10**16, 'med':10**10, 'high':10**10}
    test_file_name = 'test.wtag'
    train_file_name = 'train.wtag'
    viterbiMulFactor = 100000

    def __setattr__(self, *_):
        raise ValueError("don't you dare!")
CONST = CONST()
