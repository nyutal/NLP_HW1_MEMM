# DEFINES
import numpy as np
class CONST(object):
    reg_lambda = 0.0
    epsilon = np.finfo(float).eps
    accuracy = {'low':10**16, 'med':10**10, 'high':10**10}
    # train_file_name = 'train_small.wtag'
    train_file_name = 'train.wtag'
    viterbiMulFactor = 100000

    def __setattr__(self, *_):
        raise ValueError("don't you dare!")
CONST = CONST()
