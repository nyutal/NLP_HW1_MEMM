# DEFINES
import numpy as np
class CONST(object):
    reg_lambda = 0.0
    max_iter = 200
    epsilon = np.finfo(float).eps
    accuracy = {'low':10**16, 'med':10**10, 'high':10**10}
    test_file_name = 'test.wtag'
    train_file_name = 'train.wtag'
    comp_file_name = 'comp.words'
    #  train_file_name = 'train_short.wtag'
    viterbiMulFactor = 100000
    parallel = True
    num_of_learners = 6
    num_of_viterbers = 6

    def __setattr__(self, *_):
        raise ValueError("don't you dare!")
CONST = CONST()
