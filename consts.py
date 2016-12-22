# DEFINES
import numpy as np
class CONST(object):
    reg_lambda = 2.0
    max_iter = 100
    epsilon = np.finfo(float).eps
    test_file_name = 'test.wtag'
    train_file_name = 'train.wtag'
    comp_file_name = 'comp.words'
    parallel = True
    num_of_learners = 6
    num_of_viterbers = 6
    train = True
    isValidate = True

    def __setattr__(self, *_):
        raise ValueError("don't you dare!")
CONST = CONST()
