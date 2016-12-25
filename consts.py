# DEFINES
import numpy as np
class CONST(object):
    reg_lambda = 0.7
    max_iter = 200
    epsilon = np.finfo(float).eps
    test_file_name = 'test.wtag'
    train_file_name = 'train.wtag'
    comp_file_name = 'comp.words'
    parallel = True
    num_of_learners = 4
    num_of_viterbers = 4
    train = True
    isValidate = True
    testName = 'final_complex_0_7_model'

    def __setattr__(self, *_):
        raise ValueError("don't you dare!")
CONST = CONST()
