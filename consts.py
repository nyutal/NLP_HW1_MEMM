# DEFINES
class CONST(object):
    reg_lambda = 1
    epsilon = 1e-08

    train_file_name = 'train_small.wtag'
    # train_file_name = 'train.wtag'

    def __setattr__(self, *_):
        raise ValueError("don't you dare!")
CONST = CONST()
