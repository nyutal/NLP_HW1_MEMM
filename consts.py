# DEFINES
class CONST(object):
    reg_lambda = 1

    def __setattr__(self, *_):
        raise ValueError("don't you dare!")
CONST = CONST()
