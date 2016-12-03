# DEFINES
class CONST(object):
    graph_size = (7, 7)  # will be NxK

    def __setattr__(self, *_):
        raise ValueError("don't you dare!")
CONST = CONST()
