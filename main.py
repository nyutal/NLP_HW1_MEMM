from __future__ import print_function
from builtins import range
from future.utils import iteritems
import numpy as np
import logging
from consts import CONST

logging.basicConfig(filename='hw1.log', filemode='w', level=logging.DEBUG)

class MyClass(object):
    epsilon = 10 ** (-4)

    def __init__(self, nid):
        self.enabled = True

    def __lt__(self, other):
        if type(self) is str(type(other)):
            return self.nid < other.nid
        else:
            return str(type(self)) < str(type(other))

    def reset(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


def main():
    """Compare runs for various test graph initializations"""

    my_vars = ['A', 'B', 'C']  # Sanity check
    my_dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}

    for i in range(CONST.graph_size[1]):
        print(np.maximum(3, i))

    for k,v in iteritems(my_dict):
        print(k,v)

    print('all done')
    logging.warning('Warn1')



"""Run main"""
if __name__ == '__main__':
    main()
