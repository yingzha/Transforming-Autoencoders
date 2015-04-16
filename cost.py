__author__ = 'Ying Zhang'
__copyright__ = 'Copyright 2015, University of Montreal'

import numpy as np
import theano
import theano.tensor as T

class Cost(object):
    def __init__(self, model, input, extra_input):
        self.model = model
        self.y_pred = self.model.fprop(input, extra_input)

    def L1_regu(self):
        return sum([(abs(param)).sum() for param in self.model.params])

    def L2_regu(self):
        return sum([(param **2).sum() for param in self.model.params])

    def mse(self, y):
        loss = T.mean(((self.y_pred - y) ** 2).sum(axis=1))
        return loss

    def cross_entropy(self, y):
        return T.mean(T.nnet.categorical_crossentropy(self.y_pred, y))

    def get_output(self):
        return self.y_pred
