import theano
import numpy
import theano.tensor as T

import Capsule
from utils import load, shared, translation

class TransAE(object):
    def __init__(self, num_capsules, in_dim, recog_dim, gener_dim, activation, rng=None):
        self.num_capsules= num_capsules
        self.params = []
        self.capsules = []
        for i in xrange(num_capsules):
            cap = Capsule(in_dim, recog_dim, gener_dim, activation)
            self.capsules.append(cap)
            self.params += cap.params

    def fprop(self, input, extra_input):
        results = []
        for i in xrange(num_capsules):
            out, prob = self.capsules[i].fprop(input, extra_input)
            results.append((out, prob))
        prob_sum = sum([result[1] for result in results])
        aver_result = sum([result[0]*result[1]/prob_sum for result in results])
        return aver_result

if __name__ == "__main__":
    train, valid, test = load('mnist.pkl.gz')
    trans_train, trans, ori_train = translation(train[0], 28)
    trans_train, trans, ori_train = shared((trans_train, trans, ori_train))

    num_capsules = 10
    in_dim = 784
    recog_dim = 128
    gener_dim = 128
    activation = 'relu'

    transae = TransAE(num_capsules, in_dim, recog_dim, gener_dim, activation)
    #main-loop
