import theano
import numpy
import theano.tensor as T

from capsule import Capsule
from cost import Cost
from train import SGDTrain
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
           # out  = self.capsules[i].fprop(input, extra_input)
           # results.append(out)
            results.append((out, prob))
        prob_sum = sum([result[1] for result in results])
        aver_result = sum([result[0]*result[1]/prob_sum for result in results])
        #aver_result = sum(results)
        return aver_result

if __name__ == "__main__":
    train, valid, test = load('mnist.pkl.gz')
    trans_train, shift_train, ori_train = translation(train[0], 28)
    trans_train, shift_train, ori_train = shared((trans_train, shift_train, ori_train))
    trans_valid, shift_valid, ori_valid = translation(valid[0], 28)
    trans_valid, shift_valid, ori_valid = shared((trans_valid, shift_valid, ori_valid))
    trans_test, shift_test, ori_test = translation(test[0], 28)
    trans_test, shift_test, ori_test = shared((trans_test, shift_test, ori_test))

    num_capsules = 30
    in_dim = 784
    recog_dim = 1000
    gener_dim = 784
    activation = 'relu'

    input = T.matrix('input')
    extra_input = T.matrix('extra')
    output = T.matrix('output')
    transae = TransAE(num_capsules, in_dim, recog_dim, gener_dim, activation)
    cost = Cost(transae, input, extra_input).mse(output)
    model = SGDTrain(input, extra_input, output, (trans_train, shift_train, ori_train), transae, cost)
    model.main_loop((trans_valid, shift_valid, ori_valid),
                    (trans_test, shift_test, ori_test),
                    epochs=50)
