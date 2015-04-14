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

        self.W_out = theano.shared(value=numpy.asarray(rng.uniform(low=-w_bound_out,
                                                                   high=w_bound_out,
                                                                   size=(gener_dim*num_capsules, in_dim)),
                                                       dtype=theano.config.floatX),
                                   name='W_out',
                                   borrow=True)

        self.b_out = theano.shared(value=numpy.zeros(shape=(in_dim,),
                                                     dtype=theano.config.floatX),
                                   name='b_out',
                                   borrow=True)

        self.params += self.W_out + self.b_out

    def fprop(self, input, extra_input):
        cap_out = []
        for i in xrange(num_capsules):
            out, prob = self.capsules[i].fprop(input, extra_input)
            cap_out.append((out, prob))
        prob_sum = sum([result[1] for result in results])
        aver_result = T.concatenate([result[0]*result[1]/prob_sum for result in results],
                                    axis=1)
        shifted_img = T.dot(aver_result, self.W_out) + self.b_out
        return shifted_img

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
    recog_dim = 40
    gener_dim = 40
    activation = 'sigmoid'

    input = T.matrix('input')
    extra_input = T.matrix('extra')
    output = T.matrix('output')
    transae = TransAE(num_capsules, in_dim, recog_dim, gener_dim, activation)
    cost = Cost(transae, input, extra_input)
    model = SGDTrain(input, extra_input, output, (trans_train, shift_train, ori_train), transae, cost)
    model.main_loop((trans_valid, shift_valid, ori_valid),
                    (trans_test, shift_test, ori_test),
                    epochs=50)
