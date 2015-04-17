import theano
import numpy
import theano.tensor as T

from capsule import Capsule
from cost import Cost
from train import SGDTrain
from utils import load, shared, translation


class TransAE(object):
    def __init__(self, num_capsules, in_dim, recog_dim, gener_dim, activation, rng=None):
        if rng == None:
            rng = numpy.random.RandomState(numpy.random.randint(2015))
        self.num_capsules= num_capsules
        self.params = []
        self.capsules = []
        for i in xrange(num_capsules):
            cap = Capsule(in_dim, recog_dim, gener_dim, activation)
            self.capsules.append(cap)
            self.params += cap.params

        self.b_out = theano.shared(value=numpy.zeros(shape=(in_dim,),
                                                     dtype=theano.config.floatX),
                                   name='b_out',
                                   borrow=True)
        self.params.append(self.b_out)

    def fprop(self, input, extra_input):
        cap_out = []
        for i in xrange(num_capsules):
            out, prob = self.capsules[i].fprop(input, extra_input)
            cap_out.append((out, prob))
        #prob_sum = sum([result[1] for result in cap_out])
        caps_out = sum([result[0]*result[1] for result in cap_out])
        shifted_img = T.nnet.sigmoid(caps_out + self.b_out)
        return shifted_img

if __name__ == "__main__":
    train, valid, test = load('mnist.pkl.gz')
    trans_train, shift_train, ori_train = translation(train[0], 28)
    trans_train, shift_train, ori_train = shared((trans_train, shift_train, ori_train))
    trans_valid, shift_valid, ori_valid = translation(valid[0], 28)
    trans_valid, shift_valid, ori_valid = shared((trans_valid, shift_valid, ori_valid))
    trans_test, shift_test, ori_test = translation(test[0], 28)
    trans_test, shift_test, ori_test = shared((trans_test, shift_test, ori_test))

    num_capsules = 100
    in_dim = 784
    recog_dim = 10
    gener_dim = 20
    activation = 'sigmoid'

    input = T.matrix('input')
    extra_input = T.matrix('extra')
    output = T.matrix('output')
    transae = TransAE(num_capsules, in_dim, recog_dim, gener_dim, activation)
    cost = Cost(transae, input, extra_input)
    model = SGDTrain(input, extra_input, output, (trans_train, shift_train, ori_train), transae, cost)
    model.main_loop((trans_valid, shift_valid, ori_valid),
                    (trans_test, shift_test, ori_test),
                    epochs=1000, serialize=True)
