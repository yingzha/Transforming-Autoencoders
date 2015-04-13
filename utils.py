import gzip
import cPickle
import theano
import numpy

import theano.tensor as T

def load(dataset):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def shared(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def translation(data_x, shape):
    #TODO: consider padding
    data_x_2d = numpy.reshape(data_x, (len(data_x), shape, shape))
    rng = numpy.random.RandomState(2015)
    trans_data, trans, true_data = [], [], []
    for i in numpy.random.permutation(len(data_x_2d)):
        trans_x = rng.randint(low=-3, high=4)
        trans_y = rng.randint(low=-3, high=4)
        trans_shift_x = numpy.roll(data_x_2d[i], trans_x, axis=0)
        trans_shift_xy = numpy.roll(trans_shift_x, trans_y, axis=1)
        trans_data.append(trans_shift_xy.flatten())
        trans.append((trans_x, trans_y))
        true_data.append(data_x[i])
    return trans_data, trans, true_data



