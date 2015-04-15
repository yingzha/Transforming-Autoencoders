"""This is the capsule which performs internal computations on their inputs."""
__author__ = "Ying Zhang"
__copyright__ = "University of Montreal, 2015"

import numpy
import theano
import theano.tensor as T

class Capsule(object):
    #only consider translation now
    def __init__(self, in_dim, recog_dim, gener_dim, activation='sigmoid', rng=None):
        if rng == None:
            rng = numpy.random.RandomState(numpy.random.randint(2015))
        self.in_dim = in_dim
        self.recog_dim = recog_dim
        self.gener_dim = gener_dim
        self.activation = activation
        self.rng = rng

        if activation == 'sigmoid':
            w_bound_rec = numpy.sqrt(6. / (in_dim + recog_dim))
            w_bound_axis = numpy.sqrt(6. / (recog_dim + 2))
            w_bound_pro = numpy.sqrt(6. / (recog_dim + 1))
            w_bound_gener = numpy.sqrt(6. / (2 + gener_dim))

        self.W_rec = theano.shared(value=numpy.asarray(rng.uniform(low=-w_bound_rec,
                                                                   high=w_bound_rec,
                                                                   size=(in_dim, recog_dim)),
                                                       dtype=theano.config.floatX),
                                   name='W_rec',
                                   borrow=True)
        self.b_rec = theano.shared(value=numpy.zeros(shape=(recog_dim,),
                                                     dtype=theano.config.floatX),
                                   name='b_rec',
                                   borrow=True)

        self.W_xy_axis = theano.shared(value=numpy.asarray(rng.uniform(low=-w_bound_axis,
                                                                       high=w_bound_axis,
                                                                       size=(recog_dim, 2)),
                                                          dtype=theano.config.floatX),
                                       name='W_xy_axis',
                                       borrow=True)
        self.b_xy_axis = theano.shared(value=numpy.zeros(shape=(2,),
                                                         dtype=theano.config.floatX),
                                       name='b_xy_axis',
                                       borrow=True)
        self.W_pr = theano.shared(value=numpy.asarray(rng.uniform(low=-w_bound_pro,
                                                                  high=-w_bound_pro,
                                                                  size=(recog_dim, 1)),
                                                      dtype=theano.config.floatX),
                                  name='W_pr',
                                  borrow=True)
        self.b_pr = theano.shared(value=numpy.zeros(shape=(1,),
                                                    dtype=theano.config.floatX),
                                  name='b_pr',
                                  borrow=True)


        self.W_gen = theano.shared(value=numpy.asarray(rng.uniform(low=-w_bound_gener,
                                                                   high=w_bound_gener,
                                                                   size=(2, gener_dim)),
                                                       dtype=theano.config.floatX),
                                   name='W_gen',
                                   borrow=True)
        self.b_gen = theano.shared(value=numpy.zeros(shape=(gener_dim,),
                                                     dtype=theano.config.floatX),
                                   name='b_gen',
                                   borrow=True)

        self.W_out = theano.shared(value=numpy.asarray(rng.uniform(low=-.05,
                                                                   high=-.05,
                                                                   size=(gener_dim, in_dim)),
                                                       dtype=theano.config.floatX),
                                   name='W_out',
                                   borrow=True)

        self.params = [self.W_rec, self.b_rec, self.W_xy_axis,
                       self.b_xy_axis, self.W_gen, self.b_gen, self.W_out]

    def fprop(self, input, extra_input):
        rec = T.nnet.sigmoid(T.dot(input, self.W_rec) + self.b_rec)
        xy_axis = (T.dot(rec, self.W_xy_axis) + self.b_xy_axis)
        pro_tmp = T.nnet.sigmoid(T.dot(rec, self.W_pr) + self.b_pr)
        pro = T.extra_ops.repeat(pro_tmp, self.in_dim, axis=1)
        renew_input = xy_axis + extra_input
        gener = T.nnet.sigmoid(T.dot(renew_input, self.W_gen) + self.b_gen)
        raw_out = T.dot(gener, self.W_out)
        return raw_out, pro
