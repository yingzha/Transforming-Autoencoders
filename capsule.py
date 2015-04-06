"""This is the capsule which performs internal computations on their inputs."""
__author__ = "Ying Zhang"
__copyright__ = "University of Montreal, 2015"

import numpy
import theano
import theano.tensor as T

class Capsule(object):
    #only consider translation
    def __init__(self, in_dim, recog_dim, gener_dim, activation, rng=None):
        if rng == None:
            rng = numpy.random.RandomState(numpy.random.randint(2015))
        self.in_dim = in_dim
        self.recog_dim = recog_dim
        self.gener_dim = gener_dim
        self.activation = activation
        self.rng = rng

        if activation == 'sigmoid':
            w_bound_rec = numpy.sqrt(6. / (in_dim + recog_dim))
            w_bound_axis = numpy.sqrt(6. / (recog_dim + 1))
            w_bound_gener = numpy.sqrt(6. / (2 + gener_dim))
            w_bound_out = numpy.sqrt(6. / (gener_dim + in_dim))
            self.activation_func = T.nnet.softmax
        elif activation == 'tanh':
            w_bound_rec = numpy.sqrt(4. * 6. / (in_dim + recog_dim))
            w_bound_axis = numpy.sqrt(4. * 6. / (recog_dim + 1))
            w_bound_gener = numpy.sqrt(4. * 6. / (2 + gener_dim))
            w_bound_out = numpy.sqrt(4. * 6. / (gener_dim + in_dim))
            self.activation_func = T.tanh
        elif activation == 'relu':
            #should be more elegant
            w_bound_rec = .01
            w_bound_axis = .01
            w_bound_gener= .01
            w_bound_out = .01
            self.activation_func = lambda x: x * (x > 0)

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

        self.W_x_axis = theano.shared(value=numpy.asarray(rng.uniform(low=-w_bound_axis,
                                                                      high=w_bound_axis,
                                                                      size=(recog_dim, 1)),
                                                          dtype=theano.config.floatX),
                                      name='W_x_axis',
                                      borrow=True)
        self.b_x_axis = theano.shared(value=numpy.zeros(shape=(1,),
                                                        dtype=theano.config.floatX),
                                      name='b_x_axis',
                                      borrow=True)

        self.W_y_axis = theano.shared(value=numpy.asarray(rng.uniform(low=-w_bound_axis,
                                                                      high=w_bound_axis,
                                                                      size=(recog_dim, 1)),
                                                          dtype=theano.config.floatX),
                                      name='W_y_axis',
                                      borrow=True)
        self.b_y_axis = theano.shared(value=numpy.zeros(shape=(1,),
                                                        dtype=theano.config.floatX),
                                      name='b_y_axis',
                                      borrow=True)
        self.W_pr = theano.shared(value=numpy.asarray(rng.uniform(low=-.01,
                                                                  high=.01,
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
        self.W_out = theano.shared(value=numpy.asarray(rng.uniform(low=-w_bound_rec,
                                                                   high=w_bound_rec,
                                                                   size=(gener_dim, in_dim)),
                                                       dtype=theano.config.floatX),
                                   name='W_out',
                                   borrow=True)
        self.b_out = theano.shared(value=numpy.zeros(shape=(in_dim,),
                                                     dtype=theano.config.floatX),
                                   name='b_out',
                                   borrow=True)

        self.params = [self.W_rec, self.b_rec, self.W_x_axis, self.b_x_axis,
                       self.W_y_axis, self.b_y_axis, self.W_gen, self.b_gen,
                       self.W_out, self.b_out]

    def fprop(self, input, extra_input):
        rec = self.activation_func(T.dot(input, self.W_rec) + self.b_rec)
        x_axis = self.activation_func(T.dot(rec, self.W_x_axis) + self.b_x_axis)
        y_axis = self.activation_func(T.dot(rec, self.W_y_axis) + self.b_y_axis)
        pro = T.nnet.sigmoid(T.dot(rec, self.W_pr) + self.b_pr)
        renew_input = T.concatenate([x_axis, y_axis], axis=1) + extra_input
        gener = self.activation_func(T.dot(renew_input, self.W_gen) + self.b_gen)
        output = pro * self.activation_func(T.dot(gener, self.W_out) + self.b_out)
        return output

