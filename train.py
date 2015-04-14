__author__ = 'Ying Zhang'
__copyright__ = 'Copyright 2015, University of Montreal'

import numpy as np
import theano
import theano.tensor as T
import time
import os
import cPickle
import matplotlib
class SGDTrain(object):
    def __init__(self, input1, extra_input, output, data, model, cost,
            batch_size=50, init_lr=.001, init_mom=.1):
        self.__dict__.update(locals())
        del self.self
        self.status = {}
        self.status.update(locals())
        del self.status['self']
        del self.status['model']

        self.params = model.params
        self.pgrad = T.grad(cost, self.params)
        self.momentum = [theano.shared(param.get_value() * .0)
                         for param in self.params]
        self.batch_num = len(data[0].get_value()) / batch_size
        learning_rate = T.scalar('lr')
        momentum = T.scalar('momentum')
        self.trainmodel = self.build_trainmodel(learning_rate, momentum)

    def set_learning_rate(self, epoch, de_con=1e-3):
        return self.init_lr / (1. + de_con*epoch)

    def set_momentum(self, epoch):
        return self.init_mom

    def _get_updates(self, learning_rate, mom=.0):
        """update momentum and parameters"""
        update_moms = [mom * moment - learning_rate * grad for moment, grad in
                  zip(self.momentum, self.pgrad)]
        updates = [(param, param + update_mom) for param, update_mom in
                    zip(self.params, update_moms)]
        updates.extend([(moment, update_mom) for moment, update_mom in
                         zip(self.momentum, update_moms)])
        return updates

    def build_trainmodel(self, learning_rate, mom):
        index = T.lscalar('index')
        updates = self._get_updates(learning_rate, mom)
        func = theano.function(inputs=[index, learning_rate, mom],
                               outputs=self.cost,
                               updates=updates,
                               givens={
                                   self.input1: self.data[0][index*self.batch_size: (index+1)*self.batch_size],
                                   self.extra_input: self.data[1][index*self.batch_size: (index+1)*self.batch_size],
                                   self.output: self.data[2][index*self.batch_size: (index+1)*self.batch_size]
                                   }
                              )
        return func

    def build_validmodel(self, validset):
         func = theano.function(inputs=[],
                                outputs=self.cost,
                                givens={
                                    self.input1: validset[0],
                                    self.extra_input: validset[1],
                                    self.output: validset[2]
                                    }
                               )
         return func

    def build_testmodel(self, testset):
        func = theano.function(inputs=[],
                               outputs=self.cost,
                               givens={
                                   self.input1: testset[0],
                                   self.extra_input1: testset[1],
                                   self.output: testset[2]
                                   }
                              )
        return func


    def main_loop(self, validset, testset, epochs=30, serialize=False, verbose=False):
        best_valid_error = np.inf
        self.status['cost_per_epoch'] = []
        self.status['learning_rate_per_epoch'] = []
        self.status['momentum_per_epoch'] = []
        print "Start training..."
        for epoch in range(epochs):
            start_time = time.clock()
            lr = self.set_learning_rate(epoch)
            mom =self.set_momentum(epoch)
            for minibatch_index in xrange(self.batch_num):
                cost = self.trainmodel(minibatch_index, lr, mom)
                print 'Epoch {0}, minibatch {1}/{2}, cost {3}'.format(epoch+1, minibatch_index, self.batch_num, cost)
            self.status['cost_per_epoch'].append(cost)
            self.status['learning_rate_per_epoch'].append(lr)
            self.status['momentum_per_epoch'].append(mom)
            self.status['epoch'] = epoch
            print 'learning_rate {0}, momentum {1}'.format(lr, mom)
            end_time = time.clock()
            print 'Running time: {0}'.format(end_time - start_time)
            valid_cost = self.build_validmodel(validset)()
            test_cost = self.build_testmodel(testset)()
            print 'Epoch {0}, validation cost{1}, test cost{2}'.format(valid_cost, test_cost)

            if serialize:
                self.save_model(epoch)
            if verbose:
                pass
        print "End training..."
