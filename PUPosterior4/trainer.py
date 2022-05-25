import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import pdb

class Trainer:
    def __init__(self, nnLoss, data, maxIter, batchSize):
        # pdb.set_trace()
        self.opt = tf.keras.optimizers.Adam( )
        self.nnLoss = nnLoss
        self.data_tr, _, self.data_val = data.getTTVData()
        self.data = data
        self.batchSize = batchSize
        self.maxIter = maxIter
        self.bestNNLoss = nnLoss
        self.losses = []
        self.valLosses = []
        self.bestValLoss = np.inf

    def loss(self):
        return self.nnLoss.loss(self.data)

    def fit(self, hypPar=None, run=None):
        self.beforeTraining()
        #pdb.set_trace()
        if 'iter' in run:
            iter = run['iter']
        else:
            iter = self.maxIter
        for i in np.arange(iter):
            if 'convergence' in run and run['convergence']:
                if len(self.valLosses) >= 100 and min(self.valLosses[-100:]) > self.bestValLoss:
                    break
            self.iter = i
            self.iteration(hypPar)
            valLoss = self.nnLoss.valLoss(self.data_val, hypPar)
            self.valLosses.append(valLoss)
            if valLoss < self.bestValLoss:
                self.bestValLoss = valLoss
                self.bestNNLoss = self.nnLoss.copy()
        #pdb.set_trace()
        return

    def iteration(self, hypPar):
        #pdb.set_trace()
        self.beforeUpdate()
        #pdb.set_trace()
        loss, gradients = self.nnLoss.gradients(self.data_tr, self.batchSize, hypPar)
        self.opt.apply_gradients(zip(gradients, self.nnLoss.getNet( ).trainable_variables))
        self.losses.append(loss)
        self.afterUpdate()
        return

    def attachDebugger(self, debug):
        self.debug = debug
        self.nets = []
        self.debug.attachTrainer(self)

    def beforeTraining(self):
        if hasattr(self, 'debug'):
            self.nets.append(self.nnLoss.copy( ))

    def beforeUpdate(self):
        if hasattr(self, 'debug'):
            self.debug.beforeUpdate(self.iter)

    def afterUpdate(self):
        if hasattr(self, 'debug'):
            self.nets.append(self.nnLoss.copy( ))
            self.debug.afterUpdate()


class AlternatingTrainer:
    def __init__(self, netPU, netDisc, data, rounds, maxIter, batchSize):
        #pdb.set_trace()
        self.rounds = rounds
        self.maxIter = maxIter
        self.batchSize = batchSize
        self.data = data
        self.data_tr, _, self.data_val = data.getTTVData()
        self.PUTrainer = Trainer(netPU, data, maxIter, batchSize)
        self.discTrainer = Trainer(netDisc, data, maxIter, batchSize)
        self.netPU = netPU
        self.netDisc = netDisc
        self.netPU.attachDiscriminator(netDisc.net)
        self.netDisc.attachPNPosteriorNet(netPU.net)
        self.hypPar = {'alpha': 0.5, 'max_p': 0.5, 'max_p0': 0.5}
        self.run = {'iter': maxIter}
        #pdb.set_trace()
        #self.debug.attachData(x,y)

    def attachDebugger(self, debug):
        self.debug = debug
        self.netPUs = []
        self.netDiscs = []
        self.debug.attachTrainer(self)

    def beforePUUpdate(self):
        if hasattr(self, 'debug'):
            self.netPUs.append(self.netPU.copy())
            self.debug.beforePUUpdate(self.round)

    def beforeDiscUpdate(self):
        if hasattr(self, 'debug'):
            self.netDiscs.append(self.netDisc.copy())
            self.debug.beforeDiscUpdate(self.round)

    def endOfRound(self):
        if hasattr(self, 'debug'):
            self.debug.endOfRound(self.round)

    def fit(self):
        for i in np.arange(self.rounds):
            self.round = i
            self.iteration()
        return

    def iteration(self):
        #pdb.set_trace()
        self.beforePUUpdate()
        if self.round > 1:
            p = self.netPU.posterior(self.data_val['x'])
            p1 = self.netPU.posterior(self.data_val['x1'])
            alpha = np.mean(p)
            max_p = np.max(np.vstack([p,p1]))
            print('max_posterior on validation set: ' + str(max_p))
            max_p0 = np.max(np.vstack([1-p, 1-p1]))
            alpha = alpha * 0.9/max_p
            #alpha = 0.27
            print('alpha used in the loss: ' + str(alpha))
            self.hypPar['alpha'] = alpha
            self.hypPar['max_p'] = max_p
            self.hypPar['max_p0'] = max_p0
        self.PUTrainer.fit(self.hypPar,  self.run)

        self.beforeDiscUpdate()
        self.discTrainer.fit(run = self.run)

        self.endOfRound()
        return