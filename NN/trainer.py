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
        self.xCombVal = np.vstack((self.data_val['x'], self.data_val['x1']))
        self.data = data
        self.batchSize = batchSize
        self.maxIter = maxIter
        self.bestNNLoss = nnLoss
        self.losses = []
        self.valLosses = []
        self.bestValLoss = np.inf

    def loss(self):
        return self.nnLoss.loss(self.data)

    def fit(self, hypPar=None):
        self.beforeTraining()
        for i in np.arange(self.maxIter):
            # if len(self.valLosses) >= 100 and min(self.valLosses[-100:]) > self.bestValLoss:
            #     break
            self.iter = i
            postPosUL = self.nnLoss.posterior(self.data_val['x'])
            postPosL = self.nnLoss.posterior(self.data_val['x1'])
            alphaHat = np.mean(postPosUL)
            postPosMax = np.max(np.vstack((postPosUL, postPosL)))
            postNegMax = np.max(1-np.vstack((postPosUL, postPosL)))
            alphaMaxHat = np.mean(postPosUL/postPosMax)
            if hypPar is None:
                hypPar = dict()

            hypPar['postPosMax'] = postPosMax
            hypPar['postNegMax'] = postNegMax
            hypPar['alphaHat'] = alphaHat
            hypPar['alphaMaxHat'] = alphaMaxHat

            self.iteration(hypPar)
            valLoss = self.nnLoss.valLoss(self.data_val)
            self.valLosses.append(valLoss)
            if valLoss < self.bestValLoss and postPosMax > 0.9:
                self.bestValLoss = valLoss
                self.bestNNLoss = self.nnLoss.copy()
        return

    def iteration(self, hypPar):
        self.beforeUpdate()
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


class PriorAdaptivePUTrainer(Trainer):
    def __int__(self, *args):
        super(PriorAdaptivePUTrainer, self).__init__(*args)

    def fit(self):
        self.beforeTraining()
        for i in np.arange(self.maxIter):
            # if len(self.valLosses) >= 100 and min(self.valLosses[-100:]) > self.bestValLoss:
            #     break
            postByPriorMax, postByPriorMin = self.nnLoss.posteriorByPriorMaxMin(self.data_val)

            if postByPriorMax > 1 and postByPriorMax - postByPriorMin > 0.5:
                alpha = min(1/postByPriorMax, 0.99)
            else:
                alpha = 0.5

            alpha = 0.35

            print(postByPriorMin)
            print(postByPriorMax)
            print(alpha)
            print('======')
            hypPar = {'alpha': alpha}
            self.iter = i
            self.iteration(hypPar)
            valLoss = self.nnLoss.valLoss(self.data_val)
            self.valLosses.append(valLoss)
            if valLoss < self.bestValLoss:
                self.bestValLoss = valLoss
                self.bestNNLoss = self.nnLoss.copy()
        return

    def iteration(self, hypPar):
        self.beforeUpdate()
        loss, gradients = self.nnLoss.gradients(self.data_tr, self.batchSize, hypPar)
        self.opt.apply_gradients(zip(gradients, self.nnLoss.getNet().trainable_variables))
        self.losses.append(loss)
        self.afterUpdate()
        return



class AlternatingTrainer:
    def __init__(self, netPU, netDisc, data, rounds, maxIter, batchSize):
        #pdb.set_trace()
        self.rounds = rounds
        self.maxIter = maxIter
        self.batchSize = batchSize
        self.PUTrainer = Trainer(netPU, data, maxIter, batchSize)
        self.discTrainer = Trainer(netDisc, data, maxIter, batchSize)
        self.netPU = netPU
        self.netDisc = netDisc
        self.netPU.attachDiscriminator(netDisc.net)
        self.netDisc.attachPNPosteriorNet(netPU.net)
        #pdb.set_trace()
        #self.debug.attachData(x,y)

    def attachDebugger(self, debug):
        self.debug = debug
        self.netPUs = []
        self.netDiscs = []
        self.debug.attachTrainer(self)

    def beforePUUpdate(self):
        #pdb.set_trace()
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
        self.PUTrainer.fit( )

        self.beforeDiscUpdate()
        self.discTrainer.fit( )

        self.endOfRound()
        return