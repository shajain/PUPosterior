import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from PUPosteriorSearch.Net import Discriminator
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
        self.bestNNLoss_ix = np.nan

    def loss(self):
        return self.nnLoss.loss(self.data)

    def fit(self, hypPar=None):
        self.beforeTraining()
        for i in np.arange(self.maxIter):
            self.iter = i
            self.iteration(hypPar)
            valLoss = self.nnLoss.valLoss(self.data_val, hypPar)
            self.valLosses.append(valLoss)
            if valLoss < self.bestValLoss:
                self.bestValLoss = valLoss
                self.bestNNLoss = self.nnLoss.copy()
                self.bestNNLoss_ix = len(self.valLosses)-1
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
    def __init__(self, netPU, netDisc, data, hypPar, rounds, maxIter, batchSize):
        #pdb.set_trace()
        self.rounds = rounds
        self.maxIter = maxIter
        self.batchSize = batchSize
        self.hypPar = hypPar
        self.data = data
        self.data_tr, _, self.data_val = data.getTTVData()
        self.PUTrainer = Trainer(netPU, data, maxIter, batchSize)
        self.discTrainer = Trainer(netDisc, data, maxIter, batchSize)
        self.netPU = netPU
        self.netDisc = netDisc
        self.finalNetPU = None
        self.finalNetDisc = None
        self.netPU.attachDiscriminator(netDisc.net)
        self.netDisc.attachPNPosteriorNet(netPU.net)
        self.bestAlpha = np.nan
        self.bestPost1_max = np.max
        print('initialized trainer with'+ str(hypPar))

        #pdb.set_trace()
        #self.debug.attachData(x,y)

    def attachDebugger(self, debug):
        self.debug = debug
        self.netPUs = []
        self.netDiscs = []
        self.debug.attachTrainer(self)

    def afterPUUpdate(self):
        if hasattr(self, 'debug'):
            self.netPUs.append(self.netPU.copy())
            self.debug.afterPUUpdate(self.round)

    def afterDiscUpdate(self):
        if hasattr(self, 'debug'):
            self.netDiscs.append(self.netDisc.copy())
            self.debug.afterDiscUpdate(self.round)

    def endOfRound(self):
        if hasattr(self, 'debug'):
            self.debug.endOfRound(self.round)

    def fit(self):
        for i in np.arange(self.rounds):
            self.round = i
            #if len(self.PUTrainer.valLosses) > 100 and min(self.PUTrainer.valLosses[-100:]) > self.PUTrainer.bestValLoss:
            #    break
            self.iteration()
        self.finalNetPU = self.PUTrainer.bestNNLoss.copy()
        discAtBest = Discriminator(self.finalNetPU.disc)
        discAtBest.attachPNPosteriorNet(self.finalNetPU.net)
        #pdb.set_trace()
        print('Running 10 iterations of the discriminator for best PUPosterior Net')
        self.finalDiscTrainer = Trainer(discAtBest, self.data, 10, self.batchSize)
        #for i in np.arange(100):
            #if len(self.finalDiscTrainer.valLosses) > 100 and \
            #        min(self.finalDiscTrainer.valLosses[-100:]) > self.finalDiscTrainer.bestValLoss:
            #    break
        self.finalDiscTrainer.fit()
        self.finalNetDisc = self.finalDiscTrainer.bestNNLoss
        self.finalNetDisc.attachPNPosteriorNet(self.finalNetPU.net)
        self.finalNetPU.attachDiscriminator(self.finalNetDisc.net)
        #pdb.set_trace()
        self.afterDiscUpdate()
        return

    def iteration(self):
        #pdb.set_trace()
        alpha, post1_max, post0_max = self.netPU.posteriorMeanAndMax(self.data_val)
        if 'irr_posterior' in self.hypPar:
            alpha = alpha * self.hypPar['irr_posterior']/post1_max
        elif 'alpha' in self.hypPar:
            alpha = self.hypPar['alpha']
        hypPar = {'alpha': alpha, 'post1_max': post1_max, 'post0_max': post0_max}
        self.PUTrainer.fit(hypPar)
        self.afterPUUpdate()
        #self.discTrainer.fit()
        self.afterDiscUpdate()
        self.endOfRound()
        return


class TrainerTenMedian:
    def __init__(self, netPU, netDisc, data, rounds, maxIter, batchSize):

        self.rounds = rounds
        self.maxIter = maxIter
        self.batchSize = batchSize
        self.data = data
        self.data_tr, _, self.data_val = data.getTTVData()
        # self.hypPar = [{'irr_posterior': 0.8}]
        #self.hypPar = [{'alpha': 0.5}]
        #self.altTrainer = [AlternatingTrainer(netPU, netDisc, data, self.hypPar[0],
        #                                      rounds, maxIter, batchSize)]
        netPU.attachDiscriminator(netDisc.net)
        self.netPU = netPU
        self.netDisc = netDisc
        self.hypPar = []
        self.altTrainer = []
        self.bestNetPU = []
        self.alpha = []
        self.post1_max = []
        self.ub = np.nan
        self.lb = np.nan
        self.ballPark = np.nan
        self.pvalue_thr = 0.5
        self.exactBallParkNotFound = True
        self.finalAlpha = np.nan

    def fit(self):
        alpha = 0.5
        #pdb.set_trace()
        while np.isnan(self.ballPark) and (np.isnan(self.ub) or np.isnan(self.lb) or np.abs(self.alpha[self.ub] - self.alpha[self.lb]) > 0.01):
            netPU = self.netPU.copy()
            netPU.net = netPU.net.new()
            netDisc = self.netDisc.copy()
            netDisc.net = netDisc.net.new()
            self.hypPar.append({'alpha': alpha})
            self.altTrainer.append(AlternatingTrainer(netPU, netDisc, self.data, self.hypPar[-1], self.rounds, self.maxIter, self.batchSize))
            self.beforeAltTrainerFitting(self.altTrainer[-1])
            self.altTrainer[-1].fit()
            self.bestNetPU.append(self.altTrainer[-1].finalNetPU)
            self.alpha.append(self.bestNetPU[-1].alpha)
            self.post1_max.append(self.bestNetPU[-1].posteriorMeanAndMax(self.data_val)[1])
            if self.post1_max[-1] > 0.9:
                self.ub = len(self.bestNetPU)-1
                if np.isnan(self.lb):
                    alpha = self.alpha[-1] / 2
                else:
                    alpha = (self.alpha[self.ub] + self.alpha[self.lb]) / 2
            elif self.post1_max[-1] < 0.8:
                self.lb = len(self.bestNetPU)-1
                if np.isnan(self.ub):
                    alpha = (1 + self.alpha[-1])/2
                else:
                    alpha = (self.alpha[self.ub] + self.alpha[self.lb]) / 2
            else:
                self.ballPark = len(self.bestNetPU)-1
                self.exactBallParkNotFound = False
        if np.isnan(self.ballPark):
            post1_max = np.array(self.post1_max)
            ix_lb = post1_max < 0.9
            alphas = np.array(self.alpha)
            alpha = np.max(alphas[ix_lb])
        else:
            alpha = self.alpha[self.ballPark]

        for i in np.arange(10):
            netPU = self.netPU.copy()
            netPU.net = netPU.net.new()
            netDisc = self.netDisc.copy()
            netDisc.net = netDisc.net.new()
            self.hypPar.append({'alpha': alpha})
            data = self.data.reshuffle()
            self.altTrainer.append(AlternatingTrainer(netPU, netDisc, data, self.hypPar[-1], self.rounds, self.maxIter, self.batchSize))
            self.beforeAltTrainerFitting(self.altTrainer[-1])
            self.altTrainer[-1].fit()
            self.bestNetPU.append(self.altTrainer[-1].finalNetPU)
            self.alpha.append(self.bestNetPU[-1].alpha)
            self.post1_max.append(self.bestNetPU[-1].posteriorMeanAndMax(self.data_val)[1])

        post1_max = np.array(self.post1_max)
        alphas = np.array(self.alpha)

        ix_lbs = alphas == alpha
        self.finalAlpha = np.median(alphas[ix_lbs]/post1_max[ix_lbs])

        netPU = self.netPU.copy()
        netPU.net = netPU.net.new()
        netDisc = self.netDisc.copy()
        netDisc.net = netDisc.net.new()
        self.hypPar.append({'alpha': self.finalAlpha})
        data = self.data.reshuffle()
        self.altTrainer.append(AlternatingTrainer(netPU, netDisc, data, self.hypPar[-1], self.rounds, self.maxIter, self.batchSize))
        self.beforeAltTrainerFitting(self.altTrainer[-1])
        self.altTrainer[-1].fit()
        self.bestNetPU.append(self.altTrainer[-1].finalNetPU)
        self.alpha.append(self.bestNetPU[-1].alpha)
        self.post1_max.append(self.bestNetPU[-1].posteriorMeanAndMax(self.data_val)[1])
        self.finalNetPU = self.bestNetPU[-1]
        return self.finalNetPU, self.finalAlpha


    def attachDebugger(self, debug):
        self.debug = debug

    def beforeAltTrainerFitting(self, altTrainer):
        if hasattr(self, 'debug'):
            altTrainer.attachDebugger(self.debug)


# class BinarySearchTrainer:
#     def __init__(self, netPU, netDisc, data, rounds, maxIter, batchSize):
#
#         self.rounds = rounds
#         self.maxIter = maxIter
#         self.batchSize = batchSize
#         self.data = data
#         self.data_tr, _, self.data_val = data.getTTVData()
#         #self.hypPar = [{'irr_posterior': 0.8}]
#         self.hypPar = [{'alpha': 0.3}]
#         self.altTrainer = [AlternatingTrainer(netPU, netDisc, data, self.hypPar[0],
#                                         rounds, maxIter, batchSize)]
#         self.bestNetPU = []
#         self.alpha = []
#         self.post1_max = []
#         self.ub = np.nan
#         self.lb = np.nan
#         self.pvalue_thr = 0.5
#
#     def fit(self):
#         self.beforeAltTrainerFitting(self.altTrainer[0])
#         self.altTrainer[0].fit()
#         self.bestNetPU.append(self.altTrainer[0].finalNetPU)
#         self.alpha.append(self.bestNetPU[0].alpha)
#         self.post1_max.append(self.bestNetPU[0].posteriorMeanAndMax(self.data_val)[1])
#         pvalue = self.bestNetPU[-1].CramVonLoss(self.data_val)[1]
#         if pvalue > self.pvalue_thr or self.post1_max[-1] < 0.9:
#             if np.isnan(self.lb) and pvalue <= self.pvalue_thr:
#                 self.pvalue_thr = pvalue
#             self.lb = 0
#         else:
#             self.ub = 0
#         i = 0
#         while np.isnan(self.lb) or np.isnan(self.ub) or self.alpha[self.ub] - self.alpha[self.lb] > 0.01:
#             #pdb.set_trace()
#             # if i == 0:
#             #    newAlpha = self.alpha[-1]
#             #elif
#             if np.isnan(self.lb):
#                 newAlpha = self.alpha[-1] - 0.05
#             elif np.isnan(self.ub):
#                 newAlpha = self.alpha[-1]/self.post1_max[-1]
#                 if i > 1:
#                     newAlpha = newAlpha + 0.05
#             else:
#                 newAlpha = (self.alpha[self.ub] + self.alpha[self.lb])/2
#
#             self.hypPar.append({'alpha': newAlpha})
#             disc = Discriminator(self.bestNetPU[-1].disc.copy())
#             self.altTrainer.append(AlternatingTrainer(self.bestNetPU[-1].copy(), disc, self.data, self.hypPar[-1],
#                                                       self.rounds, self.maxIter, self.batchSize))
#             self.beforeAltTrainerFitting(self.altTrainer[-1])
#             self.altTrainer[-1].fit()
#             self.bestNetPU.append(self.altTrainer[-1].finalNetPU)
#             self.alpha.append(self.bestNetPU[-1].alpha)
#             self.post1_max.append(self.bestNetPU[-1].posteriorMeanAndMax(self.data_val)[1])
#             pvalue = self.bestNetPU[-1].CramVonLoss(self.data_val)[1]
#             if pvalue > self.pvalue_thr or self.post1_max[-1] < 0.9:
#                 if np.isnan(self.lb) and pvalue <= self.pvalue_thr:
#                     self.pvalue_thr = pvalue
#                 self.lb = len(self.bestNetPU)
#             else:
#                 self.ub = len(self.bestNetPU)
#             i = i+1
#
#     def attachDebugger(self, debug):
#         self.debug = debug
#
#     def beforeAltTrainerFitting(self, altTrainer):
#         if hasattr(self, 'debug'):
#             altTrainer.attachDebugger(self.debug)


# class TrainerTenMedian:
#     def __init__(self, netPU, netDisc, data, rounds, maxIter, batchSize):
#
#         self.rounds = rounds
#         self.maxIter = maxIter
#         self.batchSize = batchSize
#         self.data = data
#         self.data_tr, _, self.data_val = data.getTTVData()
#         # self.hypPar = [{'irr_posterior': 0.8}]
#         self.hypPar = [{'alpha': 0.5}]
#         self.altTrainer = [AlternatingTrainer(netPU, netDisc, data, self.hypPar[0],
#                                               rounds, maxIter, batchSize)]
#         self.bestNetPU = []
#         self.alpha = []
#         self.post1_max = []
#         self.ub = np.nan
#         self.lb = np.nan
#         self.pvalue_thr = 0.5
#
#     def fit(self):
#         self.beforeAltTrainerFitting(self.altTrainer[0])
#         self.altTrainer[0].fit()
#         self.bestNetPU.append(self.altTrainer[0].finalNetPU)
#         self.alpha.append(self.bestNetPU[0].alpha)
#         self.post1_max.append(self.bestNetPU[0].posteriorMeanAndMax(self.data_val)[1])
#         if self.post1_max[-1] > 0.85:
#             self.ub = 0
#         else:
#             self.lb = 0
#         i = 0
#         while np.isnan(self.lb) or np.isnan(self.ub) or self.alpha[self.ub] - self.alpha[self.lb] > 0.01:
#             if np.isnan(self.lb):
#                 newAlpha = self.alpha[-1]/2
#             elif np.isnan(self.ub):
#                 newAlpha = max(self.alpha[-1] * 2, 0.99)
#             else:
#                 newAlpha = (self.alpha[self.ub] + self.alpha[self.lb]) / 2
#             self.hypPar.append({'alpha': newAlpha})
#             disc = Discriminator(self.bestNetPU[-1].disc.new())
#             netPU = self.bestNetPU[-1].copy()
#             netPU.net = netPU.net.new()
#             self.altTrainer.append(AlternatingTrainer(netPU, disc, self.data, self.hypPar[-1],
#                                                       self.rounds, self.maxIter, self.batchSize))
#             self.beforeAltTrainerFitting(self.altTrainer[-1])
#             self.altTrainer[-1].fit()
#             self.bestNetPU.append(self.altTrainer[-1].finalNetPU)
#             self.alpha.append(self.bestNetPU[-1].alpha)
#             self.post1_max.append(self.bestNetPU[-1].posteriorMeanAndMax(self.data_val)[1])
#             if self.post1_max[-1] > 0.85:
#                 self.ub = len(self.bestNetPU)-1
#             else:
#                 self.lb = len(self.bestNetPU)-1
#             i = i + 1
#         newAlpha =  (self.alpha[self.ub] + self.alpha[self.lb]) / 2
#         newAlphas = (np.arange(1,11,1)/10)*newAlpha
#
#         for newAlpha in newAlphas:
#             self.hypPar.append({'alpha': newAlpha})
#             disc = Discriminator(self.bestNetPU[-1].disc.copy())
#             netPU = self.bestNetPU[-1].copy()
#             netPU.net = netPU.net.new()
#             self.altTrainer.append(AlternatingTrainer(netPU, disc, self.data, self.hypPar[-1],
#                                                       self.rounds, self.maxIter, self.batchSize))
#             self.beforeAltTrainerFitting(self.altTrainer[-1])
#             self.altTrainer[-1].fit()
#             self.bestNetPU.append(self.altTrainer[-1].finalNetPU)
#             self.alpha.append(self.bestNetPU[-1].alpha)
#             self.post1_max.append(self.bestNetPU[-1].posteriorMeanAndMax(self.data_val)[1])
#
#         post1_max = np.array(self.post1_max)
#         alphas = np.array(self.alpha)
#         ix_lbs = post1_max<0.85
#         alpha = np.median(alphas[ix_lbs]/post1_max[ix_lbs])
#         self.hypPar.append({'alpha': newAlpha})
#         disc = Discriminator(self.bestNetPU[-1].disc.copy())
#         self.altTrainer.append(AlternatingTrainer(self.bestNetPU[-1].copy(), disc, self.data, self.hypPar[-1],
#                                                   self.rounds, self.maxIter, self.batchSize))
#         self.beforeAltTrainerFitting(self.altTrainer[-1])
#         self.altTrainer[-1].fit()
#         self.bestNetPU.append(self.altTrainer[-1].finalNetPU)
#         self.alpha.append(self.bestNetPU[-1].alpha)
#         self.post1_max.append(self.bestNetPU[-1].posteriorMeanAndMax(self.data_val)[1])
#         return self.altTrainer[-1].finalNetPU, alpha
#
#
#     def attachDebugger(self, debug):
#         self.debug = debug
#
#     def beforeAltTrainerFitting(self, altTrainer):
#         if hasattr(self, 'debug'):
#             altTrainer.attachDebugger(self.debug)