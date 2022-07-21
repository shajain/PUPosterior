import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import pdb
from NN.trainer import Trainer

class TrainerVariationalOld(Trainer):
    def __init__(self, nnLoss, data, maxIter, batchSize):
        #pdb.set_trace()
        super(TrainerVariational, self).__init__(nnLoss, data, maxIter, batchSize)
        self.initialRunFlag = True
        return


    def fit(self, hypPar=None):
        self.beforeTraining()
        for i in np.arange(self.maxIter):
            # if len(self.valLosses) >= 100 and min(self.valLosses[-100:]) > self.bestValLoss:
            #     break
            self.iter = i
            postPosUL = self.nnLoss.posterior(self.data_val['x']).flatten()
            postPosL = np.sort(self.nnLoss.posterior(self.data_val['x1']).flatten())
            postPosAll = np.hstack((postPosUL,postPosL))
            alphaHat = np.mean(postPosUL)
            postPosMax = np.max(postPosAll)
            postNegMax = np.max(1-postPosAll)
            alphaMaxHat = np.mean(postPosUL/postPosMax)
            postPosL50 = postPosL[-50]
            print('postPostL50Prev '+ str(postPosL50))
            postPosL100 = postPosL[-100]
            if hypPar is None:
                hypPar = dict()
            #pdb.set_trace()
            hypPar['postPosMax'] = postPosMax
            hypPar['postNegMax'] = postNegMax
            hypPar['alphaHat'] = alphaHat
            hypPar['alphaMaxHat'] = alphaMaxHat
            hypPar['initialRunFlag'] = self.initialRunFlag
            if not self.initialRunFlag:
                if postPosL50 > 0.98:
                    postPosL50UB = 0.1
                else:
                    postPosL50UB = (alphaHat/alphaHatPrev)*postPosL50Prev
                postPosL50LB = postPosL50UB*0.9
                hypPar['factor_v'] = postPosL50UB/(postPosL50/postPosMax)
                hypPar['useWeightedLoss'] = postPosL50 < postPosL50LB
            else:
                hypPar['factor_v'] = 1
                hypPar['useWeightedLoss'] = postPosMax < 0.98

            print('hypPar '+ str(hypPar))

            self.iteration(hypPar)


            alphaHatPrev = alphaHat
            postPosL50Prev = postPosL50
            valLoss = self.nnLoss.valLoss(self.data_val)
            self.valLosses.append(valLoss)
            if valLoss < self.bestValLoss and ((self.initialRunFlag and postPosMax > 0.98) or (not self.initialRunFlag and postPosL50 > postPosL50LB)):
                self.bestValLoss = valLoss
                self.bestNNLoss = self.nnLoss.copy()
            self.iter = self.iter + 1
            if self.initialRunFlag and (postPosL[-1] > 0.98 and np.mean(postPosL / postPosMax) > np.mean(np.mean((1 - postPosL) / postNegMax))):
                self.initialRunFlag = False
                pdb.set_trace()
        return


class TrainerVariational(Trainer):
    def __init__(self, nnLoss, data, maxIter, batchSize):
        #pdb.set_trace()
        super(TrainerVariational, self).__init__(nnLoss, data, maxIter, batchSize)
        self.initialRunFlag = False
        self.runWithFixedPriorFlag = False
        self.postMaxLB = 0.99
        return


    def fit(self, hypPar=None):
        self.beforeTraining()
        self.iter = 0
        while self.iter < self.maxIter:
            # if len(self.valLosses) >= 100 and min(self.valLosses[-100:]) > self.bestValLoss:
            #     break
            #self.iter = i
            postPosUL = self.nnLoss.posterior(self.data_val['x']).flatten()
            postPosL = np.sort(self.nnLoss.posterior(self.data_val['x1']).flatten())
            postPosAll = np.hstack((postPosUL,postPosL))
            alphaHat = np.mean(postPosUL)
            postPosMax = np.max(postPosAll)
            postNegMax = np.max(1-postPosAll)
            alphaMaxHat = np.mean(postPosUL/postPosMax)
            if hypPar is None:
                hypPar = dict()
                hypPar['usePriorLoss']=False
            elif 'alpha' in hypPar:
                hypPar['usePriorLoss'] = True
            else:
                hypPar['usePriorLoss'] = False




            #pdb.set_trace()
            hypPar['postPosMax'] = postPosMax
            hypPar['postNegMax'] = postNegMax
            hypPar['alphaHat'] = alphaHat
            hypPar['alphaMaxHat'] = alphaMaxHat
            hypPar['initialRunFlag'] = self.initialRunFlag
            hypPar['useWeightedLoss'] = False
            hypPar['weightedLossWeight'] = 0.1
            hypPar['additiveConst'] = 0
            hypPar['useFixedPriorLoss'] = self.runWithFixedPriorFlag
            hypPar['priorLossWeight'] = 0.1
            hypPar['useHistogramLoss'] = False
            hypPar['histogramLossWeight'] = 1.0
            hypPar['modify_thr'] = np.percentile(postPosUL, 90)
            hypPar['useWeightedLoss2'] = False
            hypPar['weightedLossWeight2'] = 1.0
            hypPar['power'] = 1 - (1 - postPosMax)**3


            if self.initialRunFlag or self.runWithFixedPriorFlag:
                if self.initialRunFlag:
                    #hypPar['factor_v'] = 1
                    hypPar['factor_v'] = alphaHat / alphaMaxHat
                    if postPosMax < self.postMaxLB:
                        hypPar['weightedLossWeight'] = 0.5
                    else:
                        hypPar['useWeightedLoss'] = False
                        hypPar['weightedLossWeight'] = 0
                else:
                    hypPar['useWeightedLoss'] = False
                    hypPar['weightedLossWeight'] = 0
                    #self.alphaUB = alphaHat
                    hypPar['alphaUB'] = self.alphaUB
            else:
                if 'alpha' not in hypPar:
                    hypPar['factor_v'] = alphaHat/alphaMaxHat
                    if postPosMax < self.postMaxLB and np.random.rand() < 0.75:
                            hypPar['additiveConst'] =  alphaHat * (1 - postPosMax)
                else:
                    hypPar['factor_v'] = hypPar['alpha'] / alphaMaxHat
                    hypPar['additiveConst'] = 0


            print('hypPar '+ str(hypPar))

            self.iteration(hypPar)

            valLoss = self.nnLoss.valLoss(self.data_val)
            self.valLosses.append(valLoss)
            if valLoss < self.bestValLoss:
                self.bestValLoss = valLoss
                self.bestNNLoss = self.nnLoss.copy()
                run = 0
            else:
                run = run + 1
                if run >= 50:
                    run = 0
                    if self.initialRunFlag:
                        self.initialRunFlag = False
                        self.runWithFixedPriorFlag = True
                        self.alphaUB = alphaHat
                        self.nnLoss = self.bestNNLoss.copy()
                        self.maxIter = self.maxIter + self.iter
                        pdb.set_trace()
                    elif self.runWithFixedPriorFlag:
                        self.runWithFixedPriorFlag = False
                        self.nnLoss = self.bestNNLoss.copy()
                        self.maxIter = self.maxIter + self.iter
                        postPosUL = self.nnLoss.posterior(self.data_val['x']).flatten()
                        postPosL = np.sort(self.nnLoss.posterior(self.data_val['x1']).flatten())
                        postPosAll = np.hstack((postPosUL, postPosL))
                        self.postMaxLB = np.max(postPosAll)*0.9
                        print('postMaxLB: '+ str(self.postMaxLB))
                        pdb.set_trace()

            self.iter = self.iter + 1
            # if self.initialRunFlag and (postPosL[-1] > 0.98 and np.mean(postPosL / postPosMax) > np.mean(np.mean((1 - postPosL) / postNegMax))):
            #     self.initialRunFlag = False
            #     pdb.set_trace()
        return




