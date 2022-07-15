import tensorflow as tf
import numpy as np
from NN.NetWithLoss import NetWithLoss
from misc.randomSort import randomSort
from tensorflow.keras.losses import BinaryCrossentropy
from plots import sortedplot as sp
import pdb
from scipy.special import xlogy


class PUPosterior(NetWithLoss):
    DEBUG = False
    def __init__(self, net):
        #Assume that x is sorted
        super(PUPosterior, self).__init__(net)
        self.BCE =BinaryCrossentropy()
        self.alpha = 0.5
        self.gamma = 0.5
        self.iter = 1
        self.createBins(np.random.rand(100))


    def valLoss(self, data):
        loss = self.loss(data)
        return loss

    def loss(self, data):
        return self.varLoss(data)


    def varLoss(self, data):
        x = data['x']
        x1 = data['x1']
        p = self.posterior(x)
        p1 = self.posterior(x1)
        loss = np.log(np.mean(p)) - np.mean(np.log(p1))
        return loss


    def lossVariationalTF(self, x1, x, wL, wUL, pow=1):
        post = self.net(x)
        post1 = self.net(x1)
        alphaHat = tf.math.reduce_mean(wUL*(post**pow))/np.mean(wUL)

        loss = tf.math.log(alphaHat) - tf.math.reduce_mean(tf.math.xlogy(wL,post1))/np.mean(wL)
        #priorLoss = -alpha*tf.math.log(alphaHat) - (1-alpha)* tf.math.log(1-alphaHat)
        #pdb.set_trace()
        #priorLoss = tf.nn.relu(alpha - alphaHat)
        #pdb.set_trace()
        #loss = loss + 10*priorLoss
        return loss

    def lossWeightedTF2(self, x1, x, alpha, wL, wPosUL, wNegUL, modify_thr=1, pow=1):
        post = self.net(x)
        post1 = self.net(x1)
        post = tf.where(post > modify_thr, post*(modify_thr/post)**2, post)
        post1 = tf.where(post1 > modify_thr, post1 * (modify_thr / post1) ** 2, post1)
        l1 = -alpha * (self.gamma * tf.reduce_mean(tf.math.xlogy(wL,post1**pow))/np.mean(wL) +
                       (1 - self.gamma) * tf.reduce_mean(tf.math.xlogy(wPosUL, post**pow)) /np.mean(wPosUL))
        l0 = -(1 - alpha) * tf.reduce_mean(tf.math.xlogy(wNegUL, 1 - post**pow)) / np.mean(wNegUL)
        #l2 = -alpha * tf.math.log(tf.reduce_mean(p)) - (1 - alpha) * tf.math.log(tf.reduce_mean(1-p))
        loss = l1 + l0
        #pdb.set_trace()
        return loss

    def lossWeightedTF(self, x1, x, alpha, wL, wPosUL, wNegUL, pow=1):
        post = self.net(x)
        post1 = self.net(x1)
        l1 = -alpha * (self.gamma * tf.reduce_mean(tf.math.xlogy(wL,post1**pow))/np.mean(wL) +
                       (1 - self.gamma) * tf.reduce_mean(tf.math.xlogy(wPosUL, post**pow)) /np.mean(wPosUL))
        l0 = -(1 - alpha) * tf.reduce_mean(tf.math.xlogy(wNegUL, 1 - post**pow)) / np.mean(wNegUL)
        #l2 = -alpha * tf.math.log(tf.reduce_mean(p)) - (1 - alpha) * tf.math.log(tf.reduce_mean(1-p))
        loss = l1 + l0
        #pdb.set_trace()
        return loss

    def lossPriorTF(self, x, alpha):
        post = self.net(x)
        alphaHat = tf.math.reduce_mean(post)
        loss = -alpha*tf.math.log(alphaHat) - (1-alpha)* tf.math.log(1-alphaHat)
        return loss

    def lossPriorRangeTF(self, x):
        post = self.net(x)
        alphaHat = tf.math.reduce_mean(post)
        loss = tf.nn.relu(alphaHat - 0.99) + tf.nn.relu(0.01 - alphaHat)
        return loss

    def lossHistogramTF(self, x, binIX, histProbsL):
        post = self.net(x)
        nPosUL = tf.reduce_sum(post)
        histProbsUL = tf.map_fn(fn=lambda i: tf.reduce_sum(tf.gather(post, binIX[tf.cast(i, tf.int64)])) / nPosUL,
                              elems=tf.cast(tf.range(len(binIX)), tf.float32))
        #pdb.set_trace()
        # lHist = -tf.reduce_mean(tf.math.xlogy(histProbs1, histProbs))
        loss = tf.reduce_sum((histProbsUL - histProbsL) ** 2)
        return loss


    def gradients(self, data, BS, hypPar=None):
        x = data['x']
        x1 = data['x1']
        n = x.shape[0]
        n1 = x1.shape[0]
        if hypPar is None:
            self.alpha = np.mean(self.posterior(x))
        elif 'alpha' in hypPar:
            self.alpha = hypPar['alpha']
        elif 'alphaUB' in hypPar:
            self.alpha = hypPar['alphaUB']
        else:
            self.alpha = hypPar['alphaHat']



        BSx = np.cast['int64'](BS/self.alpha)
        ix1 = np.random.choice(n1, BS, replace=True)
        ix = np.random.choice(n, BSx, replace=True)
        xx1 = x1[ix1, :]
        xx = x[ix, :]

        postPosUL = self.posterior(xx)
        postNegUL = 1 - postPosUL
        postPosL = self.posterior(xx1)
        postPosAll = np.vstack((postPosUL, postPosL))

        if 'alphaHat' in hypPar:
            alphaHat = hypPar['alphaHat']
        else:
            alphaHat = np.mean(postPosUL)

        if 'postPosMax' in hypPar:
            postPosMax = hypPar['postPosMax']
            postNegMax = hypPar['postNegMax']
        else:
            postPosMax = np.max(postPosAll)
            postNegMax = np.max(1 - postPosAll)
        postPosUL = postPosUL / postPosMax
        postNegUL = postNegUL / postNegMax
        postPosL = postPosL / postPosMax
        wL = np.ones_like(postPosL)
        wUL = np.ones_like(postPosUL)
        wPosUL = postPosUL
        wNegUL = postNegUL
        if 'alphaMaxHat' in hypPar:
            alphaMaxHat = hypPar['alphaMaxHat']
        else:
            alphaMaxHat = np.mean(postPosUL)

        #if ('useWeightedLoss' in  hypPar and hypPar['useWeightedLoss']) or ('useWeightedLoss2' in  hypPar and hypPar['useWeightedLoss2']):
        if True:
            vPosUL = wPosUL * (postPosUL ** 2)
            vNegUL = wNegUL * (postPosUL ** 2)
            vL = wL * (postPosL ** 2)
            vUL = wUL * (postPosUL ** 2)
            alpha_v = np.mean(vPosUL) / np.mean(postPosUL ** 2)
            alpha_v = alpha_v * hypPar['factor_v']
            if 'additiveConst' in hypPar:
                alpha_v = min(alpha_v + hypPar['additiveConst'], 1)
            if 'weightedLossWeight' in hypPar:
                weightedLossWeight = hypPar['weightedLossWeight']
            else:
                weightedLossWeight = 0.1
            print('weighted alpha_v : ' + str(alpha_v))

        # uUL, uL = self.equalizeWeights(xx, postPosUL, xx1, postPosL)
        # uuPosUL = wPosUL * uUL
        # uuNegUL = wNegUL * uUL
        # uuL = wL*uL
        # alpha_u = (self.alpha/alphaMaxHat) * np.mean(uuPosUL) / np.mean(uUL)
        #print('weighted alpha_u : ' + str(alpha_u))

        # self.createBins(postPosL)
        # histProbsL = self.histProbs(postPosL)[0]
        # histProbsUL, binIX_UL = self.histProbs(postPosUL, postPosUL)
        # print('bins ' + str(self.bins))
        # print('histProbsL '+ str(histProbsL))
        # print('histProbsUL ' + str(histProbsUL))


        if 'priorLossWeight' in hypPar:
            priorLossWeight = hypPar['priorLossWeight']
        else:
            priorLossWeight = 0.1

        if 'useFixedPriorLoss' in hypPar:
            useFixedPriorLoss = hypPar['useFixedPriorLoss']
        else:
            useFixedPriorLoss = False


        print('posterior maximum: ' + str(postPosMax))
        print('alpha: ' + str(self.alpha))

        self.iter = self.iter + 1
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.net.trainable_variables)

            if useFixedPriorLoss:
                loss = self.lossWeightedTF(xx1, xx, self.alpha, wL, wPosUL, wNegUL)
                print('using fixed alpha Loss')
            else:
                loss = self.lossVariationalTF(xx1, xx, wL, wUL)
                print('using variational Loss')
                #loss = loss + 0.1*self.lossVariationalTF(xx1, xx, vL, vUL, hypPar['power'])
                #loss = 0.0
                loss = loss + 100*self.lossPriorRangeTF(xx)
                #loss = loss + self.lossVariationalTF(xx1, xx, uL, uUL, 0.9)
                #loss = loss + self.lossWeightedTF(xx1, xx, alpha_u, uuL, uuPosUL, uuNegUL)
                #print('using weighted loss u')
                #if postPosMax < 0.9:
                    # if self.iter < 200:
                if hypPar['useWeightedLoss']:
                #if True:
                    #pdb.set_trace()
                    loss_weighted = self.lossWeightedTF(xx1, xx, alpha_v, vL, vPosUL, vNegUL)
                    loss = loss + weightedLossWeight * loss_weighted
                    print('using weighted loss v')
                else:
                    # loss = self.lossTF(xx1, xx, alpha_v, vL, vPosUL, vNegUL)
                    print('not using weighted loss')

                if hypPar['useWeightedLoss2']:
                    # if True:
                    # pdb.set_trace()
                    loss_weighted = self.lossWeightedTF2(xx1, xx, alpha_v, vL, vPosUL, vNegUL, hypPar['modify_thr'])
                    loss = loss + hypPar['weightedLossWeight2'] * loss_weighted
                    print('using weighted loss 2')
                else:
                    # loss = self.lossTF(xx1, xx, alpha_v, vL, vPosUL, vNegUL)
                    print('not using weighted loss 2')

                if hypPar['useHistogramLoss']:
                    lossHistogram = self.lossHistogramTF(xx, binIX_UL, histProbsL)
                    loss = loss + hypPar['histogramLossWeight']*lossHistogram
                    print('using histogram loss')
                else:
                    print('not using histogram loss')

                if hypPar['usePriorLoss']:
                    lossPrior = self.lossPriorTF(xx, self.alpha)
                    loss = loss + hypPar['priorLossWeight']*lossPrior
                    print('using Prior loss')
                else:
                    print('not using Prior loss')
        return loss, tape.gradient(loss, self.net.trainable_variables)

    def posterior(self, x):
        return self.net.predict(x, verbose=0)

    def equalizeWeights(self, x, post, x1, post1):
        nBins = 10
        bins = np.linspace(np.min(post), np.max(post), nBins+1)
        bins[-1] = bins[-1] + (bins[-1] - bins[-2])/100
        bins1 = np.linspace(np.min(post1), np.max(post1), nBins+1)
        bins1[-1] = bins1[-1] + (bins1[-1] - bins1[-2]) / 100
        ix = np.digitize(post.flatten(), bins)
        ix1 = np.digitize(post1.flatten(), bins1)
        w = np.array([1/np.sum(ix == i) for i in np.arange(nBins+2)])
        w[np.isinf(w)] = 0
        w = w/np.sum(w)
        w1 = np.array([1/np.sum(ix1 == i) for i in np.arange(nBins + 2)])
        w1[np.isinf(w1)] = 0
        w1 = w1 / np.sum(w1)
        ww = 0.5 * w[ix] + 0.5*w1[np.digitize(post.flatten(), bins1)]
        ww1 = 0.5 * w1[ix1] + 0.5 * w[np.digitize(post1.flatten(), bins)]
        ww = np.expand_dims(ww, axis=1)/nBins
        ww1 = np.expand_dims(ww1, axis=1)/nBins
        #pdb.set_trace()
        return ww, ww1

    def createBins(self, postPosL):
        per5 = np.percentile(postPosL, 5)
        per95 = np.percentile(postPosL, 95)
        nBins = 10
        self.bins = np.linspace(per5, per95, nBins - 1)


    def histProbs(self, f, posterior=None):
        f = f.flatten()
        if posterior is None:
            posterior = np.ones_like(f)
        else:
            posterior = posterior.flatten()
        ix = np.digitize(f, self.bins)
        K = self.bins.size + 1
        IX = [np.where(ix == i)[0] for i in np.arange(K)]
        n = np.sum(posterior)
        histProb = np.array([np.sum(posterior[iix]) / n for iix in IX])
        return histProb, IX


    def copy(self):
        copy = PUPosterior(self.copyNet())
        copy.alpha = np.copy(self.alpha)
        copy.bins = np.copy(self.bins)
        return copy

