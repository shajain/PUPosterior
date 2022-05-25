import numpy
import tensorflow as tf
import numpy as np
from NN.NetWithLoss import NetWithLoss
from misc.randomSort import randomSort
from tensorflow.keras.losses import BinaryCrossentropy
from plots import sortedplot as sp
import pdb
#from tensorflow
from scipy.special import xlogy
from misc.randomSort import randomSort
from scipy.stats import cramervonmises_2samp as TwoSampleTest


class PUPosterior3(NetWithLoss):
    DEBUG = False
    def __init__(self, net):
        #Assume that x is sorted
        super(PUPosterior3, self).__init__(net)
        self.BCE =BinaryCrossentropy()
        self.alpha = 0.5
        self.alpha_ub = 1
        self.alpha_lb = 0
        self.gamma = 0.5
        self.iter = 1
        self.binarySearch = False
        self.historySize = 10
        self.historyIrr = {'train': np.zeros((self.historySize,)), 'val': np.zeros((self.historySize,))}
        self.historyIx = {'train': 0, 'val': 0}
        self.irr_prop_thr = 0.01
        self.irr_thr = 0.98
        self.nBins = 10
        self.bins = np.linspace(0.1, 0.9, self.nBins-1)


    def attachDiscriminator(self, disc):
        self.disc = disc


    def resetHistory(self):
        for set in ['train', 'val']:
            self.historyIrr[set] = np.zeros((self.historySize,))
            self.historyIx[set] = 0

    def add2History(self, irr_prop, set):
        self.historyIrr[set][self.historyIx[set] % self.historySize] = irr_prop
        self.historyIx[set] = self.historyIx[set] + 1

    def historySatisifiesIrr(self, set):
        return np.mean(self.historyIrr[set] > self.irr_prop_thr)

    def CramVonLoss(self, data):
        x = data['x']
        x1 = data['x1']
        p = self.posterior(x)
        n1 = x1.shape[0]
        n = x.shape[0]
        nULPos = np.sum(p)
        discPost1 = self.posteriorDisc(x1)
        discPost = self.posteriorDisc(x)
        y = np.vstack([np.ones((n1, 1)), np.zeros((n, 1))])
        ix_sort = randomSort(np.vstack([discPost1, discPost]))[1]
        y = y[ix_sort, :]
        ecdf_pos = np.cumsum(y) / n1
        pp = np.vstack([np.zeros((n1, 1)), p])
        pp = pp[ix_sort,:]
        ecdf_ppos = np.cumsum(pp) / nULPos
        w = np.vstack([np.ones((n1, 1)), p])
        w = w[ix_sort, :]
        lossCramVon = np.sum(w*(ecdf_ppos - ecdf_pos) ** 2)/np.sum(w)
        return lossCramVon


    def valLoss(self, data, hypPar=None):
        x = data['x']
        x1 = data['x1']
        p = self.posterior(x)
        max_p = np.max(p)
        alpha = np.mean(p)
        p1 = self.posterior(x1)
        expf1 = np.mean(p1)
        expf1hat = np.mean(p*p)/alpha
        #expf0hat = np.mean((1-p)*p)/(1-alpha)
        #beta = np.abs((expf1 - (max_p*expf1hat + (1-max_p)*expf0hat))**2
        beta = np.abs(expf1 - expf1hat)
        l1 = -(alpha + beta) * (self.gamma * np.mean(np.log(p1)) + (1 - self.gamma) * np.mean(xlogy(p, p)) / alpha)
        l0 = -np.mean(xlogy(1 - p, 1 - p))
        loss = l1 + l0
        return loss

    def loss(self, data, hypPar=None):
        x = data['x']
        x1 = data['x1']
        p = self.posterior(x)
        alpha = np.mean(p)
        p1 = self.posterior(x1)
        l1 = -alpha*(self.gamma * np.mean(np.log(p1)) + (1-self.gamma) * np.mean(xlogy(p, p))/alpha)
        l0 = -np.mean(xlogy(1-p, 1-p))
        loss = l1 + l0
        #ix = np.digitize(postTrue, bins)
        return loss


    def lossTF(self, x1, x, pmaxed1, a1, pmaxed0, a0, alpha, ix_sort, ecdf_pos, w, useCVMLoss):
        p = self.net(x)
        p1 = self.net(x1)
        l1 = -alpha * (self.gamma * tf.reduce_mean(tf.math.log(p1)) +
                      (1 - self.gamma) * tf.reduce_mean(tf.math.xlogy(pmaxed1, p))/a1)
        l0 = -(1 - alpha) * tf.reduce_mean(tf.math.xlogy(pmaxed0, 1-p))/a0
        nULPos = tf.reduce_sum(p)
        pp = tf.concat([tf.cast(tf.zeros((x1.shape[0], 1)), tf.float32), p], axis=0)
        #pdb.set_trace()
        pp = tf.gather(pp, ix_sort)
        ecdf_ppos = tf.cumsum(pp)/nULPos
        lossCramVon = tf.reduce_sum(w*(ecdf_ppos-ecdf_pos)**2)/tf.reduce_sum(w)
        #pdb.set_trace()
        loss = l1 + l0 + useCVMLoss*lossCramVon
        #loss = lossCramVon + 10*(tf.reduce_mean(p) - alpha)**2
        #pdb.set_trace()
        return loss

    def determineAlpha(self, x):
        if self.historySatisifiesIrr('train') == 1:
            self.alpha_ub = np.mean(self.posterior(x))
            self.resetHistory()
            self.alpha = (self.alpha_lb + self.alpha_ub)/2
        elif self.historySatisifiesIrr('train') == 0:
            self.alpha_lb = np.mean(self.posterior(x))
            self.resetHistory()
            self.alpha = (self.alpha_lb + self.alpha_ub)/2
        return self.alpha


    def gradients(self, data, BS, hypPar):
        x = data['x']
        x1 = data['x1']
        n = x.shape[0]
        n1 = x1.shape[0]
        self.alpha = hypPar['alpha']
        #print('alpha as posterior mean: ', self.alpha)
        max_p = hypPar['max_p']
        #print('max_p: ' + str(max_p))
        max_p0 = hypPar['max_p0']
        #print('max_p0: ' + str(max_p0))
        BSx = np.cast['int64'](BS/self.alpha)
        ix1 = np.random.choice(n1, BS, replace=True)
        ix = np.random.choice(n, BSx, replace=True)
        xx1 = x1[ix1, :]
        xx = x[ix, :]
        p = self.posterior(xx)
        pmaxed1 = p/max_p
        pmaxed1[(pmaxed1>1).flatten(), :] = 1
        pmaxed0 = (1 - p) / max_p0
        pmaxed0[(pmaxed0 > 1).flatten(), :] = 1
        a1 = np.mean(pmaxed1)
        a0 = np.mean(pmaxed0)

        discPost1 = self.posteriorDisc(xx1)
        discPost = self.posteriorDisc(xx)
        y = np.vstack([np.ones((BS, 1)), np.zeros((BSx, 1))])
        ix_sort = randomSort(np.vstack([discPost1, discPost]))[1]
        y = y[ix_sort, :]
        ecdf_pos = np.cumsum(y, axis=0)/BS
        w = np.vstack([np.ones((BS, 1)).astype('float32'), p])
        w = w[ix_sort, :]

        res = TwoSampleTest(discPost1.flatten(),
                            discPost[np.random.choice(BSx, BS, p=p.flatten()/np.sum(p)), :].flatten())
        if res.pvalue < 0.25:
            useCVMLoss = 1
        else:
            useCVMLoss = 0


        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.net.trainable_variables)
            loss = self.lossTF(xx1, xx, pmaxed1, a1, pmaxed0, a0, self.alpha, ix_sort, ecdf_pos, w, useCVMLoss)
        return loss, tape.gradient(loss, self.net.trainable_variables)

    def posterior(self, x):
        return self.net.predict(x)

    def posteriorDisc(self, x):
        return self.disc.predict(x)

    def discCDF(self, discPost, posterior=None):
        if posterior is None:
            posterior = np.ones_like(discPost)
        n = np.sum(posterior)
        discPost, ix_sort = randomSort(discPost)[0:2]
        cdf = np.cumsum(posterior[ix_sort, :])/n
        return cdf, discPost


    def copy(self):
        copy = PUPosterior3(self.copyNet())
        #pdb.set_trace()
        copy.disc = self.disc.copy()
        return copy



class Discriminator(NetWithLoss):
    DEBUG = False
    def __init__(self, net):
        #Assume that x is sorted
        super(Discriminator, self).__init__(net)
        self.BCE =BinaryCrossentropy()

    def attachPNPosteriorNet(self, pnPostNet):
        self.pnPostNet = pnPostNet

    def valLoss(self, data, hypPar=None):
        return self.loss(data)

    def loss(self, data, hypPar=None):
        x1 = data['x1']
        x = data['x']
        pnPost = self.pnPosterior(x)
        alpha = np.mean(pnPost)
        ppPost = self.posterior(x)
        ppPost1 = self.posterior(x1)
        loss = -0.5*np.mean(np.log(ppPost1)) - 0.5*np.mean(xlogy(pnPost, ppPost))/alpha
        return loss


    def lossTF(self, x, y):
        p = self.net(x)
        loss = self.BCE(y, p)
        #pdb.set_trace()
        return loss

    def gradients(self, data, BS, hypPar=None):
        x = data['x']
        x1 = data['x1']
        n1 = x1.shape[0]
        n = x.shape[0]
        ix1 = np.random.choice(n1, BS, replace=True)
        xx = x[np.random.choice(n, 5000), :]
        pnPost = self.pnPosterior(xx).flatten()
        #BSx = (BS/alpha).astype('int64')
        ix0 = np.random.choice(xx.shape[0], BS, replace=True, p=pnPost/np.sum(pnPost))
        xx1 = x1[ix1, :]
        xx0 = xx[ix0, :]
        xx = np.vstack([xx1, xx0])
        yy = np.vstack([np.ones((xx1.shape[0], 1)), np.zeros((xx0.shape[0], 1))])
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.net.trainable_variables)
            loss = self.lossTF(xx, yy)
        return loss, tape.gradient(loss, self.net.trainable_variables)

    def posterior(self, x):
        return self.net.predict(x)

    def pnPosterior(self, x):
        return self.pnPostNet.predict(x)

    def copy(self):
        copy = Discriminator(self.copyNet())
        return copy



