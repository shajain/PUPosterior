import tensorflow as tf
import numpy as np
from NN.NetWithLoss import NetWithLoss
from misc.randomSort import randomSort
from tensorflow.keras.losses import BinaryCrossentropy
from plots import sortedplot as sp
import pdb
#from tensorflow
from scipy.special import xlogy


class PUPosterior2(NetWithLoss):
    DEBUG = False
    def __init__(self, net, alpha = None):
        #Assume that x is sorted
        super(PUPosterior2, self).__init__(net)
        self.BCE =BinaryCrossentropy()
        if alpha is None:
            self.alpha = 0.5
        else:
            self.alpha = alpha
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

    def histLoss(self, data):
        x = data['x']
        x1 = data['x1']
        discPost1 = self.posteriorDisc(x1)
        discPost = self.posteriorDisc(x)
        ix1 = np.digitize(discPost1.flatten(), self.bins)
        histProbs1 = np.array([np.mean(ix1 == i) for i in np.arange(self.nBins)])
        ix = np.digitize(discPost.flatten(), self.bins)
        histProbs = np.array([np.mean(ix == i) for i in np.arange(self.nBins)])
        lHist = -np.mean(xlogy(histProbs1, histProbs))
        return lHist


    def valLoss(self, data):
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

    def loss(self, data):
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


    def lossTF(self, x1, x, pmaxed1, a1, pmaxed0, a0, alpha, binIx, histProbs1):
        p = self.net(x)
        p1 = self.net(x1)
        l1 = -alpha * (self.gamma * tf.reduce_mean(tf.math.log(p1)) +
                      (1 - self.gamma) * tf.reduce_mean(tf.math.xlogy(pmaxed1, p))/a1)
        l0 = -(1 - alpha) * tf.reduce_mean(tf.math.xlogy(pmaxed0, 1-p))/a0
        nULPos = tf.reduce_sum(p)
        #histProbs = tf.concat([tf.reduce_sum(tf.gather(p, ix))/nULPos for ix in binIx], axis=0)
        histProbs = tf.map_fn(fn=lambda i: tf.reduce_sum(tf.gather(p, binIx[tf.cast(i, tf.int64)]))/nULPos,
                              elems=tf.cast(tf.range(self.nBins), tf.float32))
        #pdb.set_trace()
        #lHist = -tf.reduce_mean(tf.math.xlogy(histProbs1, histProbs))
        lHist = tf.reduce_mean((histProbs-histProbs1)**2)
        loss = l1 + l0 + lHist
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

    def determineAlpha2(self, max_p, alpha):
        print('max_p' + str(max_p))
        if max_p > 0.99:
            print('decrease alpha mode')
            alpha = alpha - 0.01
        print('alpha' + str(alpha))
        return alpha

    def gradients(self, data, BS):
        x = data['x']
        x1 = data['x1']
        n = x.shape[0]
        n1 = x1.shape[0]
        BSx = np.cast['int64'](BS/self.alpha)
        ix1 = np.random.choice(n1, BS, replace=True)
        ix = np.random.choice(n, BSx, replace=True)
        xx1 = x1[ix1, :]
        xx = x[ix, :]
        p = self.posterior(xx)
        max_p = np.max(p)
        pmaxed1 = p/max_p
        pmaxed0 = (1 - p) / np.max(1 - p)
        a1 = np.mean(pmaxed1)
        a0 = np.mean(pmaxed0)
        #max_p_all = np.max(self.posterior(x))
        #print('max_p_all' + str(max_p_all))
        pxx1 = self.posterior(xx1)
        print('max_p: ' + str(np.max(pxx1)))
        irr_prop = np.mean(pxx1 > self.irr_thr)
        #print('irr_prop: ' + str(irr_prop))
        self.add2History(irr_prop, 'train')
        if (not self.binarySearch) and self.historySatisifiesIrr('train') == 1:
            self.binarySearch = True
        if not self.binarySearch:
            self.alpha = a1 / (a1 + a0)
        else:
            self.alpha = self.determineAlpha(x)
        #self.alpha = 0.1094
        self.alpha = 0.35
        print('alpha: ' + str(self.alpha))
        #print('alpha_ub: ' + str(self.alpha_ub))
        #print('alpha_lb: ' + str(self.alpha_lb))
        #self.alpha = a1 / (a1 + a0)

        #pdb.set_trace()
        discPost1 = self.posteriorDisc(xx1)
        discPost = self.posteriorDisc(xx)
        disc = np.vstack([discPost1, discPost[np.random.choice(BSx, BS, p=p.flatten()/np.sum(p)), :]])
        per5 = np.percentile(disc, 5)
        per95 = np.percentile(disc, 95)
        self.bins = np.linspace(per5, per95, self.nBins-1)
        histProbs1 = self.histProbs(discPost1)
        ix = np.digitize(discPost.flatten(), self.bins)
        binIX = [np.where(ix==i)[0] for i in np.arange(self.nBins)]
        self.iter = self.iter + 1
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.net.trainable_variables)
            loss = self.lossTF(xx1, xx, pmaxed1, a1, pmaxed0, a0, self.alpha, binIX, histProbs1)
        return loss, tape.gradient(loss, self.net.trainable_variables)

    def posterior(self, x):
        return self.net.predict(x, verbose=0)

    def posteriorDisc(self, x):
        return self.disc.predict(x, verbose=0)

    def histProbs(self, discPost, posterior=None):
        if posterior is None:
            posterior = np.ones_like(discPost)
        ix = np.digitize(discPost.flatten(), self.bins)
        n = np.sum(posterior)
        histProb = np.array([np.sum(posterior[ix == i])/n for i in np.arange(self.nBins)])
        return histProb


    def copy(self):
        copy = PUPosterior2(self.copyNet())
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

    def valLoss(self, data):
        return self.loss(data)

    def loss(self, data):
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

    def gradients(self, data, BS):
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
        return self.net.predict(x, verbose=0)

    def pnPosterior(self, x):
        return self.pnPostNet.predict(x, verbose=0)

    def copy(self):
        copy = Discriminator(self.copyNet())
        return copy



