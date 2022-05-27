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


    def resetHistory(self):
        for set in ['train', 'val']:
            self.historyIrr[set] = np.zeros((self.historySize,))
            self.historyIx[set] = 0

    def add2History(self, irr_prop, set):
        self.historyIrr[set][self.historyIx[set] % self.historySize] = irr_prop
        self.historyIx[set] = self.historyIx[set] + 1

    def historySatisifiesIrr(self, set):
        return np.mean(self.historyIrr[set] > self.irr_prop_thr)

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
        return loss


    def lossTF(self, x1, x, pmaxed1, a1, pmaxed0, a0, alpha):
        p = self.net(x)
        p1 = self.net(x1)
        l1 = -alpha * (self.gamma * tf.reduce_mean(tf.math.log(p1)) +
                      (1 - self.gamma) * tf.reduce_mean(tf.math.xlogy(pmaxed1, p))/a1)
        l0 = -(1 - alpha) * tf.reduce_mean(tf.math.xlogy(pmaxed0, 1-p))/a0
        #l2 = -alpha * tf.math.log(tf.reduce_mean(p)) - (1 - alpha) * tf.math.log(tf.reduce_mean(1-p))
        loss = l1 + l0
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
        #alpha = 0.35
        #pdb.set_trace()
        self.iter = self.iter + 1
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.net.trainable_variables)
            loss = self.lossTF(xx1, xx, pmaxed1, a1, pmaxed0, a0, self.alpha)
        return loss, tape.gradient(loss, self.net.trainable_variables)

    def posterior(self, x):
        return self.net.predict(x, verbose=0)

    def copy(self):
        copy = PUPosterior(self.copyNet())
        return copy



class PUPosterior2(NetWithLoss):
    DEBUG = False
    def __init__(self, net):
        #Assume that x is sorted
        super(PUPosterior2, self).__init__(net)
        self.BCE =BinaryCrossentropy()
        self.alpha = 0.5
        self.alpha_ub = 1
        self.alpha_lb = 0
        self.gamma = 0.5
        self.iter = 1
        self.binarySearch = False
        self.historySize = 10
        self.historyIrr =  np.zeros((self.historySize,))
        self.historyIx = 0
        self.irr_prop_thr = 0.01
        self.irr_thr = 0.98


    def resetHistory(self,):
        self.historyIrr = np.zeros((self.historySize,))
        self.historyIx = 0

    def add2History(self, irr_prop):
        self.historyIrr[self.historyIx % self.historySize] = irr_prop
        self.historyIx = self.historyIx + 1

    def historySatisifiesIrr(self):
        return np.mean(self.historyIrr > self.irr_prop_thr)

    def valLoss(self, data):
        self.loss(data)

    def loss(self, data):
        x = data['x']
        x1 = data['x1']
        p = self.posterior(x)
        pmax = p/np.max(p)
        alpha = np.mean(pmax)
        p1 = self.posterior(x1)
        l1 = -alpha*(self.gamma * np.mean(np.log(p1)) + (1-self.gamma) * np.mean(xlogy(p, p)))
        l0 = -(1-alpha)* np.mean(xlogy(1-p, 1-p))
        loss = l1 + l0
        return loss


    def lossTF(self, x1, x, pmaxed1, a1, pmaxed0, a0, beta1, alphaC):
        p = self.net(x)
        p1 = self.net(x1)
        alpha = tf.reduce_mean(p)
        #beta2 = tf.reduce_mean(pmaxed1*p)/alpha
        #+ tf.math.abs(beta1 - beta2)
        l1 = -(alphaC ) * (self.gamma * tf.reduce_mean(tf.math.log(p1)) +
                      (1 - self.gamma) * tf.reduce_mean(tf.math.xlogy(pmaxed1, p))/a1)
        l0 = -(1-alphaC) * tf.reduce_mean(tf.math.xlogy(pmaxed0, 1-p))/a0
        #l2 = -alpha * tf.math.log(tf.reduce_mean(p)) - (1 - alpha) * tf.math.log(tf.reduce_mean(1-p))
        loss = l1 + l0
        #pdb.set_trace()
        return loss

    def determineAlpha(self, x):
        if self.historySatisifiesIrr() == 1:
            self.alpha_ub = np.mean(self.posterior(x))
            self.resetHistory()
            self.alpha = (self.alpha_lb + self.alpha_ub)/2
        elif self.historySatisifiesIrr() == 0:
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
        beta1 = np.mean(self.posterior(xx1)/max_p)
        pmaxed1 = p/max_p
        pmaxed0 = (1 - p) / np.max(1 - p)
        a1 = np.mean(pmaxed1)
        a0 = np.mean(pmaxed0)
        #max_p_all = np.max(self.posterior(x))
        #print('max_p_all' + str(max_p_all))
        pxx1 = self.posterior(xx1)
        print('max_pxx1: ' + str(np.max(pxx1)))
        irr_prop = np.mean(pxx1 > self.irr_thr)
        print('irr_prop: ' + str(irr_prop))
        self.add2History(irr_prop)
        if (not self.binarySearch) and self.historySatisifiesIrr() == 1:
            self.binarySearch = True
        if not self.binarySearch:
            self.alpha = a1 / (a1 + a0)
        else:
            self.alpha = self.determineAlpha(x)
        print('alpha: ' + str(self.alpha))
        print('alpha_ub: ' + str(self.alpha_ub))
        print('alpha_lb: ' + str(self.alpha_lb))
        #self.alpha = a1 / (a1 + a0)
        self.alpha = np.mean(p)
        self.alpha = 0.35
        #pdb.set_trace()
        self.iter = self.iter + 1
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.net.trainable_variables)
            loss = self.lossTF(xx1, xx, pmaxed1, a1, pmaxed0, a0, beta1, self.alpha)
        return loss, tape.gradient(loss, self.net.trainable_variables)

    def posterior(self, x):
        return self.net.predict(x, verbose=0)

    def copy(self):
        copy = PUPosterior2(self.copyNet())
        return copy



