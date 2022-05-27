import tensorflow as tf
import numpy as np
from NN.NetWithLoss import NetWithLoss
from misc.randomSort import randomSort
from tensorflow.keras.losses import BinaryCrossentropy
from plots import sortedplot as sp
import pdb
from scipy.special import xlogy


class NNPU(NetWithLoss):
    DEBUG = False
    def __init__(self, net, alpha, largeScale=False, gamma=1, beta=0):
        #Assume that x is sorted
        super(NNPU, self).__init__(net)
        self.alpha = alpha
        self.gamma = gamma
        self.iter = 1
        self.beta = beta
        self.largeScale = largeScale

    def valLoss(self, data):
        loss = self.loss(data)
        return loss

    def loss(self, data):
        x = data['x']
        x1 = data['x1']
        p = self.posterior(x)
        alpha = self.alpha
        p1 = self.posterior(x1)
        l1 = alpha * np.mean((1-p1)**2)
        l0 = max(0, np.mean(p**2) - alpha* np.mean(p1**2))
        loss = l1 + l0
        return loss

    def lossNeg(self, data):
        x = data['x']
        x1 = data['x1']
        p = self.posterior(x)
        alpha = self.alpha
        p1 = self.posterior(x1)
        l0 = np.mean(p ** 2) - alpha * np.mean(p1 ** 2)
        return l0

    def lossTF(self, x1, x, alpha):
        p = self.net(x)
        p1 = self.net(x1)
        l1 = alpha * tf.reduce_mean((1 - p1) ** 2)
        l0 = tf.reduce_mean(p ** 2) - alpha * tf.reduce_mean(p1 ** 2)
        #l2 = -alpha * tf.math.log(tf.reduce_mean(p)) - (1 - alpha) * tf.math.log(tf.reduce_mean(1-p))
        loss = l1 + tf.maximum(l0, 0)
        #pdb.set_trace()
        return loss

    def lossNegTF(self, x1, x, alpha):
        p = self.net(x)
        p1 = self.net(x1)
        l0 = tf.reduce_mean(p ** 2) - alpha * tf.reduce_mean(p1 ** 2)
        return l0


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
        self.iter = self.iter + 1
        if not self.largeScale or self.lossNeg({'x':xx, 'x1': xx1}) > -self.beta:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                # pdb.set_trace()
                tape.watch(self.net.trainable_variables)
                loss = self.lossTF(xx1, xx, self.alpha)
            return loss, tape.gradient(loss, self.net.trainable_variables)
        else:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.net.trainable_variables)
                loss = -self.gamma*self.lossNegTF(xx1, xx, self.alpha)
            return loss, tape.gradient(loss, self.net.trainable_variables)


    def posterior(self, x):
        return self.net.predict(x, verbose=0)

    def copy(self):
        copy = NNPU(self.copyNet(), self.alpha)
        return copy











