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

    def valLoss(self, data):
        loss = self.loss(data)
        return loss

    def loss(self, data):
        x = data['x']
        x1 = data['x1']
        p = self.posterior(x)
        p1 = self.posterior(x1)
        a1 = np.mean(p)
        a0 = np.mean(1-p)
        l1 = -self.alpha*(self.gamma * np.mean(np.log(p1)) + (1-self.gamma) * np.mean(xlogy(p, p))/a1)
        l0 = -(1-self.alpha) * np.mean(xlogy(1-p, 1-p))/a0
        loss = l1 + l0
        return loss

    def lossTF(self, x1, x, pmaxed1, a1, pmaxed0, a0, alpha):
        p = alpha * self.net(x)
        penalty = 100 * tf.reduce_sum(tf.nn.relu(p - 1))
        p = tf.clip_by_value(p, 0, 1)
        p1 = alpha * self.net(x1)
        penalty1 = 100 * tf.reduce_sum(tf.nn.relu(p1 - 1))
        p1 = tf.clip_by_value(p1, 0, 1)
        l1 = -alpha * (self.gamma * tf.reduce_mean(tf.math.log(p1)) +
                      (1 - self.gamma) * tf.reduce_mean(tf.math.xlogy(pmaxed1, p))/a1)
        l0 = -(1 - alpha) * tf.reduce_mean(tf.math.xlogy(pmaxed0, 1-p))/a0
        loss = l1 + l0
        #pdb.set_trace()
        return loss + penalty1 + penalty


    def gradients(self, data, BS, hypPar):
        self.alpha = hypPar['alpha']
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
        #pdb.set_trace()
        a1 = np.mean(pmaxed1)
        a0 = np.mean(pmaxed0)

        self.iter = self.iter + 1
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.net.trainable_variables)
            loss = self.lossTF(xx1, xx, pmaxed1, a1, pmaxed0, a0, self.alpha)
        return loss, tape.gradient(loss, self.net.trainable_variables)

    def posterior(self, x):
        return np.clip(self.alpha * self.net.predict(x, verbose=0), 0, 1)

    def posteriorByPriorMaxMin(self, data):
        x = data['x']
        x1 = data['x1']
        postByPrior = np.vstack([self.net.predict(x), self.net.predict(x1)])
        postByPriorMax = np.max(postByPrior)
        postByPriorMin = np.min(postByPrior)
        return postByPriorMax, postByPriorMin

    def copy(self):
        copy = PUPosterior(self.copyNet())
        copy.alpha = np.copy(self.alpha)
        return copy