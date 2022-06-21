import tensorflow as tf
import numpy as np
from NN.NetWithLoss import NetWithLoss
from misc.randomSort import randomSort
from tensorflow.keras.losses import BinaryCrossentropy
from plots import sortedplot as sp
import pdb
from scipy.special import xlogy


class PNPosterior(NetWithLoss):
    DEBUG = False
    def __init__(self, net):
        #Assume that x is sorted
        super(PNPosterior, self).__init__(net)
        self.BCE =BinaryCrossentropy()
        self.alpha =0.5

    def valLoss(self, data):
        loss = self.loss(data)
        return loss

    def loss(self, data):
        x1 = data['x1']
        x0 = data['x0']
        p1 = self.posterior(x1)
        p0 = 1 - self.posterior(x0)
        loss = - self.alpha * np.mean(np.log(p1)) - (1 - self.alpha) * np.mean(np.log(p0))
        # loss = - np.mean(np.vstack([np.log(p1), np.log(1-p0)]))
        return loss

    def lossTF(self, x1, x0):
        p1 = self.alpha * self.net(x1)
        penalty1 = 100*tf.reduce_sum(tf.nn.relu(p1-1))
        p1 = tf.clip_by_value(p1, 0, 1)
        p0 = 1 - self.alpha * self.net(x0)
        penalty0 = 100*tf.reduce_sum(tf.nn.relu(-p0))
        p0 = tf.clip_by_value(p0, 0, 1)

        # a = tf.reduce_mean(p)
        loss = - self.alpha * tf.reduce_mean(tf.math.log(p1)) \
               - (1 - self.alpha) * tf.reduce_mean(tf.math.log(p0))
        # pdb.set_trace()
        return loss + penalty1 + penalty0


    def gradients(self, data, BS):
        #self.alpha = hypPar['alpha']
        x0 = data['x0']
        x1 = data['x1']
        n1 = x1.shape[0]
        n0 = x0.shape[0]
        self.alpha = n1 / (n0 + n1)
        print('n1/(n1+n0): ' + str(self.alpha))
        #self.alpha = 0.35
        ix1 = np.random.choice(n1, BS, replace=True)
        ix0 = np.random.choice(n0, BS, replace=True)
        xx1 = x1[ix1, :]
        xx0 = x0[ix0, :]
        print('alpha used in loss ' + str(self.alpha))
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.net.trainable_variables)
            loss = self.lossTF(xx1, xx0)
        return loss, tape.gradient(loss, self.net.trainable_variables)

    def posterior(self, x):
        #pdb.set_trace()
        return np.clip(self.alpha * self.net.predict(x, verbose=0), 0, 1)

    def posteriorByPriorMaxMin(self, data):
        x = data['x']
        x1 = data['x1']
        postByPrior = np.vstack([self.net.predict(x), self.net.predict(x1)])
        postByPriorMax = np.max(postByPrior)
        postByPriorMin = np.min(postByPrior)
        return postByPriorMax, postByPriorMin

    def copy(self):
        copy = PNPosterior(self.copyNet())
        copy.alpha = np.copy(self.alpha)
        return copy