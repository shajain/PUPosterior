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


    def lossTF(self, x1, x, alpha):
        post = self.net(x)
        post1 = self.net(x1)
        alphaHat = tf.reduce_mean(post)

        loss = tf.math.log(alphaHat) - tf.reduce_mean(np.log(post1))
        #priorLoss = -alpha*tf.math.log(alphaHat) - (1-alpha)* tf.math.log(1-alphaHat)
        #pdb.set_trace()
        #loss = loss + 10*priorLoss
        return loss


    def gradients(self, data, BS, hypPar=None):
        x = data['x']
        x1 = data['x1']
        n = x.shape[0]
        n1 = x1.shape[0]
        if hypPar is None:
            self.alpha = 0.35
        else:
            self.alpha = hypPar['alpha']

        BSx = np.cast['int64'](BS/self.alpha)
        ix1 = np.random.choice(n1, BS, replace=True)
        ix = np.random.choice(n, BSx, replace=True)
        xx1 = x1[ix1, :]
        xx = x[ix, :]
        # postPosUL = self.posterior(xx)
        # alphaHat = np.mean(postPosUL)
        # postPosL = self.posterior(xx1)
        # postPosMax = np.max(np.vstack((postPosUL,postPosL)))
        # postNegMax = np.max(1-np.vstack((postPosUL, postPosL)))
        # postPosUL = postPosUL/postPosMax
        # postNegUL = (1 - postPosUL)/postNegMax
        # postPosL = postPosL/postPosMax
        # wPosUL = postPosUL
        # wNegUL = postNegUL
        # wL = np.ones_like(postPosL)
        # vPosUL = wPosUL * (postPosUL**2)
        # vNegUL = wNegUL * (postPosUL**2)
        # vL = wL * (postPosL**2)
        # alphaMaxHat = np.mean(postPosUL)
        # alpha_v = np.mean(vPosUL)/np.mean(postPosUL**2)

        #if postPosMax > 0.9:
        # if True:
        #     alpha_v = (self.alpha/alphaMaxHat) * alpha_v
        # else:
        #     alpha_v = alpha_v * 0.95
        # print('posterior maximum: ' + str(postPosMax))

        print('alpha: ' + str(self.alpha))
        #print('weighted alpha : ' + str(alpha_v))
        self.iter = self.iter + 1
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.net.trainable_variables)
            loss = self.lossTF(xx1, xx, self.alpha)
            #if self.iter < 200 or postPosMax < 0.9:
            #if self.iter < 200:
            # if True:
            #     loss_weighted = self.lossTF(xx1, xx, alpha_v, vL, vPosUL, vNegUL)
            #     loss = loss + 0.1*loss_weighted
            # else:
            #     #loss = self.lossTF(xx1, xx, alpha_v, vL, vPosUL, vNegUL)
            #     print('not using weighted loss')
        return loss, tape.gradient(loss, self.net.trainable_variables)

    def posterior(self, x):
        return self.net.predict(x, verbose=0)

    def copy(self):
        copy = PUPosterior(self.copyNet())
        return copy

