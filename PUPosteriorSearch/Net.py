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
from misc.TwoSampleTests import cramVonTest
from scipy.stats import cramervonmises_2samp as TwoSampleTest


class PUPosterior3(NetWithLoss):
    DEBUG = False
    def __init__(self, net):
        #Assume that x is sorted
        super(PUPosterior3, self).__init__(net, )
        self.BCE =BinaryCrossentropy()
        self.alpha = 0.5
        self.gamma = 0.5
        self.nBins = 10
        self.bins = np.linspace(0.1, 0.9, self.nBins - 1)


    def attachDiscriminator(self, disc):
        self.disc = disc


    def CramVonLoss(self, data):
        x = data['x']
        x1 = data['x1']
        discPost_pos = self.posteriorDisc(x1)
        discPost_ul = self.posteriorDisc(x)
        post1_ul = self.posterior(x)
        lossCramVon, pvalue, reject, softLabel1_all, ix_sort, ecdf_pos, ecdf_ppos = cramVonTest(discPost_pos, discPost_ul, post1_ul)
        return lossCramVon, pvalue, reject, softLabel1_all, ix_sort, ecdf_pos, ecdf_ppos


    def valLoss(self, data, hypPar=None):
        #x = data['x']
        #x1 = data['x1']
        #post1_ul = self.posterior(x)
        #post1_pos = self.posterior(x1)
        #loss = -np.mean(np.log(post1_pos)) + np.log(np.mean(post1_ul))
        loss = self.loss(data, hypPar)
        return loss

    def loss(self, data, hypPar=None):
        x = data['x']
        x1 = data['x1']
        p = self.posterior(x)
        alpha = np.mean(p)
        p1 = self.posterior(x1)
        l1 = -self.alpha*(self.gamma * np.mean(np.log(p1)) + (1-self.gamma) * np.mean(xlogy(p, p))/alpha)
        l0 = -(1-self.alpha)*np.mean(xlogy(1-p, 1-p))/(1-alpha)
        loss = l1 + l0
        #ix = np.digitize(postTrue, bins)
        return loss

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

    def lossTFHist(self, x1, x, pmaxed1, a1, pmaxed0, a0, alpha, binIx, histProbs1):
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

    def lossTF(self, x1, x, pmaxed1, a1, pmaxed0, a0, alpha, ix_sort, ecdf_pos, softLabel1_all, useCVMLoss):
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
        #pdb.set_trace()
        lossCramVon = tf.reduce_sum(softLabel1_all * (ecdf_ppos - ecdf_pos) ** 2) / tf.reduce_sum(softLabel1_all)
        #pdb.set_trace()
        loss = l1 + l0 + useCVMLoss*lossCramVon
        #loss = lossCramVon + 10*(tf.reduce_mean(p) - alpha)**2
        #pdb.set_trace()
        return loss



    def gradients(self, data, BS, hypPar):
        x = data['x']
        x1 = data['x1']
        n = x.shape[0]
        n1 = x1.shape[0]
        self.alpha = hypPar['alpha']
        print('alpha used in the loss: ', self.alpha)
        post1_max = hypPar['post1_max']
        print('max_p: ' + str(post1_max))
        post0_max = hypPar['post0_max']
        print('max_p0: ' + str(post0_max))
        BSx = np.cast['int64'](BS/self.alpha)
        ix1 = np.random.choice(n1, BS, replace=True)
        ix = np.random.choice(n, BSx, replace=True)
        xx1 = x1[ix1, :]
        xx = x[ix, :]
        post1_ul = self.posterior(xx)
        post1_max = np.max(post1_ul)
        post1_ul_norm = post1_ul/post1_max
        #post1_ul_norm[(post1_ul_norm > 1).flatten(), :] = 1
        post0_max = np.max(1-post1_ul)
        post0_ul_norm = (1 - post1_ul) / post0_max

        #post0_ul_norm[(post0_ul_norm > 1).flatten(), :] = 1
        a1 = np.mean(post1_ul_norm)
        a0 = np.mean(post0_ul_norm)
        _, _, reject, softLabel1_all, ix_sort, ecdf_pos, _ = self.CramVonLoss({'x': xx, 'x1': xx1})

        # discPost1 = self.posteriorDisc(xx1)
        # discPost = self.posteriorDisc(xx)
        # disc = np.vstack([discPost1, discPost[np.random.choice(BSx, BS, p=p.flatten() / np.sum(p)), :]])
        # per5 = np.percentile(disc, 5)
        # per95 = np.percentile(disc, 95)
        # self.bins = np.linspace(per5, per95, self.nBins - 1)
        # histProbs1 = self.histProbs(discPost1)
        # ix = np.digitize(discPost.flatten(), self.bins)
        # binIX = [np.where(ix == i)[0] for i in np.arange(self.nBins)]

        useCVMLoss = np.sum(reject)
        useCVMLoss = 0
        print('UseCVM'+str(useCVMLoss))
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # pdb.set_trace()
            tape.watch(self.net.trainable_variables)
            loss = self.lossTF(xx1, xx, post1_ul_norm, a1, post0_ul_norm, a0, self.alpha, ix_sort, ecdf_pos.astype(np.float32),
                               softLabel1_all.astype(np.float32), useCVMLoss)
        return loss, tape.gradient(loss, self.net.trainable_variables)

    def posterior(self, x):
        return self.net.predict(x)

    def posteriorDisc(self, x):
        return self.disc.predict(x)

    def discCDF(self, discPost, probs=None):
        if probs is None:
            probs = np.ones_like(discPost)
        n = np.sum(probs)
        discPost, ix_sort = randomSort(discPost)[0:2]
        cdf = np.cumsum(probs[ix_sort, :]) / n
        return cdf, discPost

    def posteriorMeanAndMax(self, data):
        x = data['x']
        x1 = data['x1']
        post1_ul = self.posterior(x)
        post1_pos = self.posterior(x1)
        alpha = np.mean(post1_ul)
        post1_max = np.max(np.vstack([post1_ul, post1_pos]))
        post0_max = np.max(np.vstack([1 - post1_ul, 1 - post1_pos]))
        return alpha, post1_max, post0_max


    def copy(self):
        copy = PUPosterior3(self.copyNet())
        copy.alpha = self.alpha
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
        valLoss = -self.CramVonLoss(data)[0]
        return valLoss


    def CramVonLoss(self, data):
        x = data['x']
        x1 = data['x1']
        discPost_pos = self.posterior(x1)
        discPost_ul = self.posterior(x)
        post1_ul = self.pnPosterior(x)
        lossCramVon, pvalue, reject, softLabel1_all, ix_sort, ecdf_pos, ecdf_ppos = cramVonTest(discPost_pos,
                                                                                                discPost_ul, post1_ul)
        return lossCramVon, pvalue, reject, softLabel1_all, ix_sort, ecdf_pos, ecdf_ppos

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



