import numpy as np

from Variational.Net import PUPosterior as NNLoss
from NN.models import BasicSigmoid as Model
from Variational.trainer import TrainerVariational as Trainer
from PUPosterior.debug import Debug
from misc.dictUtils import safeUpdate
from misc.dictUtils import safeRemove
from plots import sortedplot as sp
from DataGenDistributions.datagen import GaussianDG
from DataGenDistributions.randomParameters import NormalMixPNParameters
from PUPosterior.data import DataPU
from TrainTestVal.utilities import CVSplits
from DataGenDistributions.datagen import NormalMixDG


class PosteriorFitting:

    netDEF = {'n_units': 10, 'n_hidden': 5, 'dropout_rate': 0.5}
    trainDEF =  {'batchSize': 200, 'maxIter': 800, 'debug': False}

    def __init__(self, dim, **kwargs):
        self.netDEF = safeUpdate(PosteriorFitting.netDEF, kwargs)
        self.trainDEF = safeUpdate(PosteriorFitting.trainDEF, kwargs)
        model = Model(**self.netDEF)
        model.build((None, dim))
        self.nnLoss = NNLoss(model)
        self.trainer = None

    def fit(self, data, **kwargs):
        self.fitArgs = {'data': data, **kwargs}
        self.trainer = Trainer(self.nnLoss, data, **safeRemove(self.trainDEF, 'debug'))
        if self.trainDEF['debug']:
            self.debug = Debug()
            self.debug.attachData(data)
            self.trainer.attachDebugger(self.debug)
        #pdb.set_trace()
        if 'hypPar' in kwargs:
            self.trainer.fit(kwargs['hypPar'])
        else:
            self.trainer.fit()

    def getNet(self):
        return self.nnLoss

    def setNet(self, nnLoss):
        self.nnLoss = nnLoss

    def refit(self):
        self.fit(**self.fitArgs)


    @classmethod
    def fromDataPU(cls, dataPU, hypPar=None):
        x = dataPU.x
        nDims = x.shape[1]
        fitting = PosteriorFitting(nDims, debug=False)
        fitting.fit(dataPU, hypPar=hypPar)
        return fitting, dataPU

    @classmethod
    def fromDG(cls, dg, n, n1):
        x1, x, y, c1, c = dg.pu_data(n1, n)[4:]
        nDims = x.shape[1]
        posterior = np.expand_dims(dg.pn_posterior(x), axis=1)
        posterior1 = np.expand_dims(dg.pn_posterior(x1), axis=1)
        sp.hist(posterior.flatten(), bins=20, density=True, alpha=0.5)
        sp.show()
        fitting = PosteriorFitting(nDims, debug=True)
        ex = {'y': y, 'c': c, 'posterior': posterior}
        ex1 = {'c': c, 'posterior': posterior1}
        dataPU = DataPU(x, x1, ex, ex1, dg)
        fitting.fit(dataPU)
        return fitting, dataPU

    @classmethod
    def demo(cls, alpha=None, hypPar=None):
        if alpha is None:
            alpha = 0.35
        mu = -1
        sig = 1
        #dg = GaussianDG(mu=mu, sig=sig, alpha=alpha)
        dg = NormalMixDG([-1,1], [1,1], np.array([0.7,0.3]), [0], [1], np.array([1]), alpha)
        n1 = 5000
        n = 10000
        x1, x, y, c1, c = dg.pu_data(n1, n, alpha)[4:]
        posterior = np.expand_dims(dg.pn_posterior(x, alpha), axis=1)
        posterior1 = np.expand_dims(dg.pn_posterior(x1, alpha), axis=1)
        sp.sortedplot(x, posterior)
        sp.sortedplot(x, dg.dens_neg(x))
        sp.sortedplot(x, dg.dens_pos(x))
        sp.hist(x, bins=20, density=True, alpha=0.5)
        sp.hist(x1, bins=20, density=True, alpha=0.5)
        sp.show( )
        fitting = PosteriorFitting(1, debug=False)
        ex = {'y': y, 'c': c, 'posterior': posterior}
        ex1 = {'c': c, 'posterior': posterior1}
        dataPU = DataPU(x, x1, ex, ex1, dg)
        fitting.fit(dataPU, hypPar=hypPar)
        return fitting, dataPU

    @classmethod
    def demoMultiDim(cls, nDims, nComps, alpha, hypPar=None):
        NMix = NormalMixPNParameters(nDims, nComps)
        aucpn_range = [0.7, 0.75]
        irr_vec = [0.01, 0.95, False]
        NMix.perturb2SatisfyMetrics(aucpn_range, irr_vec)
        dg = NMix.dg
        n = 50000
        n1 = 10000
        #alpha = 0.35
        dg.alpha = alpha
        x1, x, y, c1, c = dg.pu_data(n1, n, alpha)[4:]
        posterior = np.expand_dims(dg.pn_posterior(x, alpha), axis =1)
        posterior1 =  np.expand_dims(dg.pn_posterior(x1, alpha), axis =1)
        sp.hist(posterior.flatten(), bins=20, density=True, alpha=0.5)
        sp.show( )
        fitting = PosteriorFitting(nDims, debug=False)
        ex = {'y': y, 'c': c, 'posterior': posterior}
        ex1 = {'c': c, 'posterior': posterior1}
        dataPU = DataPU(x, x1, ex, ex1, dg)
        fitting.fit(dataPU, hypPar = hypPar)
        return fitting, dataPU