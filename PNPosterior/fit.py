import numpy as np
from PNPosterior.Net import PNPosterior as NNLoss
from NN.models import BasicSigmoid as Model
from NN.trainer import Trainer as Trainer
from PNPosterior.debug import Debug
from misc.dictUtils import safeUpdate
from misc.dictUtils import safeRemove
from plots import sortedplot as sp
from DataGenDistributions.datagen import GaussianDG
from DataGenDistributions.randomParameters import NormalMixPNParameters
from PNPosterior.data import DataPN
import pdb


class PosteriorFitting:

    netDEF = {'n_units': 10, 'n_hidden': 5, 'dropout_rate': 0.5}
    trainDEF =  {'batchSize': 200, 'maxIter': 500, 'debug': False}

    def __init__(self, dim, **kwargs):
        self.netDEF = safeUpdate(PosteriorFitting.netDEF, kwargs)
        self.trainDEF = safeUpdate(PosteriorFitting.trainDEF, kwargs)
        model = Model(**self.netDEF)
        model.build((None, dim))
        self.nnLoss = NNLoss(model)

    def fit(self, data, **kwargs):
        self.fitArgs = {'data': data, **kwargs}
        trainer = Trainer(self.nnLoss, data, **safeRemove(self.trainDEF, 'debug'))
        if self.trainDEF['debug']:
            self.debug = Debug()
            #pdb.set_trace()
            self.debug.attachData(data)
            trainer.attachDebugger(self.debug)
        #pdb.set_trace()
        trainer.fit( )

    def getNet(self):
        return self.nnLoss

    def setNet(self, nnLoss):
        self.nnLoss = nnLoss

    def refit(self):
        self.fit(**self.fitArgs)

    @classmethod
    def fromDataPN(cls, dataPN):
        x = dataPN.x
        nDims = x.shape[1]
        fitting = PosteriorFitting(nDims, debug=True)
        fitting.fit(dataPN)
        return fitting, dataPN

    @classmethod
    def fromDataPU(cls, dataPU):
        x = dataPU.x
        nDims = x.shape[1]
        ex = dataPU.ex
        y = ex['y']
        x0 = x[(y == 0).flatten(), :]
        x1 = x[(y == 1).flatten(), :]
        ex1 = dict()
        ex0 = dict()
        if 'c' in ex:
            c = ex['c']
            c0 = c[(y == 0).flatten(), :]
            c1 = c[(y == 1).flatten(), :]
            ex0 = {'c': c0}
            ex1 = {'c': c1}
        if 'posterior' in ex:
            posterior = ex['posterior']
            ex0['posterior'] = posterior[(y == 0).flatten(), :]
            ex1['posterior'] = posterior[(y == 1).flatten(), :]
            sp.hist(posterior.flatten(), bins=20, density=True, alpha=0.5)
            sp.show()
        if dataPU.hasDG:
            dg = dataPU.dg
        else:
            dg = None
        fitting = PosteriorFitting(nDims, debug=True)
        dataPN = DataPN(x0, x1, ex0, ex1, dg)
        fitting.fit(dataPN)
        return fitting, dataPN

    @classmethod
    def demo(cls):
        alpha = 0.35
        mu = -1
        sig = 1
        dg = GaussianDG(mu=mu, sig=sig, alpha=alpha)
        n = 10000
        x, y, c = dg.pn_data(n)[0:3]
        posterior = np.expand_dims(dg.pn_posterior(x), axis=1)
        #pdb.set_trace()
        sp.sortedplot(x, posterior)
        sp.sortedplot(x, dg.dens_neg(x))
        sp.sortedplot(x, dg.dens_pos(x))
        sp.hist(x[(y == 1).flatten()], bins=20, density=True, alpha=0.5)
        sp.hist(x[(y == 0).flatten()], bins=20, density=True, alpha=0.5)
        sp.show()
        fitting = PosteriorFitting(1, debug=False)
        x0 = x[(y == 0).flatten(), :]
        x1 = x[(y == 1).flatten(), :]
        c0 = c[(y == 0).flatten(), :]
        c1 = c[(y == 1).flatten(), :]
        ex0 = {'c': c0}
        ex1 = {'c': c1}
        ex0['posterior'] = posterior[(y == 0).flatten(), :]
        ex1['posterior'] = posterior[(y == 1).flatten(), :]
        dataPN = DataPN(x0, x1, ex0, ex1, dg)
        fitting.fit(dataPN)
        return fitting, dataPN

    @classmethod
    def demoMultiDim(cls, nDims, nComps):
        NMix = NormalMixPNParameters(nDims, nComps)
        aucpn_range = [0.7, 0.75]
        irr_vec = [0.01, 0.95, False]
        NMix.perturb2SatisfyMetrics(aucpn_range, irr_vec)
        #pdb.set_trace()
        dg = NMix.dg
        dg.alpha = 0.35
        n = 10000
        x, y, c = dg.pn_data(n)[0:3]
        posterior = np.expand_dims(dg.pn_posterior(x), axis=1)
        sp.hist(posterior.flatten(), bins=20, density=True, alpha=0.5)
        sp.show()
        fitting = PosteriorFitting(nDims, debug=True)
        x0 = x[(y == 0).flatten(), :]
        x1 = x[(y == 1).flatten(), :]
        c0 = c[(y == 0).flatten(), :]
        c1 = c[(y == 1).flatten(), :]
        ex0 = {'c': c0}
        ex1 = {'c': c1}
        ex0['posterior'] = posterior[(y == 0).flatten(), :]
        ex1['posterior'] = posterior[(y == 1).flatten(), :]
        dataPN = DataPN(x0, x1, ex0, ex1, dg)
        fitting.fit(dataPN)
        return fitting, dataPN
