import pdb

import numpy as np
from PUPosterior2.Net import PUPosterior2 as PUPost
from PUPosterior2.Net import Discriminator as Disc
from NN.models import BasicSigmoid as ModelPU
from NN.models import BasicSigmoid as ModelDisc
from NN.trainer import AlternatingTrainer as AltTrain
from PUPosterior2.debug import Debug
from misc.dictUtils import safeUpdate
from misc.dictUtils import safeRemove
from plots import sortedplot as sp
from DataGenDistributions.datagen import GaussianDG
from DataGenDistributions.randomParameters import NormalMixPNParameters
from PUPosterior.data import DataPU

class PosteriorFitting:
    DiscNetDEF = {'n_units': 3, 'n_hidden': 3, 'dropout_rate': 0.5}
    PUNetDEF = {'n_units': 10, 'n_hidden': 5, 'dropout_rate': 0.5}
    trainDEF = {'batchSize': 200, 'maxIter': 15, 'debug': False, 'rounds': 60}

    def __init__(self, nDims, **kwargs):
        self.DiscNetDEF = safeUpdate(PosteriorFitting.DiscNetDEF, kwargs)
        self.PUNetDEF = safeUpdate(PosteriorFitting.PUNetDEF, kwargs)
        self.trainDEF = safeUpdate(PosteriorFitting.trainDEF, kwargs)
        #pdb.set_trace()
        modelPU = ModelPU(**self.PUNetDEF)
        modelDisc = ModelDisc(**self.DiscNetDEF)
        #pdb.set_trace()
        modelPU.build((None, nDims))
        modelDisc.build((None, nDims))
        self.PUPost = PUPost(modelPU)
        self.PUPost.attachDiscriminator(modelDisc)
        self.disc = Disc(modelDisc)
        self.disc.attachPNPosteriorNet(modelPU)
        self.trainer = None

    def fit(self, data, **kwargs):
        self.fitArgs = {'data': data, **kwargs}
        self.trainer = AltTrain(self.PUPost, self.disc, data, **safeRemove(self.trainDEF, 'debug'))
        #pdb.set_trace()
        if self.trainDEF['debug']:
            self.debug = Debug()
            self.debug.attachData(data)
            self.trainer.attachDebugger(self.debug)
        self.trainer.fit( )

    def refit(self):
        self.fit(**self.fitArgs)

    def getNets(self):
        return self.PUPost, self.disc

    @classmethod
    def fromDataPU(cls, dataPU):
        x = dataPU.x
        nDims = x.shape[1]
        fitting = PosteriorFitting(nDims, debug=True)
        fitting.fit(dataPU)
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
    def demo(cls):
        alpha = 0.35
        mu = -1
        sig = 1
        dg = GaussianDG(mu=mu, sig=sig, alpha=alpha)
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
        sp.show()
        fitting = PosteriorFitting(1, debug=False)
        ex = {'y': y, 'c': c, 'posterior': posterior}
        ex1 = {'c': c, 'posterior': posterior1}
        dataPU = DataPU(x, x1, ex, ex1, dg)
        fitting.fit(dataPU)
        return fitting, dataPU

    @classmethod
    def demoMultiDim(cls, nDims, nComps):
        NMix = NormalMixPNParameters(nDims, nComps)
        aucpn_range = [0.7, 0.75]
        irr_vec = [0.01, 0.95, False]
        NMix.perturb2SatisfyMetrics(aucpn_range, irr_vec)
        dg = NMix.dg
        n = 10000
        n1 = 1000
        alpha = 0.35
        dg.alpha = alpha
        x1, x, y, c1, c = dg.pu_data(n1, n, alpha)[4:]
        posterior = np.expand_dims(dg.pn_posterior(x, alpha), axis=1)
        posterior1 = np.expand_dims(dg.pn_posterior(x1, alpha), axis=1)
        sp.hist(posterior.flatten(), bins=20, density=True, alpha=0.5)
        sp.show()
        fitting = PosteriorFitting(nDims, debug=True)
        ex = {'y': y, 'c': c, 'posterior': posterior}
        ex1 = {'c': c, 'posterior': posterior1}
        dataPU = DataPU(x, x1, ex, ex1, dg)
        fitting.fit(dataPU)
        return fitting, dataPU