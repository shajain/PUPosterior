#import os
#os.chdir('../PUPosterior')
import os
import sys
#sys.path.remove(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from plots import sortedplot as sp
from PUPosterior2.fit import PosteriorFitting
import numpy as np
fits = []
NNLosses = []
nIter = 10
xs = []
posteriors = []
NNLossesLast = []
posteriorsLast = []
for i in np.arange(nIter):
    fit, dataPU = PosteriorFitting.demo()
    fits.append(fit)
    NNLoss = fit.trainer.PUTrainer.bestNNLoss
    NNLosses.append(NNLoss)
    NNLossLast = fit.trainer.PUTrainer.nnLoss
    NNLossesLast.append(NNLossLast)
    x = dataPU.x
    truePosterior = dataPU.ex['posterior']
    xs.append(x)
    posterior = NNLoss.posterior(x)
    posteriors.append(posterior)
    posteriorLast = NNLossLast.posterior(x)
    posteriorsLast.append(posteriorLast)
    sp.sortedplot(x, posterior)
    sp.sortedplot(x, truePosterior)
    sp.sortedplot(x, posteriorLast)
    sp.show()
    print('end of iteration '+ str(i))

fig, ax = sp.subplots()
sp.sortedplot(x, truePosterior, ax=ax)
for i in np.arange(nIter):
    [sp.sortedplot(x, posterior, ax=ax) for (x,posterior) in zip(xs, posteriors)]
#sp.show()
fig.savefig('../figures/PUPosterior2.png')

fig, ax = sp.subplots()
sp.sortedplot(x, truePosterior, color='k', ax=ax)
for i in np.arange(nIter):
    [sp.sortedplot(x, posterior, ax=ax) for (x,posterior) in zip(xs, posteriorsLast)]

#sp.show()
fig.savefig('../figures/PNPosterior2Last.png')
