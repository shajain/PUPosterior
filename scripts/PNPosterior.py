import os
#os.chdir('../PUPosterior')
import sys
#sys.path.remove(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from plots import sortedplot as sp
from PNPosterior.fit import PosteriorFitting
import numpy as np
fits = []
NNLosses = []
nIter = 10
xs = []
posteriors = []
NNLossesLast = []
posteriorsLast = []

for i in np.arange(nIter):
    fit, dataPN = PosteriorFitting.demo()
    fits.append(fit)
    NNLoss = fit.trainer.bestNNLoss
    NNLosses.append(NNLoss)
    NNLossLast = fit.trainer.nnLoss
    NNLossesLast.append(NNLossLast)
    x1 = dataPN.x1
    x0 = dataPN.x0
    x = np.vstack([x1,x0])
    truePosterior = np.vstack([dataPN.ex1['posterior'],dataPN.ex0['posterior']])
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
fig.savefig('../figures/PNPosterior.png')

fig, ax = sp.subplots()
sp.sortedplot(x, truePosterior, color='k', ax=ax)
for i in np.arange(nIter):
    [sp.sortedplot(x, posterior, ax=ax) for (x,posterior) in zip(xs, posteriorsLast)]

#sp.show()
fig.savefig('../figures/PNPosteriorLast.png')