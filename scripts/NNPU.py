import os
#os.chdir('../PUPosterior')
import sys
sys.path.remove(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from plots import sortedplot as sp
from NNPU2.fit import PosteriorFitting
import numpy as np
fits = []
NNLosses = []
nIter = 10
xs = []
posteriors = []
for i in np.arange(nIter):
    fit, dataPU = PosteriorFitting.demo()
    fits.append(fit)
    NNLoss = fit.trainer.bestNNLoss
    NNLosses.append(NNLoss)
    x = dataPU.x
    truePosterior = dataPU.ex['posterior']
    xs.append(x)
    posterior = NNLoss.posterior(x)
    posteriors.append(posterior)
    sp.sortedplot(x, posterior)
    sp.sortedplot(x, truePosterior)
    sp.show()
    print('end of iteration '+ str(i))

fig, ax = sp.subplots()
sp.sortedplot(x, truePosterior, ax=ax)
for i in np.arange(nIter):
    [sp.sortedplot(x, posterior, ax=ax) for (x,posterior) in zip(xs, posteriors)]

#sp.show()
fig.savefig('../figures/NNPU.png')