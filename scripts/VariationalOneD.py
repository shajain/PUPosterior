import os
#os.chdir('../PUPosterior')
import sys
#sys.path.remove(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from plots import sortedplot as sp
from Variational.fit import PosteriorFitting
import numpy as np
fits = []
alphas = np.random.rand(10)
nIter = alphas.size
xs = []
posteriors = []
posteriorsLast = []
posteriorsTrue = []
NNLosses = []
NNLossesLast = []
i = 0
for a in alphas:
    print('true alpha:' + str(a))
    fit, dataPU = PosteriorFitting.demo(a)
    fits.append(fit)
    NNLoss = fit.trainer.bestNNLoss
    NNLosses.append(NNLoss)
    NNLossLast = fit.trainer.nnLoss
    NNLossesLast.append(NNLossLast)
    x = dataPU.x
    xs.append(x)
    posteriorTrue = dataPU.ex['posterior']
    posteriorsTrue.append(posteriorTrue)
    posterior = NNLoss.posterior(x)
    posteriors.append(posterior)
    posteriorLast = NNLossLast.posterior(x)
    posteriorsLast.append(posteriorLast)
    fig, ax = sp.subplots()
    sp.sortedplot(x, posteriorTrue, ax=ax, label='T')
    sp.sortedplot(x, posterior, ax=ax, label='B')
    sp.sortedplot(x, posteriorLast, label='L')
    sp.title('alpha: {n:.2f} (T)'.format(n=a) + ' vs. {n:.2f} (B)'.format(n=NNLoss.alpha))
    ax.legend()
    fig.savefig('../figures/VariationalOneD' + str(i)+ '.png')
    print('end of iteration '+ str(i))
    i = i + 1