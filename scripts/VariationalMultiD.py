import os
os.chdir('../PUPosterior')
import sys
#sys.path.remove(os.path.dirname(__file__))
#sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from plots import sortedplot as sp
from plotsPU.posterior import posteriorBoxPlots
from Variational.fit import PosteriorFitting
import numpy as np
from sklearn.metrics import roc_curve, auc
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
    fit, dataPU = PosteriorFitting.demoMultiDim(5,3,a)
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
    y = dataPU.ex['y']
    fpr, tpr, _ = roc_curve(y, posteriorTrue)
    aucTrue = auc(fpr, tpr)
    fpr, tpr, _ = roc_curve(y, posterior)
    aucBest = auc(fpr, tpr)
    fpr, tpr, _ = roc_curve(y, posteriorLast)
    aucLast = auc(fpr, tpr)

    fig, ax = sp.subplots(1,2)
    titleAlpha = 'alpha: {n:.2f} (T)'.format(n=a) + ' vs. {n:.2f} (B)'.format(n=NNLoss.alpha)
    titleAUC = 'AUC: {n:.2f} (T)'.format(n=aucTrue) + ' vs. {n:.2f} (B)'.format(n=aucBest)
    posteriorBoxPlots(posterior, posteriorTrue, ax[0], titleAlpha + '\n' + titleAUC)

    titleAlpha = 'alpha: {n:.2f} (T)'.format(n=a) + ' vs. {n:.2f} (L)'.format(n=NNLossLast.alpha)
    titleAUC = 'AUC: {n:.2f} (T)'.format(n=aucTrue) + ' vs. {n:.2f} (L)'.format(n=aucLast)
    posteriorBoxPlots(posterior, posteriorTrue, ax[1], titleAlpha + '\n' + titleAUC)

    fig.savefig('../figures/VariationalMultiD' + str(i)+ '.png')
    print('end of iteration '+ str(i))
    i = i + 1