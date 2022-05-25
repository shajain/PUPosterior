import numpy as np
from misc.randomSort import randomSort
from scipy.stats import uniform
from scipy.stats import bernoulli
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.stats import cramervonmises_2samp as TwoSampleTest
import pdb

def cpeBinning(x, x1, binSize = 25, prior_approx=0.2, violation_prop = 0.01):
    x = randomSort(x)[0]
    n = np.size(x)
    x1 = randomSort(x1)[0]
    n1 = np.size(x1)
    cdf = np.zeros((n, 1))
    j = 0
    for i in np.arange(n):
        while (j < n1) and (x[i] >= x1[j]):
            j = j + 1
        cdf[i] = j / n1
    nBins = (n*prior_approx/binSize).astype('int64')
    binWidth = 1 / nBins
    binEdges = np.arange(binWidth, 1, binWidth)
    binEdges = np.hstack([binEdges, 1])
    #ix1 = np.digitize(x1, binEdges)
    ix = np.digitize(cdf, binEdges)
    lix = np.hstack((0, np.where(ix[0:-1] - ix[1:] == -1)[0]+1))
    rix = np.hstack((np.where(ix[1:] - ix[0:-1] == -1)[0], np.size(x)-1))
    w = np.ones_like(x)
    wBin = [r - l + 1 for (r,l) in zip(rix, lix)]
    priors = np.arange(nBins, n, nBins)/n
    dist = np.zeros_like(priors)
    violations
    for i in np.arange(np.size(priors)):
        for j in np.arange(nBins):
            if wBin[j] > 0


def gaussianLL(x, x1, prior):
    x = randomSort(x)[0]
    n = np.size(x)
    x1 = randomSort(x1)[0]
    n1 = np.size(x1)
    cdf = np.zeros((n, 1))
    j = 0
    for i in np.arange(n):
        while (j < n1) and (x[i] >= x1[j]):
            j = j + 1
        cdf[i] = j/n1
    nBins = np.sqrt(n*prior).astype('int64')
    binWidth = 1/nBins
    binEdges = np.arange(binWidth, 1, binWidth)
    binEdges = np.hstack([binEdges, 1])
    j = 0
    k = 0
    mixpdf = np.zeros((n, 1))
    f0pdf = np.zeros((n, 1))
    beta = 0
    for i in np.arange(nBins+1):
        while k < n and (binEdges[i] >= cdf[k]):
            k = k + 1
        #pdb.set_trace()
        mixp = (k - j) / n
        f0p = max(mixp - prior*binWidth, 0)
        beta = beta + f0p
        mixpdf[j:k, :] = mixp/binWidth
        f0pdf[j:k, :] = f0p
        j = k
    #pdb.set_trace()
    #mixpdf = mixpdf/binWidth
    #f0pdf = f0pdf/binWidth
    #mixpdf = prior + (1-prior)*f0pdf/beta
    #mixpdf = prior + f0pdf
    posterior = prior/mixpdf
    alphaEst = np.mean(posterior)
    posterior_tr = posterior
    posterior_tr[(posterior > 1).flatten(), :] = 1
    ix = bernoulli.rvs(posterior_tr).astype('bool')
    smp = cdf[ix]
    smpG = norm.ppf(smp)
    smpG[smpG == np.inf] = np.max(smpG[np.isfinite(smpG)])
    smpG[smpG == -np.inf] = np.min(smpG[np.isfinite(smpG)])
    ll = np.mean(np.log(norm.pdf(smpG)))
    return ll, alphaEst

def CrVmTest(x, x1, prior):
    x = randomSort(x)[0]
    n = np.size(x)
    x1 = randomSort(x1)[0]
    n1 = np.size(x1)
    cdf = np.zeros((n, 1))
    j = 0
    for i in np.arange(n):
        while (j < n1) and (x[i] >= x1[j]):
            j = j + 1
        cdf[i] = j/n1
    nBins = (n*prior/25).astype('int64')
    binWidth = 1/nBins
    binEdges = np.arange(binWidth, 1, binWidth)
    binEdges = np.hstack([binEdges, 1])
    j = 0
    k = 0
    mixpdf = np.zeros((n, 1))
    f0pdf = np.zeros((n, 1))
    beta = 0
    for i in np.arange(nBins+1):
        while k < n and (binEdges[i] >= cdf[k]):
            k = k + 1
        #pdb.set_trace()
        mixp = (k - j) / n
        f0p = max(mixp - prior*binWidth, 0)
        beta = beta + f0p
        mixpdf[j:k, :] = mixp/binWidth
        f0pdf[j:k, :] = f0p
        j = k
    #pdb.set_trace()
    #mixpdf = mixpdf/binWidth
    f0pdf = f0pdf/binWidth
    mixpdf = prior + (1-prior)*f0pdf/beta
    #mixpdf = prior + f0pdf
    posterior = prior/mixpdf
    alphaEst = np.mean(posterior)
    posterior_tr = posterior
    posterior_tr[(posterior > 1).flatten(), :] = 1
    ix = bernoulli.rvs(posterior_tr).astype('bool')
    smp = cdf[ix]
    res = TwoSampleTest(uniform.rvs(size=np.size(smp)), smp.flatten())
    return res.pvalue

def CrVmTest2(x, x1, prior):
    x = randomSort(x)[0]
    n = np.size(x)
    x1 = randomSort(x1)[0]
    n1 = np.size(x1)
    cdf = np.zeros((n, 1))
    j = 0
    for i in np.arange(n):
        while (j < n1) and (x[i] >= x1[j]):
            j = j + 1
        cdf[i] = j/n1
    nBins = np.sqrt(n*prior).astype('int64')
    binWidth = 1/nBins
    binEdges = np.arange(binWidth, 1, binWidth)
    binEdges = np.hstack([binEdges, 1])
    j = 0
    k = 0
    mixpdf = np.zeros((n, 1))
    f0pdf = np.zeros((n, 1))
    beta = 0
    for i in np.arange(nBins+1):
        while k < n and (binEdges[i] >= cdf[k]):
            k = k + 1
        #pdb.set_trace()
        mixp = (k - j) / n
        f0p = max(mixp - prior*binWidth, 0)
        beta = beta + f0p
        mixpdf[j:k, :] = mixp/binWidth
        f0pdf[j:k, :] = f0p
        j = k
    #pdb.set_trace()
    #mixpdf = mixpdf/binWidth
    f0pdf = f0pdf/binWidth
    mixpdf = prior + (1-prior)*f0pdf/beta
    #mixpdf = prior + f0pdf
    posterior = prior/mixpdf
    alphaEst = np.mean(posterior)
    posterior_tr = posterior
    posterior_tr[(posterior > 1).flatten(), :] = 1
    ix = bernoulli.rvs(posterior_tr).astype('bool')
    smp = cdf[ix]
    res = TwoSampleTest(uniform.rvs(size=np.size(smp)), smp.flatten())
    return res.pvalue