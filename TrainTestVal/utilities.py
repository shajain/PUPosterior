import numpy as np


def CVSplits(n, K=5, needVal=True):
    ix = np.random.permutation(np.arange(n))
    size = np.cast['int64'](np.ceil(n/K))
    folds = [ix[i*size: min((i+1)*size, n)] for i in np.arange(K)]
    test = folds
    if needVal:
        valFoldIx =  (np.arange(K)+1) % K
        val = [folds[i] for i in valFoldIx]
        train = [np.hstack([folds[j] for j in np.arange(K) if j not in [i, valFoldIx[i]]]) for i in np.arange(K)]
        return train, test, val
    train = [np.hstack([folds[j] for j in np.arange(K) if j != i]) for i in np.arange(K)]
    return train, test