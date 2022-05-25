import numpy as np
from misc.randomSort import randomSort
from scipy.stats import cramervonmises_2samp as TwoSampleTest

def cramVonTest(x1, x, p):
    n1 = x1.shape[0]
    n = x.shape[0]
    nPseudoPos = np.sum(p)
    y = np.vstack([np.ones((n1, 1)), np.zeros((n, 1))])
    ix_sort = randomSort(np.vstack([x1, x]))[1]
    y = y[ix_sort, :]
    ecdf_pos = np.cumsum(y, axis=0) / n1
    pseudoSoftLabel_all = np.vstack([np.zeros((n1, 1)), p])
    pseudoSoftLabel_all = pseudoSoftLabel_all[ix_sort, :]
    ecdf_ppos = np.cumsum(pseudoSoftLabel_all, axis=0) / nPseudoPos
    softLabel1_all = np.vstack([np.ones((n1, 1)), p])
    softLabel1_all = softLabel1_all[ix_sort, :]
    lossCramVon = np.sum(softLabel1_all * (ecdf_ppos - ecdf_pos) ** 2) / np.sum(softLabel1_all)
    tests = [TwoSampleTest(x1.flatten(),
                           x[np.random.choice(n, n1, p=p.flatten() / np.sum(p)), :].flatten())
             for i in np.arange(20)]
    pvalue = np.mean(np.array([test.pvalue for test in tests]))
    pvalue = tests[1].pvalue
    reject = pvalue < 0.25
    return lossCramVon, pvalue, reject, softLabel1_all, ix_sort, ecdf_pos, ecdf_ppos