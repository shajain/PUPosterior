import numpy as np

def posteriorBoxPlots(postEst, postTrue, ax, title):
    nBins = 10
    bins = np.linspace(0.1, 0.9, nBins - 1)
    ix = np.digitize(postTrue, bins)
    # pdb.set_trace()
    errors = postTrue - postEst
    ax.boxplot([errors[ix == i] for i in np.arange(10)])
    tickStr = ['.05']
    tickStr.extend([('{n:.2f}'.format(n=(bins[i] + bins[i + 1]) / 2))[1:] for i in np.arange(nBins - 2)])
    tickStr.extend(['.95'])
    ax.set_xticklabels(tickStr)
    ax.set_title(title)
    ax.set_xlabel('True posterior')