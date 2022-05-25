import numpy as np

def binBy(x, y, nbins=30):
    """
    Bin x by y.
    Returns the binned "x" values and the left edges of the bins
    """
    bins = np.linspace(y.min(), y.max(), nbins+1)
    # To avoid extra bin for the max value
    bins[-1] += 1

    indicies = np.digitize(y, bins)

    output = []
    for i in range(1, len(bins)):
        output.append(x[indicies==i])

    # Just return the left edges of the bins
    bins = bins[:-1]

    return output, bins