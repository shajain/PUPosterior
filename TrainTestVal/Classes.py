import numpy as np
from abc import ABC, abstractmethod
from TrainTestVal.utilities import CVSplits
import pdb

class TrainTest(ABC):

    def __init__(self, train, test):
        self.dataUnits = len(train)
        self.train = train
        self.test = test
        self.K = len(train[0])


    def filterPartitionFold(self, V, U, i, partition):
        #pdb.set_trace()
        IX = getattr(self, partition)
        if V is list:
            V = [v[IX[u][i], :] for (v, u) in zip(V, U)]
        else:
            V = V[IX[U][i], :]
        return V

    def filterTrain(self, V, U, i):
        return self.filterPartitionFold(V, U, i, 'train')

    def filterTest(self, V, U, i):
        return self.filterPartitionFold(V, U, i, 'test')



class TrainTestVal(TrainTest):

    def __init__(self, train, test, val):
        super(TrainTestVal, self).__init__(train, test)
        self.val = val

    def filterVal(self, V, U, i):
        return self.filterPartitionFold(V, U, i, 'val')

    def toTrainTest(self):
        train = [[np.hstack(tr[i], va[i]) for i in np.arange(self.K)] for (tr, va) in zip(self.train, self.val)]
        return TrainTest(train, self.test)


class TrainTestCV(TrainTest):
    def __init__(self, N, K):
        train = []
        test = []
        for n in N:
            tr, te = CVSplits(n, K, False)
            train.append(tr)
            test.append(te)
        super(TrainTestCV, self).__init__(train, test)


class TrainTestValCV(TrainTestVal):
    def __init__(self, N, K):
        train = []
        test = []
        val = []
        for n in N:
            tr, te, va = CVSplits(n, K, True)
            train.append(tr)
            test.append(te)
            val.append(va)
        super(TrainTestValCV, self).__init__(train, test, val)


