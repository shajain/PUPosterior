import numpy as np
from abc import ABC, abstractmethod

class TrainTest(ABC):

    def __init__(self, train, test, data, extra=None):
        self.data = data
        self.extra = extra
        self.dataUnits = len(data)
        self.train = train
        self.test = test
        self.K = len(train[0])

    def getData(self):
        return self.data

    def getExtra(self):
        return self.extra

    def getTrainData(self, i):
        data = [{k:DU[k][IX[i], :] for k in [*DU]} for (DU, IX) in zip(self.data, self.train)]
        return data

    def getTestData(self, i):
        data = [{k: DU[k][IX[i], :] for k in [*DU]} for (DU, IX) in zip(self.data, self.test)]
        return data

    def getTrainExtra(self, i):
        extra = [{k: EU[k][IX[i], :] for k in [*EU]} for (EU, IX) in zip(self.extra, self.train)]
        return extra

    def getTestExtra(self, i):
        extra = [{k: EU[k][IX[i], :] for k in [*EU]} for (EU, IX) in zip(self.extra, self.test)]
        return extra

    def apply2Data(self, fnc, i, unitsData, keysData, unitsExtra=None, keysExtra=None):
        args = [self.data[u][k] for (u, k) in zip(unitsData, keysData)]
        if unitsExtra is not None:
            extraArgs = [self.extra[u][k] for (u, k) in zip(unitsExtra, keysExtra)]
            args = args + extraArgs
            return fnc(*args)

    def apply2DataPartition(self, fnc, partition, i, unitsData, keysData, unitsExtra=None, keysExtra=None):
        IX = getattr(self, partition)
        args = [self.data[u][k][IX[u][i]] for (u, k) in zip(unitsData, keysData)]
        if unitsExtra is not None:
            extraArgs = [self.extra[u][k][IX[u][i]] for (u, k) in zip(unitsExtra, keysExtra)]
            args = args + extraArgs
            return fnc(*args)

    def apply2TrainData(self, fnc, i, unitsData, keysData, unitsExtra=None, keysExtra=None):
        return self.apply2DataPartition(fnc, 'train', i, unitsData, keysData, unitsExtra, keysExtra)

    def apply2TestData(self, fnc, i, unitsData, keysData, unitsExtra=None, keysExtra=None):
        return self.apply2DataPartition(fnc, 'test', i, unitsData, keysData, unitsExtra, keysExtra)




class TrainTestVal(TrainTest):

    def __init__(self, train, test, val, data, extra=None):
        super(TrainTestVal, self).__init__(train, test, data, extra)
        self.val = val

    def getValData(self, i):
        data = [{k: DU[k][IX[i], :] for k in [*DU]} for (DU, IX) in zip(self.data, self.val)]
        return data

    def getValExtra(self, i):
        extra = [{k: EU[k][IX[i], :] for k in [*EU]} for (EU, IX) in zip(self.extra, self.val)]
        return extra

    def apply2ValData(self, fnc, i, unitsData, keysData, unitsExtra=None, keysExtra=None):
        return self.apply2DataPartition(fnc, 'val', i, unitsData, keysData, unitsExtra, keysExtra)

