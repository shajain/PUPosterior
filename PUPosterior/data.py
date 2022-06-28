from TrainTestVal.Classes import TrainTestValCV
from TrainTestVal.utilities import CVSplits


class DataPU:
    def __init__(self, x, x1, ex, ex1, dg=None):
        n = x.shape[0]
        n1 = x1.shape[0]
        self.N = [n, n1]
        self.K = 5
        self.TTV = TrainTestValCV(self.N, self.K)
        #self.TT = self.TTV.toTrainTest()
        self.x = x
        self.x1 = x1
        self.ex = ex
        self.ex1 = ex1
        if dg is not None:
            self.dg = dg
            self.hasDG = True
        else:
            self.dg = None
            self.hasDG = False

    def getTTVUL(self, i=None):
        if i is None:
            i = 1
        x_tr = self.TTV.filterTrain(self.x, 0, i)
        x_te = self.TTV.filterTest(self.x, 0, i)
        x_va = self.TTV.filterVal(self.x, 0, i)
        return x_tr, x_te, x_va

    def getTTVPos(self, i=None):
        if i is None:
            i = 1
        x1_tr = self.TTV.filterTrain(self.x1, 1, i)
        x1_te = self.TTV.filterTest(self.x1, 1, i)
        x1_va = self.TTV.filterVal(self.x1, 1, i)
        return x1_tr, x1_te, x1_va

    def getTTVKeyUL(self, key, i=None):
        if i is None:
            i = 1
        v_tr = self.TTV.filterTrain(self.ex[key], 0, i)
        v_te = self.TTV.filterTest(self.ex[key], 0, i)
        v_va = self.TTV.filterVal(self.ex[key], 0, i)
        return v_tr, v_te, v_va

    def getTTVKeyPos(self, key, i=None):
        if i is None:
            i = 1
        v_tr = self.TTV.filterTrain(self.ex1[key], 1, i)
        v_te = self.TTV.filterTest(self.ex1[key], 1, i)
        v_va = self.TTV.filterVal(self.ex1[key], 1, i)
        return v_tr, v_te, v_va

    def hasKeyUL(self, key):
        return key in self.ex

    def hasKeyPos(self, key):
        return key in self.ex1

    def getTTVValUL(self, val, i=None):
        if i is None:
            i = 1
        v_tr = self.TTV.filterTrain(val, 0, i)
        v_te = self.TTV.filterTest(val, 0, i)
        v_va = self.TTV.filterVal(val, 0, i)
        return v_tr, v_te, v_va

    def getTTVValPos(self, val, i=None):
        if i is None:
            i = 1
        v_tr = self.TTV.filterTrain(val, 1, i)
        v_te = self.TTV.filterTest(val, 1, i)
        v_va = self.TTV.filterVal(val, 1, i)
        return v_tr, v_te, v_va

    def getTTVData(self, i=None):
        data_tr = {}
        data_te = {}
        data_val = {}
        if i is None:
            i = 1
        data_tr['x'], data_te['x'], data_val['x'] = self.getTTVUL(i)
        data_tr['x1'], data_te['x1'], data_val['x1'] = self.getTTVPos(i)
        return data_tr, data_te, data_val

    def reshuffle(self):
        data = DataPU(self.x, self.x1, self.ex, self.ex1, self.dg)
        return data
