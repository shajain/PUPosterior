import numpy as np

from TrainTestVal.Classes import TrainTestValCV
from TrainTestVal.utilities import CVSplits


class DataPN:
    def __init__(self, x0, x1, ex0, ex1, dg=None):
        n1 = x1.shape[0]
        n0 = x0.shape[0]
        self.N = [n0, n1]
        self.K = 5
        self.TTV = TrainTestValCV(self.N, self.K)
        #self.TT = self.TTV.toTrainTest()
        self.x1 = x1
        self.x0 = x0
        self.ex1 = ex1
        self.ex0 = ex0
        if dg is not None:
            self.dg = dg
            self.hasDG = True
        else:
            self.hasDG = False

    def getTTVNeg(self, i=None):
        if i is None:
            i = 1
        x0_tr = self.TTV.filterTrain(self.x0, 0, i)
        x0_te = self.TTV.filterTest(self.x0, 0, i)
        x0_va = self.TTV.filterVal(self.x0, 0, i)
        return x0_tr, x0_te, x0_va

    def getTTVPos(self, i=None):
        if i is None:
            i = 1
        x1_tr = self.TTV.filterTrain(self.x1, 1, i)
        x1_te = self.TTV.filterTest(self.x1, 1, i)
        x1_va = self.TTV.filterVal(self.x1, 1, i)
        return x1_tr, x1_te, x1_va

    def getTTVKeyNeg(self, key, i=None):
        if i is None:
            i = 1
        v_tr = self.TTV.filterTrain(self.ex0[key], 0, i)
        v_te = self.TTV.filterTest(self.ex0[key], 0, i)
        v_va = self.TTV.filterVal(self.ex0[key], 0, i)
        return v_tr, v_te, v_va

    def getTTVKeyPos(self, key, i=None):
        if i is None:
            i = 1
        v_tr = self.TTV.filterTrain(self.ex1[key], 1, i)
        v_te = self.TTV.filterTest(self.ex1[key], 1, i)
        v_va = self.TTV.filterVal(self.ex1[key], 1, i)
        return v_tr, v_te, v_va

    def hasKeyNeg(self, key):
        return key in self.ex0

    def hasKeyPos(self, key):
        return key in self.ex1

    def getTTVValNeg(self, val, i=None):
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
        data_tr['x0'], data_te['x0'], data_val['x0'] = self.getTTVNeg(i)
        data_tr['x1'], data_te['x1'], data_val['x1'] = self.getTTVPos(i)
        return data_tr, data_te, data_val

    def getTTVDataXY(self, i=None):
        data_tr = {}
        data_te = {}
        data_val = {}
        if i is None:
            i = 1
        x0_tr, x0_te, x0_val = self.getTTVNeg(i)
        x1_tr, x1_te, x1_val = self.getTTVPos(i)
        data_tr['x'] = np.vstack([x1_tr, x0_tr])
        data_te['x'] = np.vstack([x1_te, x0_te])
        data_val['x'] = np.vstack([x1_val, x0_val])
        data_tr['y'] = np.vstack([np.ones((x1_tr.shape[0], 1)), np.zeros((x0_tr.shape[0], 1))])
        data_te['y'] = np.vstack([np.ones((x1_te.shape[0], 1)), np.zeros((x0_te.shape[0], 1))])
        data_val['y'] = np.vstack([np.ones((x1_val.shape[0], 1)), np.zeros((x0_val.shape[0], 1))])
        return data_tr, data_te, data_val

    def getTTVKeyPosNeg(self, key, i=None):
        if i is None:
            i = 1
        v0_tr, v0_te, v0_val = self.getTTVKeyNeg(key, i)
        v1_tr, v1_te, v1_val = self.getTTVKeyPos(key, i)
        v_tr = np.vstack([v1_tr, v0_tr])
        v_te = np.vstack([v1_te, v0_te])
        v_val = np.vstack([v1_val, v0_val])
        return v_tr, v_te, v_val


