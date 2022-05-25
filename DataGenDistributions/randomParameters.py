from scipy.stats import norm
from scipy.stats import uniform
from random import seed
from random import randint
from random import random
import numpy as np
from DataGenDistributions.distributions import mixture
from sklearn.datasets import make_spd_matrix as spd
from scipy.stats import dirichlet
from sklearn import metrics
from scipy.stats import multivariate_normal as mvn
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import pdb as pdb
from sklearn.utils import shuffle
from types import SimpleNamespace as SN
from DataGenDistributions.datagen import MVNormalMixDG
from DataGenDistributions.datagen import DataGenerator
from itertools import combinations
from plots.CIEllipse import CIEllipse
from matplotlib.patches import Ellipse



class NormalMixRandomParameters:

    def __init__(self, dim, max_comps):
        self.dim = dim
        self.max_comps = max_comps
        self.n_comps_pos = randint(1, max_comps)
        self.n_comps_neg = randint(1, max_comps)
        self.mu_pos = [np.array([2 * random() - 1 for i in np.arange(dim)]) for j in np.arange(self.n_comps_pos)]
        self.mu_neg = [np.array([2 * random() - 1 for i in np.arange(dim)]) for j in np.arange(self.n_comps_neg)]
        self.sig_pos = [spd(dim) for j in np.arange(self.n_comps_pos)]
        self.sig_neg = [spd(dim) for j in np.arange(self.n_comps_neg)]
        self.p_pos = dirichlet(np.ones(self.n_comps_pos)).rvs([])
        self.p_neg = dirichlet(np.ones(self.n_comps_neg)).rvs([])
        self.alpha = random()

    def computePNDataMetrics(self):
        epsilon = 10 ** -7
        n = 10000
        _, x, y, pos, neg, dg = self.generatePNData(n, n)
        posterior_pos = dg.pn_posterior_balanced(pos)
        irreducibility = np.mean(np.cast['int32'](posterior_pos > 1-epsilon).flatten())
        posterior_x = dg.pn_posterior_balanced(x)
        fpr, tpr, thresholds = metrics.roc_curve(y, posterior_x)
        aucpn = metrics.auc(fpr, tpr)
        #pdb.set_trace()
        return {'aucpn': aucpn, 'irreducibility': irreducibility}

    def createDataGenerator(self, n_pos, n_ul):
        dg = MVNormalMixDG(self.mu_pos, self.sig_pos, self.p_pos, self.mu_neg, self.sig_neg, self.p_neg, self.alpha,
                         n_pos, n_ul)
        return dg

    def generatePNData(self, n_pos, n_neg):
        dg = self.createDataGenerator(500, 2000)
        pos = dg.data_pos(n_pos)
        neg = dg.data_neg(n_neg)
        y = np.concatenate((np.ones([n_pos, 1]), np.zeros([n_neg, 1])), axis=0)
        x = np.concatenate((pos, neg), axis=0)
        xy = np.concatenate((x, y), axis=1)
        return xy, x, y, pos, neg, dg

    def perturb2Irreducibility(self, irr_range):
        metricsPN = self.computePNDataMetrics()
        if irr_range[0] <= metricsPN['irreducibility'] <= irr_range[1]:
            return
        sigma_flag = random( ) > 0.5
        if metricsPN['irreducibility'] < irr_range[0]:
            if sigma_flag:
                self.decrease_covar(irr_range)
            else:
                self.move_mean_out(irr_range)
        else:
            if sigma_flag:
                self.increase_covar(irr_range)
            else:
                if self.equalMeans():
                    self.alignCovar(irr_range)
                else:
                    self.move_mean_in(irr_range)
        metricsPN = self.computePNDataMetrics()
        if irr_range[0] <= metricsPN['irreducibility'] <= irr_range[1]:
            return
        else:
            self.perturb2Irreducibility(irr_range)

    def increase_covar(self, irr_range):
        i_pos = randint(0, self.n_comps_pos - 1)
        sig = self.sig_pos[i_pos]
        sig_ratio = self.sigmaRatio(sig)
        dim_float = np.cast['float32'](self.dim)
        if all(sig_ratio <= 2):
            up = 1.1
            while up > 1.001:
                self.sig_pos[i_pos] = up * sig
                metricsPN = self.computePNDataMetrics()
                if metricsPN['irreducibility'] < irr_range[0]:
                    up = 1 + (up-1)/2
                else:
                    break

    def decrease_covar(self, irr_range):
        i_pos = randint(0, self.n_comps_pos - 1)
        sig = self.sig_pos[i_pos]
        sig_ratio = self.sigmaRatio(sig)
        dim_float = np.cast['float32'](self.dim)
        if all(sig_ratio > 0.5):
            up = 0.5
            while up < .99:
                self.sig_pos[i_pos] = up * sig
                metricsPN = self.computePNDataMetrics( )
                if metricsPN['irreducibility'] > irr_range[1]:
                    up = 1 - (1 - up) / 2
                else:
                    break

    def move_mean_out(self, irr_range):
        i_pos, _ = self.componentIrreducibilitySampling()
        #pdb.set_trace()
        mu_pos = self.mu_pos[i_pos]
        i_neg, _ = self.closestNegComp(mu_pos)
        mu_neg = self.mu_neg[i_neg]
        delta = mu_pos - mu_neg
        self.mu_pos[i_pos] = self.mu_pos[i_pos] + 0.1 * delta
        metricsPN = self.computePNDataMetrics( )

        if metricsPN['irreducibility'] > irr_range[1]:
            k = 0
            lower = self.mu_pos[i_pos] - delta
            upper = self.mu_pos[i_pos]
            while ((metricsPN['irreducibility'] > irr_range[1]) or (metricsPN['irreducibility'] < irr_range[0])) and (k < 5):
                self.mu_pos[i_pos] =  (lower + upper)/2
                metricsPN = self.computePNDataMetrics( )
                if irr_range[0] <= metricsPN['irreducibility'] <= irr_range[1]:
                    break
                if metricsPN['irreducibility'] > irr_range[1]:
                    upper = self.mu_pos[i_pos]
                else:
                    lower = self.mu_pos[i_pos]
                k = k + 1

    def move_mean_in(self, irr_range):
        i_pos = randint(0, self.n_comps_pos - 1)
        upper = self.mu_pos[i_pos]
        i_neg, _ = self.closestNegComp(upper)
        lower = self.mu_neg[i_neg]
        metricsPN = self.computePNDataMetrics( )
        k = 0
        while (metricsPN['irreducibility'] > irr_range[1]) and (k < 5):
            self.mu_pos[i_pos] = (lower + upper)/2
            metricsPN = self.computePNDataMetrics( )
            if irr_range[0] <= metricsPN['irreducibility'] <= irr_range[1]:
                break
            if metricsPN['irreducibility'] > irr_range[1]:
                upper = self.mu_pos[i_pos]
            else:
                lower = self.mu_pos[i_pos]
            k = k + 1

    def align_covar(self, irr_range):
        i_pos = randint(0, self.n_comps_pos - 1)
        sig_pos = self.sig_pos[i_pos]
        i_neg, _ = self.closestNegComp(self.mu_pos[i_pos])
        sig_neg = self.sig_neg[i_neg]
        metricsPN = self.computePNDataMetrics( )
        k = 0
        up = 1.0
        low = 0.0
        while (metricsPN['irreducibility'] > irr_range[1]) and (k < 5):
            a = (up + low)/2.0
            self.sig_pos[i_pos] = a * sig_pos + (1-a) * sig_neg
            metricsPN = self.computePNDataMetrics( )
            if irr_range[0] <= metricsPN['irreducibility'] <= irr_range[1]:
                break
            if metricsPN['irreducibility'] > irr_range[1]:
                up = a
            else:
                low = a
            k = k + 1

    def equalMeans(self):
        epsilon = 10**-1
        ix_delta = [self.closestNegComp(mu) for mu in self.mu_pos]
        ix_delta = list(zip(*ix_delta))
        delta = ix_delta[1]
        sum = np.sum(delta, axis=0)
        return sum < epsilon

    def closestNegComp(self, mean):
        delta = np.array([np.sum((mean-mu)**2, axis=0) for mu in self.mu_neg])
        ix = np.argmin(delta, axis=0)
        return ix, delta[ix]

    def sigmaRatio(self, sigma):
        det = np.linalg.det(sigma)
        return np.array([det/np.linalg.det(sig_neg) for sig_neg in self.sig_neg])


    def componentIrreducibilitySampling(self):
        dg = self.createDataGenerator(500, 2000)
        comp_irr = [np.mean(dg.pn_posterior_balanced(comp.rvs(size=500)), axis=0) for comp in dg.components_pos]
        comp_irr = np.array(comp_irr)
        p = (1-comp_irr)/np.sum(1-comp_irr, axis=0)
        ix = np.random.choice(self.n_comps_pos, 1, p=p)
        ix = np.reshape(ix, newshape=())
        return ix, comp_irr




class NormalMixPNParameters:
# Used to gnerate PN DataGenDistributions with desired AUC and irreducibility.\
# The P and N distributions are mixtures of gaussians with equal number of components
    def __init__(self, dim, max_comps):
        self.dim = dim
        self.max_comps = max_comps
        #self.aucpn_range = aucpn_range
        #self.irr_vec = irr_vec
        #self.n_comps_pos = randint(1, max_comps)
        #self.n_comps_neg = randint(1, max_comps)
        self.n_comps_pos = max_comps
        self.n_comps_neg = max_comps
        self.mu_pos = list()
        self.mu_neg = list()
        for i in np.arange(max(self.n_comps_pos, self.n_comps_neg)):
            mu = np.array([2* random() - 1 for i in np.arange(self.dim)])
            mu = 0.5*mu/np.sqrt(self.dim)
            if i < self.n_comps_pos:
                self.mu_pos.append(mu)
            if i < self.n_comps_neg:
                self.mu_neg.append(mu)
        #self.mu_pos = [np.zeros(dim) for j in np.arange(self.n_comps_pos)]
        #self.mu_neg = [np.zeros(dim) for j in np.arange(self.n_comps_neg)]
        self.sig_pos = [np.identity(dim) for j in np.arange(self.n_comps_pos)]
        self.sig_neg = [np.identity(dim) for j in np.arange(self.n_comps_neg)]
        self.p_pos = dirichlet(np.ones(self.n_comps_pos)).rvs([])
        #self.p_neg = dirichlet(np.ones(self.n_comps_neg)).rvs([])
        self.p_neg = self.p_pos
        #self.changeInfo = {'changed': False, 'positive': True, 'mu': True, 'ix':0, 'oldvalue': self.mu_pos[0]}
        self.changeInfo = {'changed': False}
        self.alpha = random()
        self.updateDataGenerator()

    def computePNDataMetrics(self, aucpn_range, irr_vec):
        irr_pr = irr_vec[1]
        n = 10000
        x, y, _, x_p, x_n = self.dg.pn_data(n, 0.5)[0:5]
        posterior_pos = self.dg.pn_posterior_balanced(x_p)
        posterior_neg = 1-self.dg.pn_posterior_balanced(x_n)
        irr_pos = np.mean(np.cast['int32'](posterior_pos > irr_pr).flatten())
        irr_neg = np.mean(np.cast['int32'](posterior_neg > irr_pr).flatten())
        irreducibility = [irr_pos, irr_neg]
        posterior_x = self.dg.pn_posterior_balanced(x)
        fpr, tpr, thresholds = metrics.roc_curve(y, posterior_x)
        aucpn = metrics.auc(fpr, tpr)
        #pdb.set_trace()
        return {'aucpn': aucpn, 'irreducibility': irreducibility}

    def updateDataGenerator(self):
        self.dg = MVNormalMixDG(self.mu_pos, self.sig_pos, self.p_pos, self.mu_neg, self.sig_neg, self.p_neg, self.alpha)

    # def generatePNData(self, n_pos, n_neg):
    #     pos, c_pos = self.dg.data_pos(n_pos)
    #     neg, c_neg = self.dg.data_neg(n_neg)
    #     y = np.concatenate((np.ones([n_pos, 1]), np.zeros([n_neg, 1])), axis=0)
    #     x = np.concatenate((pos, neg), axis=0)
    #     xy = np.concatenate((x, y), axis=1)
    #     #pdb.set_trace()
    #     return xy, x, y, pos, neg, dg

    def perturb2SatisfyMetrics(self, aucpn_range, irr_vec):
        #irr_mid = np.mean(irr_range, axis=0)
        #aucpn_min = min_aucpn(irr_mid)
        # if aucpn_range[0] < aucpn_min:
        #    raise ValueError('Irreducibility range and AUCPN range are not compatible:\n',
        #                      'AUCPN should be above', aucpn_min, 'for midpoint irreducibility of', irr_mid)
        while not self.isMetricSatisfied(aucpn_range, irr_vec):
            self.markRandomParForChange()
            #print(self.changeInfo)
            if self.muMarked():
                self.perturbMu(aucpn_range, irr_vec)
            else:
                if self.pMarked():
                    self.perturbProportion(aucpn_range, irr_vec)
                else:
                    if random() <= 0.5:
                        self.perturbSigmaShape(aucpn_range, irr_vec)
                    else:
                        self.perturbSigmaScale(aucpn_range, irr_vec)
            self.commitChange()

    def perturbMu(self,  aucpn_range, irr_vec):
        print('Mu Perturb')
        c = 1
        delta = np.array([2 * random( ) - 1 for i in np.arange(self.dim)])
        delta = c * delta/np.linalg.norm(delta)
        mu = self.getMarkedParOldValue()
        up = 1.0
        self.proposeChange(mu + up * delta)
        while not self.isMetricUBSatisfied(aucpn_range, irr_vec):
            up = up/2
            self.proposeChange(mu + up * delta)

    def perturbSigmaShape(self,  aucpn_range, irr_vec):
        print('Sigma Shape Perturb')
        newsigma = spd(self.dim)
        sigma = self.getMarkedParOldValue()
        a = 1.0
        self.proposeChange((1-a) * sigma + a * newsigma)
        while (not (self.isMetricUBSatisfied(aucpn_range, irr_vec) and self.acceptableSigma((1-a) * sigma + a * newsigma))):
            a = a/2
            self.proposeChange((1-a) * sigma + a * newsigma)
        if not self.acceptableSigma((1-a) * sigma + a * newsigma):
            self.revert2OldValue()

    def perturbSigmaScale(self, aucpn_range, irr_vec):
        print('Sigma Scale Perturb')
        sigma = self.getMarkedParOldValue()
        a = 0.5
        self.proposeChange(a * sigma)
        while (not (self.isMetricUBSatisfied(aucpn_range, irr_vec) and self.acceptableSigma(a * sigma))):
            a = 1 + (a - 1)/2
            print(a)
            #print('metric:', self.isMetricUBSatisfied(irr_range, aucpn_range))
            #print('acceptable Sigma:', self.acceptableSigma(a * sigma) )
            self.proposeChange(a * sigma)
        if not self.acceptableSigma(a * sigma):
            self.revert2OldValue()

    def perturbProportion(self, aucpn_range, irr_vec):
        print('Perturb Proportion')
        prop = self.getMarkedParOldValue( )
        a = 1
        if self.changeInfo['is_positive']:
            prop_1 = dirichlet(np.ones(self.n_comps_pos)).rvs([])
        else:
            prop_1 = dirichlet(np.ones(self.n_comps_neg)).rvs([])
        new_prop = (1 - a) * prop + a * prop_1
        self.proposeChange(new_prop)
        while not (self.isMetricUBSatisfied(aucpn_range, irr_vec)):
            a = a/2
            new_prop = (1 - a) * prop + a * prop_1
            # print(a)
            self.proposeChange(new_prop)

    def muMarked(self):
        return self.changeInfo['is_mu']

    def pMarked(self):
        return self.changeInfo['is_proportion']

    def acceptableSigma(self, sigma):
        det = np.linalg.det(sigma)
        ratios = np.array([det/np.linalg.det(sig) for sig in self.sig_pos + self.sig_neg])
        print('Sigma Ratios: '+ str(ratios))
        #ratios[:] = 1
        return all(ratios >= 0.5) and all(ratios <= 2.0)

    def isMetricSatisfied(self, aucpn_range, irr_vec):
        #aucpn_range = self.aucpn_range
        anchorSetProp = irr_vec[0]
        metrics = self.computePNDataMetrics(aucpn_range, irr_vec)
        irr_pos = metrics['irreducibility'][0]
        irr_neg = metrics['irreducibility'][1]
        irr_satisfied_pos = anchorSetProp <= irr_pos
        irr_satisfied_neg = anchorSetProp <= irr_neg
        auc_satisfied = aucpn_range[0] <= metrics['aucpn'] <= aucpn_range[1]
        print('isMetricSatisfied' + str(metrics))
        print('anchorSetProp' + str(anchorSetProp))
        return irr_satisfied_pos and auc_satisfied and irr_satisfied_neg

    def isMetricUBSatisfied(self, aucpn_range, irr_vec):
        #anchorSetProp = irr_vec[0]
        metrics = self.computePNDataMetrics(aucpn_range, irr_vec)
        irr_pos = metrics['irreducibility'][0]
        irr_neg = metrics['irreducibility'][1]
        #irr_satisfied_pos = irr_pos >= anchorSetProp
        #irr_satisfied_neg = irr_neg >= anchorSetProp
        auc_satisfied = metrics['aucpn'] <= aucpn_range[1]
        return auc_satisfied

    def proposeChange(self, newValue):
        self.changeInfo['changed'] = True
        V = SN(**self.changeInfo)
        self.updatePar(V.is_positive, V.is_mu, V.is_proportion, V.ix, newValue)

    def commitChange(self):
        self.changeInfo = {'changed': False}

    def updatePar(self, is_positive, is_mu, is_proportion, ix, newValue):
        if is_positive:
            if is_mu:
                self.mu_pos[ix] = newValue
            else:
                if is_proportion:
                    self.p_pos = newValue
                else:
                    self.sig_pos[ix] = newValue
        else:
            if is_mu:
                self.mu_neg[ix] = newValue
            else:
                if is_proportion:
                    self.p_neg = newValue
                else:
                    self.sig_neg[ix] = newValue
        self.updateDataGenerator()

    def markRandomParForChange(self):
        if self.changeInfo['changed']:
            raise ValueError('Attempting to change a new parameter before committing the previous one')
        is_positive = random() < 0.5
        rr = random()
        is_mu = rr < 1.0/3.0
        is_proportion = 1.0/3.0 <= rr <= 2.0/3.0
        ix = np.nan
        if is_positive:
            ix = randint(0, self.n_comps_pos - 1)
            if is_mu:
                value = self.mu_pos[ix]
            else:
                if is_proportion:
                    value = self.p_pos
                else:
                    value = self.sig_pos[ix]
        else:
            ix = randint(0, self.n_comps_neg - 1)
            if is_mu:
                value = self.mu_neg[ix]
            else:
                if is_proportion:
                    value = self.p_neg
                else:
                    value = self.sig_neg[ix]

        self.changeInfo.update({'is_positive': is_positive, 'is_mu': is_mu, 'is_proportion': is_proportion, 'ix': ix, 'oldValue': value})

    def getMarkedParOldValue(self):
        return self.changeInfo['oldValue']

    def revert2OldValue(self):
        print('reverting to old value')
        V = SN(**self.changeInfo)
        self.updatePar(V.is_positive, V.is_mu, V.is_proportion, V.ix, V.oldValue)
        self.changeInfo['changed'] = False
        return



def min_aucpn(irreducibility):
    return irreducibility + (1-irreducibility)/2



class NormalMixPNParameters2:
# Used to generate PN DataGenDistributions. The P and N distributions are mixtures of gaussians with equal number of components
# The positive and negative components are paired. Each pair of components satisfy the AUC and irreducibility constraints
    def __init__(self, dim, nComps):
        self.dim = dim
        self.nComps = nComps
        #self.aucpn_range = aucpn_range
        #self.irr_vec = irr_vec
        self.NormalPNPars = [NormalMixPNParameters(dim, 1) for i in np.arange(nComps)]
        self.alpha = random()
        self.p_pos = dirichlet(np.ones(self.nComps)).rvs([])
        self.p_neg = dirichlet(np.ones(self.nComps)).rvs([])
        self.updateDataGenerator()


    def perturb2SatisfyMetrics(self,  aucpn_range,  irr_vec):
        R = 5
        anchorSetProp = irr_vec[0]
        [self.NormalPNPars[i].perturb2SatisfyMetrics(aucpn_range,  irr_vec) for i in np.arange(self.nComps)]
        #pdb.set_trace()
        if self.nComps > 1:
            for i in np.arange(1, self.nComps):
                pair1 = self.NormalPNPars[i]
                aucs = np.array([self.betweenPairAUC(i, j) for j in range(i)])
                irr_pos, irr_neg = self.irreducibility(irr_vec, i+1)[0:2]
                #pdb.set_trace()
                print('i loop')
                print('aucs' + str(np.min(aucs)))
                print('irr_pos' + str(irr_pos))
                print('irr_neg' + str(irr_neg))
                if np.min(irr_pos) < anchorSetProp or np.min(irr_neg) < anchorSetProp or np.min(aucs) < aucpn_range[0]:
                    mu_pos = np.copy(pair1.mu_pos[0])
                    mu_neg = np.copy(pair1.mu_neg[0])
                    mu_pos_best = mu_pos
                    mu_neg_best = mu_neg
                    bestAUC = 1
                    for j in np.arange(R):
                        pair1.mu_pos[0] = np.copy(mu_pos)
                        pair1.mu_neg[0] = np.copy(mu_neg)
                        irr_pos, irr_neg = self.irreducibility(irr_vec, i + 1)[0:2]
                        aucs = np.array([self.betweenPairAUC(i, j) for j in range(i)])
                        vec = np.array([2 * random() - 1 for i in np.arange(self.dim)])
                        vec = 0.5 * vec / np.sqrt(self.dim)
                        vec = vec / np.linalg.norm(vec)
                        #pdb.set_trace()
                        while np.min(irr_pos) < anchorSetProp or np.min(irr_neg) < anchorSetProp \
                                or np.min(aucs) < aucpn_range[0]:
                            pair1.mu_pos[0] = pair1.mu_pos[0] + vec
                            pair1.mu_neg[0] = pair1.mu_neg[0] + vec
                            irr_pos, irr_neg = self.irreducibility(irr_vec, i+1)[0:2]
                            aucs = np.array([self.betweenPairAUC(i, j) for j in range(i)])
                            print('first loop in j loop')
                            print('aucs' + str(np.min(aucs)))
                            print('irr_pos' + str(irr_pos))
                            print('irr_neg' + str(irr_neg))
                            #pdb.set_trace()
                        #pdb.set_trace()
                        ix = np.random.choice(i,1)
                        pair2 = self.NormalPNPars[ix[0]]
                        mean1 = (pair1.mu_pos[0] + pair1.mu_neg[0])/2
                        mean2 = (pair2.mu_pos[0] + pair2.mu_neg[0])/2
                        vec2 = mean2-mean1
                        a1 = 0
                        a2 = 1
                        aa1 = 0
                        iii = 1
                        while np.min(aucs) < aucpn_range[0] or np.min(aucs) > aucpn_range[1] or \
                                np.min(irr_pos) < anchorSetProp or np.min(irr_neg) < anchorSetProp:
                            if np.abs(a1 - a2) < 10**-5:
                                break
                            aa2 = (a1 + a2)/2
                            pair1.mu_pos[0] = pair1.mu_pos[0] + (aa2 - aa1)*vec2
                            pair1.mu_neg[0] = pair1.mu_neg[0] + (aa2 - aa1)*vec2
                            aucs = np.array([self.betweenPairAUC(i, j) for j in range(i)])
                            irr_pos, irr_neg = self.irreducibility(irr_vec, i + 1)[0:2]
                            aa1 = aa2
                            print('second loop in j loop')
                            print('aucs' + str(np.min(aucs)))
                            print('irr_pos' + str(irr_pos))
                            print('irr_neg' + str(irr_neg))
                            if np.min(aucs) < aucpn_range[0] or np.min(irr_pos) < anchorSetProp or \
                                    np.min(irr_neg) < anchorSetProp:
                                a2 = aa2
                            else:
                                a1 = aa2
                            iii = iii + 1
                            if iii == 1000:
                                pdb.set_trace()

                        #pdb.set_trace()
                        print('after two j loops')
                        print('aucs' + str(np.min(aucs)))
                        print('irr_pos' + str(irr_pos))
                        print('irr_neg' + str(irr_neg))
                        if np.min(aucs) >= aucpn_range[0] and np.min(aucs) <= aucpn_range[1] and \
                                np.min(irr_pos) > anchorSetProp and np.min(irr_neg) > anchorSetProp:
                            mu_pos_best = np.copy(pair1.mu_pos[0])
                            mu_neg_best = np.copy(pair1.mu_neg[0])
                            break
                        if np.min(aucs) < bestAUC:
                            bestAUC = np.min(aucs)
                            mu_pos_best = np.copy(pair1.mu_pos[0])
                            mu_neg_best = np.copy(pair1.mu_neg[0])



                    pair1.mu_pos[0] = mu_pos_best
                    pair1.mu_neg[0] = mu_neg_best
                    [self.NormalPNPars[i].updateDataGenerator() for i in np.arange(self.nComps)]
                    self.updateDataGenerator()
                    aucs = np.array([self.betweenPairAUC(i, j) for j in range(i)])
                    irr_pos, irr_neg = self.irreducibility(irr_vec, i + 1)[0:2]
                    #pdb.set_trace()
                    print('after j loops')
                    print('aucs' + str(np.min(aucs)))
                    print('irr_pos' + str(irr_pos))
                    print('irr_neg' + str(irr_neg))


    def irreducibility(self, irr_vec, ncmps=None):
        if ncmps is None:
            ncmps = self.nComps
        irr_p = irr_vec[1]
        isStringent = irr_vec[2]
        [self.NormalPNPars[i].updateDataGenerator() for i in np.arange(ncmps)]
        posComps = []
        negComps = []
        for nPNPars in self.NormalPNPars[0:ncmps]:
            posComps.append(nPNPars.dg.dist_pos)
            negComps.append(nPNPars.dg.dist_neg)
        pSamp = [pDist.rvs(size=1000) for pDist in posComps]
        nSamp = [nDist.rvs(size=1000) for nDist in negComps]
        p_probs = [[cmp.pdf(smp) for cmp in posComps + negComps] for smp in pSamp]
        n_probs = [[cmp.pdf(smp) for cmp in negComps + posComps] for smp in nSamp]
        if isStringent:
            post_ps = [p_prob[i]/sum(p_prob) for (i, p_prob) in enumerate(p_probs)]
            post_ns = [n_prob[i]/sum(n_prob) for (i, n_prob) in enumerate(n_probs)]
            irr_pos = np.array([np.mean(post_p > irr_p) for post_p in post_ps])
            irr_neg = np.array([np.mean(post_n > irr_p) for post_n in post_ns])
        else:
            post_ps = [[p_prob[i] / (p_prob[i] + p_prob[j]) for j in np.arange(len(p_prob)) if j != i] for (i, p_prob) in enumerate(p_probs)]
            post_ns = [[n_prob[i] / (n_prob[i] + n_prob[j]) for j in np.arange(len(n_prob)) if j != i] for (i, n_prob) in enumerate(n_probs)]
            irr_pos = np.array([[np.mean(pp > irr_p) for pp in post_p] for post_p in post_ps])
            irr_neg = np.array([[np.mean(pn > irr_p) for pn in post_n] for post_n in post_ns])
        #pdb.set_trace()
        return irr_pos, irr_neg, post_ps, post_ns

    def vecBetweenPair(self, i, j):
        pairi = self.NormalPNPars[i]
        pairj = self.NormalPNPars[j]
        mu_posi = pairi.mu_pos[0]
        mu_posj = pairj.mu_pos[0]
        mu_negi = pairi.mu_neg[0]
        mu_negj = pairj.mu_neg[0]
        return (mu_posi + mu_negi)/2 - (mu_posj + mu_negj)/2

    def distBetweenPair(self, i, j):
        v = self.vecBetweenPair(i,j)
        return np.linalg.norm(v)


    def betweenAllPairsAUC(self):
        #cmb = combinations(np.arange(self.nComps), 2)
        aucs = [self.betweenPairAUC(i, j) for i, j in combinations(np.arange(self.nComps), 2)]
        #[print(str(i)+str(j)) for i, j in cmb]
        return aucs

    def betweenPairAUC(self, i, j):
        #pdb.set_trace()
        self.NormalPNPars[i].updateDataGenerator()
        self.NormalPNPars[j].updateDataGenerator()
        posi = self.NormalPNPars[i].dg.dist_pos
        negi = self.NormalPNPars[i].dg.dist_neg
        posj = self.NormalPNPars[j].dg.dist_pos
        negj = self.NormalPNPars[j].dg.dist_neg
        auc1 = AUCFromDistributions(posi, posj)
        auc2 = AUCFromDistributions(posi, negj)
        auc3 = AUCFromDistributions(negi, posj)
        auc4 = AUCFromDistributions(negi, negj)
        return [auc1, auc2, auc3, auc4]

    def withinPairAUC(self):
        #pdb.set_trace()
        [self.NormalPNPars[i].updateDataGenerator() for i in np.arange(self.nComps)]
        aucs = [AUCFromDistributions(pnPar.dg.dist_pos, pnPar.dg.dist_neg)  for pnPar in self.NormalPNPars]
        return aucs

    def updateDataGenerator(self):
        self.mu_pos = [self.NormalPNPars[i].mu_pos[0] for i in np.arange(self.nComps)]
        self.mu_neg = [self.NormalPNPars[i].mu_neg[0] for i in np.arange(self.nComps)]
        self.sig_pos = [self.NormalPNPars[i].sig_pos[0] for i in np.arange(self.nComps)]
        self.sig_neg = [self.NormalPNPars[i].sig_neg[0] for i in np.arange(self.nComps)]
        self.dg = MVNormalMixDG(self.mu_pos, self.sig_pos, self.p_pos, self.mu_neg, self.sig_neg, self.p_neg, self.alpha)

    def plotData(self, nComps=None):
        if nComps is None:
            nComps = self.nComps
        self.updateDataGenerator()
        posComps = self.dg.dist_pos.comps[0:nComps]
        negComps = self.dg.dist_neg.comps[0:nComps]
        pSamp = [pDist.rvs(size=1000) for pDist in posComps]
        nSamp = [nDist.rvs(size=1000) for nDist in negComps]
        [plt.scatter(s[:, 0], s[:, 1]) for s in pSamp]
        [plt.scatter(s[:, 0], s[:, 1]) for s in nSamp]
        plt.show()
        posComps = [nPNPars.dg.dist_pos for nPNPars in self.NormalPNPars[0:nComps]]
        negComps = [nPNPars.dg.dist_neg for nPNPars in self.NormalPNPars[0:nComps]]
        pSamp = [pDist.rvs(size=1000) for pDist in posComps]
        nSamp = [nDist.rvs(size=1000) for nDist in negComps]
        [plt.scatter(s[:, 0], s[:, 1]) for s in pSamp]
        [plt.scatter(s[:, 0], s[:, 1]) for s in nSamp]
        plt.show()

    def plotCIEllipse(self, compsDict=None, ax=None):
        if compsDict is None:
            compsDict = {'pos': [i for i in np.arange(self.nComps)], 'neg': [i for i in np.arange(self.nComps)]}
        if ax is None:
           fig, ax = plt.subplots()
        posComps = self.dg.dist_pos.comps
        negComps = self.dg.dist_neg.comps
        pos_ix = compsDict['pos']
        neg_ix = compsDict['neg']
        posComps = [posComps[ix] for ix in pos_ix]
        negComps = [negComps[ix] for ix in neg_ix]
        [CIEllipse(cmp.mean, cmp.cov, ax, edgecolor='red') for cmp in posComps]
        [CIEllipse(cmp.mean, cmp.cov, ax, edgecolor='k') for cmp in negComps]
        pSamp = [pDist.rvs(size=100) for pDist in posComps]
        nSamp = [nDist.rvs(size=100) for nDist in negComps]
        [plt.scatter(s[:, 0], s[:, 1], alpha=0.5) for s in pSamp]
        [plt.scatter(s[:, 0], s[:, 1], alpha=0.5) for s in nSamp]
        if len(pos_ix + neg_ix)>2:
            ax.set_title(str(compsDict))
        elif len(pos_ix + neg_ix) == 2:
            comps = posComps + negComps
            auc = AUCFromDistributions(comps[0], comps[1])
            #pdb.set_trace()
            ax.set_title(str(compsDict)+': auc '+str(auc))

        plt.show()




    # def perturb2SatisfyMetrics(self, irr_range, aucpn_range):
    #     [self.NormalPNPars[i].perturb2SatisfyMetrics(irr_range, aucpn_range) for i in np.arange(self.nComps)]
    #     #pdb.set_trace()
    #     if self.nComps > 1:
    #         for i in np.arange(self.nComps):
    #             pair1 = self.NormalPNPars[i]
    #             aucs = np.array([self.betweenPairAUC(i, j) for j in range(self.nComps) if j != i])
    #             irr_pos, irr_neg = self.irreducibility()
    #             pdb.set_trace()
    #             a1 = 0
    #             a2 = 1
    #
    #             if np.min(aucs) < aucpn_range[0] or min(irr_pos[0:(i+1)]) < irr_range[0] or min(irr_neg[0:(i+1)]) < irr_range[0]:
    #                 vec = np.array([2 * random() - 1 for i in np.arange(self.dim)])
    #                 vec = 0.5 * vec / np.sqrt(self.dim)
    #                 vec = vec / np.linalg.norm(vec)
    #                 au = np.inf
    #                 al = 0
    #                 while (np.min(aucs) < aucpn_range[0] or np.min(aucs) > aucpn_range[1] or
    #                        min(irr_pos[0:(i+1)]) < irr_range[0] or min(irr_neg[0:(i+1)]) < irr_range[0]) and np.abs(a2-a1) > 10**-5:
    #                     pair1.mu_pos[0] = pair1.mu_pos[0] - a1 * vec + a2 * vec
    #                     pair1.mu_neg[0] = pair1.mu_neg[0] - a1 * vec + a2 * vec
    #                     a1 = a2
    #                     aucs = np.array([self.betweenPairAUC(i, j) for j in range(self.nComps) if j != i])
    #                     irr_pos, irr_neg = self.irreducibility()
    #                     if np.min(aucs) < aucpn_range[1] or min(irr_pos[0:(i+1)]) < irr_range[0] or min(irr_neg[0:(i+1)]) < irr_range[0]:
    #                         al = a2
    #                         if not np.isinf(au):
    #                             a2 = (a2 + au)/2
    #                         else:
    #                             a2 = a2*2
    #                     elif np.min(aucs) > aucpn_range[1]:
    #                         au = a2
    #                         a2 = (a2 + al)/2
    #             elif np.min(aucs) > aucpn_range[1]:
    #                 pairDist = np.array([self.distBetweenPair(i, j) for j in range(self.nComps)])
    #                 pairDist[i] = np.inf
    #                 ix = np.argmin(pairDist)
    #                 vec = self.vecBetweenPair(ix, i)
    #                 vec = vec / np.linalg.norm(vec)
    #                 au = 0
    #                 al = 1
    #                 while np.min(aucs) < aucpn_range[0] and np.min(aucs) > aucpn_range[1] and np.abs(a2-a1) > 10**-5:
    #                     pair1.mu_pos[0] = pair1.mu_pos[0] - a1 * vec + a2 * vec
    #                     pair1.mu_neg[0] = pair1.mu_neg[0] - a1 * vec + a2 * vec
    #                     a1 = a2
    #                     aucs = np.array([self.betweenPairAUC(i, j) for j in range(self.nComps) if j != i])
    #                     irr_pos, irr_neg = self.irreducibility()
    #                     if np.min(aucs) < aucpn_range[1] or min(irr_pos[0:(i+1)]) < irr_range[0] or min(irr_neg[0:(i+1)]) < irr_range[0]:
    #                         al = a2
    #                         a2 = (a2 + au)/2
    #                     elif np.min(aucs) > aucpn_range[1]:
    #                         au = a2
    #                         a2 = (a2 + al)/2
    #                 #pdb.set_trace()
    #     self.updateDataGenerator()

def AUCFromDistributions(dist1, dist2):
    dg = DataGenerator(dist1, dist2, 0.5)
    n=5000
    x, y = dg.pn_data(n)[0:2]
    posterior_x = dg.pn_posterior_balanced(x)
    fpr, tpr, thresholds = metrics.roc_curve(y, posterior_x)
    aucpn = metrics.auc(fpr, tpr)
    return aucpn