import pdb

from plots import sortedplot as sp
from IPython.display import display
import numpy as np
from sklearn.metrics import roc_curve, auc
from plots.binBy import binBy



class Debug:

    def __init__(self):
        fig, axs = sp.subplots(2, 3, figsize=(10, 7))
        self.fig = fig
        self.fig.tight_layout(pad=1.0)
        self.axs = axs

    def attachData(self, data, posterior=None):
        self.data = data
        self.x1 = data.x1
        self.x0 = data.x0
        self.dim = self.x1.shape[1]
        self.hasPosterior = self.data.hasKeyPos('posterior')
        self.data_tr, self.data_te, self.data_val = self.data.getTTVDataXY()
        self.hasY = True
        self.aucEst_tr = []
        self.aucEst_te = []
        self.aucEst_val = []

    def attachTrainer(self, trainer):
        self.trainer = trainer

    def plotLossHistory(self):
        losses_tr = self.trainer.losses
        losses_val = self.trainer.valLosses
        sp.sortedplot(losses_tr, label='Tr', ax=self.axs[0, 0])
        sp.sortedplot(losses_val, label='Val', ax=self.axs[0, 0])
        self.axs[0, 0].set_title('PU Loss')

    def plotPosteriorFit(self):
        net = self.trainer.nnLoss
        bestNN = self.trainer.bestNNLoss
        x_val = self.data_val['x']
        x_tr = self.data_tr['x']
        # x1 = data_val['x1']
        postEst_val = net.posterior(x_val)
        postEst_tr = net.posterior(x_tr)
        # postBest = bestNN.posterior(x_val)
        # self.posteriorFit.append(posteriorFit)
        if self.hasPosterior:
            postTrue_tr, _, postTrue_val = self.data.getTTVKeyPosNeg('posterior')
            if self.dim == 1:
                sp.sortedplot(x_val, postTrue_val, label='True', ax=self.axs[1, 0])
                #pdb.set_trace()
            else:
                # pdb.set_trace()
                nBins = 10
                bins = np.linspace(0.1, 0.9, nBins - 1)
                ix = np.digitize(postTrue_tr, bins)
                # pdb.set_trace()
                errors_tr = postTrue_tr - postEst_tr
                self.axs[1, 0].boxplot([errors_tr[ix == i] for i in np.arange(10)])
                tickStr = ['.05']
                tickStr.extend([('{n:.2f}'.format(n=(bins[i] + bins[i + 1]) / 2))[1:] for i in np.arange(nBins - 2)])
                tickStr.extend(['.95'])
                self.axs[1, 0].set_xticklabels(tickStr)
                self.axs[1, 0].set_title('Posterior Error Train')
                self.axs[1, 0].set_xlabel('True posterior')
                # errorsBest = postTrue - postBest
                # self.axs[2, 1].boxplot([errorsBest[ix == i] for i in np.arange(10)])
                # tickStr = ['l']
                # tickStr.extend([('{n:.1f}'.format(n=(bins[i] + bins[i + 1]) / 2))[1:] for i in np.arange(nBins - 2)])
                # tickStr.extend(['r'])
                # self.axs[2, 1].set_xticklabels(tickStr)
                ix = np.digitize(postTrue_val, bins)
                # pdb.set_trace()
                errors_val = postTrue_val - postEst_val
                self.axs[1, 1].boxplot([errors_val[ix == i] for i in np.arange(10)])
                tickStr = ['.05']
                tickStr.extend([('{n:.2f}'.format(n=(bins[i] + bins[i + 1]) / 2))[1:] for i in np.arange(nBins - 2)])
                tickStr.extend(['.95'])
                self.axs[1, 1].set_xticklabels(tickStr)
                self.axs[1, 1].set_title('Posterior Error Val')
                self.axs[1, 1].set_xlabel('True posterior')
                if not hasattr(self, 'postMAE'):
                    self.postMAE = {'train': [], 'val': []}
                self.postMAE['train'].append(np.mean(np.abs(errors_tr)))
                self.postMAE['val'].append(np.mean(np.abs(errors_val)))
                sp.sortedplot(self.postMAE['train'], label='Tr', ax=self.axs[1, 2])
                sp.sortedplot(self.postMAE['val'], label='Val', ax=self.axs[1, 2])
                self.axs[1, 2].set_title('Posterior MAE')

        if self.dim == 1:
            # sp.sortedplot(x_val, postBest, label='Best', ax=self.axs[2, 0])
            sp.sortedplot(x_val, postEst_val, label='Est.', ax=self.axs[1, 0])
        elif not self.hasPosterior:
            self.axs[1, 0].hist(postEst_val, bins=20, density=True)

        # pdb.set_trace()
        # if len(self.posteriorFit) >=2 and self.dim == 1:
        #    sp.sortedplot(self.x, self.posteriorFit[-2], label='Posterior Est. Old',  ax=self.axs[1, 0])
        # pdb.set_trace()

    def plotAUCs(self):
        net = self.trainer.nnLoss
        bestNN = self.trainer.bestNNLoss
        x_tr = self.data_tr['x']
        x_te = self.data_te['x']
        x_val = self.data_val['x']
        y_tr = self.data_tr['y']
        y_te = self.data_te['y']
        y_val = self.data_val['y']
        if self.hasPosterior:
            pTrue_tr, pTrue_te, pTrue_val = self.data.getTTVKeyPosNeg('posterior')
            fpr, tpr, _ = roc_curve(y_tr, pTrue_tr)
            aucTrue_tr = auc(fpr, tpr)
            self.axs[0, 1].axhline(aucTrue_tr)
            fpr, tpr, _ = roc_curve(y_te, pTrue_te)
            aucTrue_te = auc(fpr, tpr)
            fpr, tpr, _ = roc_curve(y_val, pTrue_val)
            aucTrue_val = auc(fpr, tpr)
            self.axs[0, 2].axhline(aucTrue_val)

        pEst_tr = net.posterior(x_tr)
        pEst_te = net.posterior(x_te)
        pEst_val = net.posterior(x_val)
        pBest_tr = bestNN.posterior(x_tr)
        pBest_te = bestNN.posterior(x_te)
        pBest_val = bestNN.posterior(x_val)

        fpr, tpr, _ = roc_curve(y_tr, pEst_tr)
        self.aucEst_tr.append(auc(fpr, tpr))
        self.axs[0, 1].plot(self.aucEst_tr)
        self.axs[0, 1].set_title('AUC Tr')
        fpr, tpr, _ = roc_curve(y_te, pEst_te)
        self.aucEst_te.append(auc(fpr, tpr))
        fpr, tpr, _ = roc_curve(y_val, pEst_val)
        self.aucEst_val.append(auc(fpr, tpr))
        self.axs[0, 2].plot(self.aucEst_val)
        self.axs[0, 2].set_title('AUC Val')
        fpr, tpr, _ = roc_curve(y_tr, pBest_tr)
        # aucBest_tr = auc(fpr, tpr)
        # self.axs[1, 2].axhline(aucBest_tr)
        # fpr, tpr, _ = roc_curve(y_te, pBest_te)
        # aucBest_te=auc(fpr, tpr)
        # fpr, tpr, _ = roc_curve(y_val, pBest_val)
        # aucBest_val=auc(fpr, tpr)
        # self.axs[1, 2].axhline(aucBest_val)

    def afterUpdate(self):
        print('after Update')
        self.plotPosteriorFit()
        self.plotLossHistory()
        if self.hasY:
            self.plotAUCs()
        self.displayPlots()
        # sp.show()

    def beforeUpdate(self, iter):
        if np.remainder(iter, 10) == 0:
            print('Iteration' + str(iter))
        return

    def beforeTraining(self):
        # print('before Training')
        return

    def displayPlots(self):
        for axs in self.axs.reshape(-1):
            # pdb.set_trace()
            axs.legend()
        display(self.fig)
        sp.close()
        for axs in self.axs.reshape(-1):
            axs.clear()
        fig, axs = sp.subplots(2, 3, figsize=(10, 7))
        self.fig = fig
        self.fig.tight_layout(pad=1.0)
        self.axs = axs