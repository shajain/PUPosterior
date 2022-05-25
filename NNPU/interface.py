from PUPosterior.data import DataPU
import copy
from NNPU.train import select_loss, select_model, PULoss, make_optimizer, MultiUpdater
import numpy as np
import chainer
from chainer import functions as F
class NNPU:
    def __init__(self, dataPU, prior, fold=1):

        BS = 1000
        gamma = 1.0
        beta = 0.0
        stepsize = 1e-3
        gpu = -1
        epoch = 10000
        out = 'save/NNPU'
        data_tr, data_te, data_val = dataPU.getTTVData()
        dim = data_tr['x'].shape[1]
        X = np.vstack([data_tr['x1'], data_tr['x']]).astype('float32')
        n1 = data_tr['x1'].shape[0]
        n = data_tr['x'].shape[0]
        Y = np.vstack([np.ones((n1,1)), np.zeros((n,1))]).astype('int32')
        XYtrain = list(zip(X, Y.flatten()))
        X = np.vstack([data_val['x1'], data_val['x']])
        n1 = data_val['x1'].shape[0]
        n = data_val['x'].shape[0]
        Y = np.vstack([np.ones((n1, 1)), np.zeros((n, 1))])
        XYval = list(zip(X, Y.flatten()))
        train_iter = chainer.iterators.SerialIterator(XYtrain, BS)
        valid_iter = chainer.iterators.SerialIterator(XYval, BS, repeat=False, shuffle=False)
        #test_iter = chainer.iterators.SerialIterator(XYtest, args.batchsize, repeat=False, shuffle=False)

        # model setup
        loss_type = select_loss('sigmoid')
        selected_model = select_model('mlp')
        self.model = selected_model(prior, dim)
        models = {"nnPU": copy.deepcopy(self.model)}
        loss_funcs = {"nnPU": PULoss(prior, loss=loss_type, nnpu=True, gamma=gamma, beta=beta)}
        #if args.gpu >= 0:
        #    for m in models.values():
        #        m.to_gpu(args.gpu)

        # trainer setup
        optimizers = {k: make_optimizer(v, stepsize) for k, v in models.items()}
        updater = MultiUpdater(train_iter, optimizers, models, device=gpu, loss_func=loss_funcs)
        trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=out)
        #trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
        #train_01_loss_evaluator = MultiPUEvaluator(prior, valid_iter, models, device=args.gpu)
        #train_01_loss_evaluator.default_name = 'train'
        #trainer.extend(train_01_loss_evaluator)
        #trainer.extend(MultiEvaluator(test_iter, models, device=args.gpu))
        #trainer.extend(extensions.ProgressBar())
        #trainer.extend(extensions.PrintReport(
        #    ['epoch', 'train/nnPU/error', 'test/nnPU/error', 'train/uPU/error', 'test/uPU/error', 'elapsed_time']))
        #if extensions.PlotReport.available():
        #    trainer.extend(
        #        extensions.PlotReport(['train/nnPU/error', 'train/uPU/error'], 'epoch', file_name='training_error.png'))
        #    trainer.extend(
        #        extensions.PlotReport(['test/nnPU/error', 'test/uPU/error'], 'epoch', file_name='test_error.png'))
        print("prior: {}".format(prior))
        print("batchsize: {}".format(BS))
        print("model: {}".format(selected_model))
        print("beta: {}".format(beta))
        print("gamma: {}".format(gamma))
        print("")

        # run training
        trainer.run()

    def posterior(self, x):
        presigmoid = self.model(x)
        posterior = F.sigmoid(presigmoid)
        return posterior.data

