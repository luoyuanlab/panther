import torch
from torch import nn
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import utils
import pandas as pd
from sklearn.metrics import f1_score
import tensorly as tl
from tensorly.kruskal_tensor import kruskal_to_tensor
tl.set_backend('pytorch')

## this implements the transductive unsupervised non-negative CP factorization with orthogonality constraint on 1st (patient) mode
class NCP(nn.Module):
    def __init__(self, rank, device = torch.device('cpu'),
                 n_iter = 10, eps = 1e-7, wortho=1, early_stopping = 10, 
                 floss = 'l2', weight_decay = 1e-5, tol = 1e-5,
                 lr = 1e-2, verbose = False, seed=None, fn=None):
        super(NCP, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.n_iter = n_iter
        self.floss = floss
        self.weight_decay = weight_decay
        self.tol = tol
        self.lr = lr
        self.verbose = verbose
        self.eps = eps
        self.fn = fn
        self.wortho = wortho
        self.early_stopping = early_stopping
        self.decomposed = False
        self.device = device
        self.report = defaultdict(list)
        self.rank = rank

    def __initfact__(self, X):
        self.scale = np.cbrt(torch.mean(X).cpu() / self.rank)
        self.factors = nn.ParameterList([nn.Parameter(tl.tensor(self.scale * torch.rand((X.shape[i], self.rank)), device=self.device)) for i in range(tl.ndim(X))])  # normal init
        self.identity = torch.eye(self.rank, device=self.device)

        if self.floss == 'l2':
            self.loss_fac = utils.l2
        elif self.floss == 'kl':
            self.loss_fac = utils.kl_div
        elif self.floss == 'l2_sp':
            self.loss_fac = utils.l2_sparse

        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def to(self,device):
        self.device = device
        return super(NCP, self).to(device)

    def plus(self,X):
        X[X < 0] = 0 # self.eps
        return X

    def __autograd__(self, X, epoch):
        """
           autograd update, using ReLU for non-negativity
        """
        self.opt.zero_grad()

        l = self.loss_fac(kruskal_to_tensor(self.factors), X)
        # add l2 regularization for orthogonality
        AtA = torch.mm(torch.t(self.factors[0]), self.factors[0])

        AtA = AtA/torch.mean(AtA)/self.rank
        l += self.wortho * self.loss_fac(AtA, self.identity) # scale invariant orthogonal
        # add l2 regularization for magnitude
        # for p in self.parameters():
        #     l += self.mreg*torch.mean(p**2) # or to just penalize the max value
        l.backward()
        self.opt.step()
        for f in self.factors:
            f.data = self.plus(f.data)
        return l.item()


    def __update__(self, X, epoch):
        l = self.__autograd__(X, epoch)

        self.report['epoch'].append(epoch)
        self.report['loss'].append(l)
        if self.verbose and epoch % 100 == 0:
            print("%d\tloss: %.4f"%(epoch,l))


    def fit(self, X):
        it = range(self.n_iter)
        # for autograd solver
        for e in it:
            self.__update__(X, e)
            lcur = self.report['loss'][-1]
            lold = np.mean(self.report['loss'][-(self.early_stopping+1):-1])
            if e > self.early_stopping and (lold - lcur)/lold < self.tol:
                print("Early stopping...")
                break

        self.decomposed = True

        torch.save({'epoch': max(it)+1,
                    'state_dict': self.state_dict(),
                    'optimizer': self.opt.state_dict(),
                    'report': self.report,
        }, self.fn)
        return self

    def show_report(self):
        return pd.DataFrame(self.report)


    def fit_transform(self, X):
        if not self.decomposed:
            X = tl.tensor(X, device=self.device)
            self.__initfact__(X)
            self.fit(X)
            # detach all params including the linear fc layer
            for p in self.parameters():
                p.requires_grad = False
        del X # release memory
        torch.cuda.empty_cache()

        if self.device.type == 'cuda':
            return [factor.detach().cpu().numpy() for factor in self.factors]
        else:
            return [factor.detach().numpy() for factor in self.factors]




