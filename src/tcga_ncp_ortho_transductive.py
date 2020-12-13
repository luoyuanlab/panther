import pandas as pd
import pickle
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score, log_loss
from math import floor, ceil
import numpy as np
import os
import os.path
import sys
import importlib
import torch
from NCPotr import NCP
from time import time
from datetime import timedelta
from tensorly import kruskal_to_tensor, tensor
import warnings
warnings.filterwarnings(action='ignore')
import getopt

opts, extraparams = getopt.getopt(sys.argv[1:], 's:c:i:l:r:w:m:',
                                  ['seed=', 'iter=', 'loss=','lr=','weight_decay=', 'mode='])

print(opts)

devstr = 'cuda'

niter = 6000
seed = 1444055 # generally should pick a large odd number for good quality pseudo random number generation
loss_fun = 'l2'
lr=0.05
weight_decay=0
mode = 'co2' # co-occurrence counting scheme as specified in the paper

for o,p in opts:
    if o in ['-s', '--seed']:
        seed = int(p)
    if o in ['-i', '--iter']:
        niter = int(p)
    if o in ['-l', '--loss']:
        loss_fun = p
    if o in ['-r', '--lr']:
        lr = float(p)
    if o in ['-w', '--weight_decay']:
        weight_decay = float(p)
    if o in ['-m', '--mode']:
        mode = p

# TODO: please specify your root genetic data directory
dn = 'genetic data root'

# the pickle file contains the feature matrix tcga_mat (subject by gene), the confounding variables (e.g., patient demographic) pts_sel, and the label y
f = open(f'{dn}/tcga_cf.pik', 'rb')
[tcga_mat, pts_sel, y] = pickle.load(f)
f.close()
sel_pts = np.array(pts_sel)



# the pickle file contains the feature tensor tcga_t and the label y
# the tcga_t tensor has 3 modes: subjects, pathway and genes, in that order
f = open(f'{dn}/tcga_t{mode}.pik', 'rb')

[tcga_t, y] = pickle.load(f)
f.close()
print('tensor shape: {0} x {1} x {2}'.format(*tcga_t.shape))

train_indices = pd.read_csv(f'{dn}/train_indices_0.2val_0.2te.csv', header=None)
test_indices = pd.read_csv(f'{dn}/test_indices_0.2val_0.2te.csv', header=None)
val_indices = pd.read_csv(f'{dn}/val_indices_0.2val_0.2te.csv', header=None)

X = tcga_t
y, yuniques = pd.factorize(y, sort=True)


ncs = range(50,501,50)
X = X.astype(np.float32)
device = torch.device(devstr)
r = 0
train_index = train_indices[r]; val_index = val_indices[r]; test_index = test_indices[r]
X_train, X_val, X_test = X[train_index], X[val_index], X[test_index]
y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
pts_tr, pts_val, pts_te = sel_pts[train_index], sel_pts[val_index], sel_pts[test_index]    


ntr = X_train.shape[0]
nval = X_val.shape[0]
X = np.concatenate((X_train, X_val, X_test))

print('nc,C,tr acc,val acc,te acc,w2,b2,mse_tr,mse_val,mse_te,celoss,telap')
for nc in ncs:
    if mode != '':
        fn = f'{dn}/ncp_{mode}_{loss_fun}/ortho_transductive/ncp_{nc}.p'
    else:
        fn = f'{dn}/ncp_{loss_fun}/ortho_transductive/ncp_{nc}.p'
    if os.path.isfile(fn):
        chkpt = torch.load(fn)
        factors = [chkpt['state_dict']['factors.0'].cpu().numpy(),
                   chkpt['state_dict']['factors.1'].cpu().numpy(),
                   chkpt['state_dict']['factors.2'].cpu().numpy()]
        X_tr_ncp = factors[0][:ntr,:]
        X_val_ncp = factors[0][ntr:(ntr+nval),:]
        X_te_ncp = factors[0][(ntr+nval):,:]
        telap = chkpt['telap']
        mse_tr = chkpt['mse_tr']
        mse_val = chkpt['mse_val']
        mse_te = chkpt['mse_te']
    else:
        m = NCP(rank=nc, n_iter=niter, weight_decay=weight_decay, lr=lr, floss=loss_fun, seed=seed, fn=fn)
        m.to(device)
        start_time = time()
        factors = m.fit_transform(X) 
        # print(factors)
        X_tr_ncp = factors[0][:ntr,:]
        X_val_ncp = factors[0][ntr:(ntr+nval),:]
        X_te_ncp = factors[0][(ntr+nval):,:]
        elapsed_time = time() - start_time
        telap = str(timedelta(seconds=elapsed_time))
        err = kruskal_to_tensor([tensor(X_tr_ncp)] + [tensor(f) for f in factors[1:]]).numpy() - X_train
        # mse_tr = m.show_report().loc[niter-1,'loss']
        mse_tr = np.square(err).mean(axis=None)
        err = kruskal_to_tensor([tensor(X_val_ncp)] + [tensor(f) for f in factors[1:]]).numpy() - X_val
        mse_val = np.square(err).mean(axis=None)
        err = kruskal_to_tensor([tensor(X_te_ncp)] + [tensor(f) for f in factors[1:]]).numpy() - X_test
        mse_te = np.square(err).mean(axis=None)
        del err

        chkpt = torch.load(fn)
        chkpt['telap'] = telap
        chkpt['mse_tr'] = mse_tr
        chkpt['mse_val'] = mse_val
        chkpt['mse_te'] = mse_te
        torch.save(chkpt, fn)
        del chkpt
    torch.cuda.empty_cache()

    for C in [0.01, 0.1, 1, 10, 100]: 
        clf = LogisticRegression(C=C, solver='lbfgs', multi_class='multinomial', max_iter=1000) 
        clf.fit(np.hstack((pts_tr, X_tr_ncp)), y_train)
        w2 = np.square(clf.coef_).sum(axis=None)
        b2 = np.square(clf.intercept_).sum(axis=None)

        y_tr_pred = clf.predict(np.hstack((pts_tr, X_tr_ncp)))
        y_val_pred = clf.predict(np.hstack((pts_val, X_val_ncp)))
        y_te_pred = clf.predict(np.hstack((pts_te, X_te_ncp)))
        y_tr_prob = clf.predict_proba(np.hstack((pts_tr, X_tr_ncp)))
        nll_tr = log_loss(y_train, y_tr_prob)

        acctr = accuracy_score(y_train, y_tr_pred) 
        accval = accuracy_score(y_val, y_val_pred) 
        accte = accuracy_score(y_test, y_te_pred) 
        print('%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s' % (nc, C, acctr, accval, accte, w2, b2, mse_tr, mse_val, mse_te, nll_tr, telap))
