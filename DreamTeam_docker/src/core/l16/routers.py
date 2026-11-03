# -*- coding: utf-8 -*-
# VENDORED VERBATIM from agents/L16/src/routers.py @ origin/agent/L16.
# The ONLY edits are the removal of the module-level absolute paths and of the two data
# loaders that used them (`common.load`, `feats2.load2`); every arithmetic line is L16's,
# because a 5.7e-14 difference in one feature column is enough to flip a GBDT split and
# stop `router_s.npy` reproducing bit-for-bit (measured -- a reimplementation drifted by
# 0.0118 in router score on 7,500 rows).
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from scipy.stats import rankdata

def oof_clf(X,y,seed=0,**kw):
    p=np.zeros(len(y))
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
        m=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=250,
            l2_regularization=5.0,min_samples_leaf=60,random_state=seed,**kw).fit(X[tr],y[tr])
        p[va]=m.predict_proba(X[va])[:,1]
    return p

def oof_reg(X,y,seed=0,**kw):
    p=np.zeros(len(y))
    for tr,va in KFold(5,shuffle=True,random_state=seed).split(X):
        m=HistGradientBoostingRegressor(max_depth=3,learning_rate=0.05,max_iter=250,
            l2_regularization=5.0,min_samples_leaf=60,random_state=seed,**kw).fit(X[tr],y[tr])
        p[va]=m.predict(X[va])
    return p

def bag(fn,X,y,seeds=(0,1,2,3,4)):
    return np.mean([rankdata(fn(X,y,seed=s)) for s in seeds],0)/len(y)

def quota_weight(gate_argmax, lam=1.0, target=2500):
    cnt=np.bincount(gate_argmax,minlength=3)
    surplus=(cnt-target)/float(target)
    return 1.0+lam*surplus[gate_argmax]
