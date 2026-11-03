# -*- coding: utf-8 -*-
# VENDORED VERBATIM from agents/L16/src/common.py @ origin/agent/L16.
# The ONLY edits are the removal of the module-level absolute paths and of the two data
# loaders that used them (`common.load`, `feats2.load2`); every arithmetic line is L16's,
# because a 5.7e-14 difference in one feature column is enough to flip a GBDT split and
# stop `router_s.npy` reproducing bit-for-bit (measured -- a reimplementation drifted by
# 0.0118 in router score on 7,500 rows).
import numpy as np
LABELS=['irrelevant','relevant','semi-relevant']
IRR,REL,SEM=0,1,2

def norm_log(L):
    L=np.asarray(L,dtype=np.float64)
    m=L.max(1,keepdims=True)
    return L-(m+np.log(np.exp(L-m).sum(1,keepdims=True)))


def probs(L): return np.exp(L)
def margin(L):
    P=np.sort(probs(L),1); return P[:,-1]-P[:,-2]
def margin_nats(L):
    S=np.sort(L,1); return S[:,-1]-S[:,-2]
