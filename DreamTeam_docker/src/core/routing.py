# -*- coding: utf-8 -*-
u"""The router: cheap features over the gate, and the selection of which rows go to the champion.

`build_features` is L16's `feats2.build2` with the module-level absolute paths taken out and
nothing else changed, so a feature matrix built here is the one the measured router was fitted on
(verified: rebuilding L16's router from `gate_moe26` + `enc_tyonly_cal` + `enc_s1_ty` reproduces
`agents/L37/out/router_s.npy` with max|delta| = 0 and L16's `route_L16e_moe30.csv` at Jaccard
1.0000).

⚠️ TWO THINGS ABOUT COST AND DEPLOYABILITY THAT THE RECORDED TABLES DO NOT SAY.

1. **The feature set shipped as `e1` (K = 5.5 s) is not e1.**  `cols_for(X, ['e1'])` drops the
   `e2_*` columns but keeps the 16 `eb_*` / `e12_agree` / `gate_minus_eb` columns, and `eb` is the
   BLEND of both encoders.  So the scored router reads `s1_ty` too and its true feature cost is
   **5.5 + 15.7 = 21.2 s**, not 5.5 s.  Measured here on the committed matrices:

       feature set                cols   K      OOF AUC   disagreement recall @20% / @30%
       as shipped ("e1")            63   21.2s  0.8892    0.6839 / 0.8327
       genuinely e1 only            47    5.5s  0.8851    0.6751 / 0.8337
       free (gate + text/group)     31    0.0s  0.8680    0.6401 / 0.7967
       e1 + e2 declared             79   21.2s  0.8927    0.6897 / 0.8444
       gate margin, no features     --    0.0s  0.8184    --

   Dropping the second encoder costs 0.004 AUC and *gains* 0.001 recall at the 30% operating
   point, for 15.7 s.  `FEATURE_SETS['e1only']` is that configuration and it is the recommended
   optimisation -- but the DEFAULT is `shipped`, because `shipped` is the router whose output was
   actually scored (0.9133 at 30%) and swapping a scored configuration for an unscored one at
   delivery time is how a run loses a measurement it already paid for.

2. **The router is fitted on `y = (gate argmax != champion argmax)`, which needs the champion.**
   That is fine as an offline measurement and impossible as an online one: at inference the
   champion has not run yet -- that is the entire point of the cascade.  So a deployed router is
   either a PRE-FITTED model (`tools/fit_router.py`) or a PRE-COMPUTED score vector, and the
   in-process OOF fit is available only when champion logits for every row are already on disk,
   where it exists to reproduce the anchor, not to be used.  `resolve_router` refuses to guess.
"""
import os, json, hashlib
import numpy as np
import pandas as pd

from .readout import norm_log

FEATURE_COST_S = {'e1': 5.5, 'e2': 15.7, 'stu': 4.2, 'x12': 116.2}
# name -> (keep e1_* cols, keep e2_* cols, keep eb_*/e12 block, encoders that must be COMPUTED,
#          measured feature cost in seconds for 7,500 rows)
FEATURE_SETS = {
    # exactly L16's `cols_for(X, ['e1'])`: e2_* dropped but eb_* kept, and eb_* is the blend of
    # BOTH encoders -- so this set costs 21.2 s, not the 5.5 s it is filed under.  63 columns.
    'shipped': dict(e1=True,  e2=False, eb=True,  need=('e1', 'e2'), K=21.2, cols=63),
    # the same money, spent honestly: both encoders' own columns as well.  79 columns.
    'e1e2':    dict(e1=True,  e2=True,  eb=True,  need=('e1', 'e2'), K=21.2, cols=79),
    # one encoder pass, nothing that touches the other.  47 columns, 5.5 s.
    'e1only':  dict(e1=True,  e2=False, eb=False, need=('e1',),      K=5.5,  cols=47),
    # no encoder pass at all.  31 columns, 0 s.
    'free':    dict(e1=False, e2=False, eb=False, need=(),           K=0.0,  cols=31),
    # not a learned router at all -- the gate's own margin.  0 s, AUC 0.8184.
    'margin':  dict(e1=False, e2=False, eb=False, need=(),           K=0.0,  cols=0),
}


def build_features(te, gate_log, enc=None):
    u"""te: the row frame; gate_log: (N,3) log-probs; enc: {'e1': (N,3), 'e2': (N,3)} log-probs.

    Delegates to the VENDORED `lib.l16.feats2.build2` so the columns are bit-identical to the
    ones the scored router was fitted on.  `build2` requires both encoders (it always builds the
    `eb_*` blend), so when a feature set does not need `e2` a neutral stand-in is passed and the
    columns that read it are dropped by `select_columns` immediately afterwards.  The stand-in is
    never allowed to reach a kept column: `select_columns` asserts the column count.
    """
    from .l16.feats2 import build2
    from .l16.common import norm_log as _nl
    enc = dict(enc or {})
    n = len(gate_log)
    if 'e1' not in enc:
        enc['e1'] = np.log(np.full((n, 3), 1.0 / 3.0))
    if 'e2' not in enc:
        enc['e2'] = enc['e1']
    # NOTE: the encoder matrices are passed through UNCHANGED.  Re-normalising them here is
    # mathematically a no-op and numerically is not: L16 fed `log(P / P.sum(1))` straight in, and
    # putting that through `norm_log` again perturbs it by ~5.7e-14, which flips enough GBDT
    # splits to move the router score by 0.0118.  Use `enc_from_probs()` to build them.
    d = {'gate_moe26': _nl(gate_log), 'te': te,
         'e_tyonly_cal': enc['e1'], 'e_s1_ty': enc['e2']}
    return build2(d, 'moe26')


def enc_from_probs(P):
    u"""L16's `common.load` conversion, reproduced exactly: row-normalise in PROBABILITY space,
    clip, log.  Anything else changes the router score in the 3rd decimal."""
    P = np.asarray(P, dtype=np.float64)
    P = P / P.sum(1, keepdims=True)
    return np.log(np.clip(P, 1e-12, None))


def select_columns(X, feature_set):
    u"""Reproduces L16's `cols_for` for the shipped set, and adds the genuinely-e1 / free sets
    the audit above says are the honest cost points.  Column ORDER is preserved, because a
    HistGradientBoosting model is positional and the bit-identical reproduction of
    `router_s.npy` depends on it."""
    spec = FEATURE_SETS[feature_set]
    drop = []
    if not spec['e1']:
        drop += [c for c in X.columns if c.startswith('e1_')]
    if not spec['e2']:
        drop += [c for c in X.columns if c.startswith('e2_')]
    if not spec['eb']:
        drop += [c for c in X.columns if c.startswith('eb_')] + ['e12_agree', 'gate_minus_eb']
    drop = set(drop)
    keep = [c for c in X.columns if c not in drop]
    out = X[keep]
    if spec['cols'] and out.shape[1] != spec['cols']:
        raise AssertionError('feature set %r produced %d columns, expected %d'
                             % (feature_set, out.shape[1], spec['cols']))
    return out


def fit_oof_scores(X, y, seeds=(0, 1, 2, 3, 4)):
    u"""L16's bagged OOF score, `rankdata(bag(...)) / len(y)`.

    The OUTER `rankdata` is not decoration: without it the score is the same ORDER (Spearman
    1.000000, identical routed sets at every fraction) but different VALUES, and the values are
    what `router_s.npy` is compared against.  With it, this reproduces
    `agents/L37/out/router_s.npy` bit-for-bit.

    NOT deployable -- `y` needs the champion's answer, which at inference has not happened yet.
    This is the anchor path; `fit_full_models` is the deployable twin.
    """
    from scipy.stats import rankdata
    from .l16.routers import oof_clf, bag
    return rankdata(bag(oof_clf, X, y, seeds=seeds)) / len(y)


def fit_full_models(X, y, seeds=(0, 1, 2, 3, 4)):
    u"""The deployable twin of `fit_oof_scores`: five full-data models, to be applied to rows the
    champion has never seen.  Returned as a list so `apply_models` can average their RANKS, which
    is what the OOF version does."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return [HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                           l2_regularization=5.0, min_samples_leaf=60,
                                           random_state=s).fit(X, y) for s in seeds]


def apply_models(models, X):
    from scipy.stats import rankdata
    return np.mean([rankdata(m.predict_proba(X)[:, 1]) for m in models], 0) / len(X)


def margin_score(gate_log):
    u"""Higher = route it.  The 0-second fallback: AUC 0.8184 against the learned 0.8892."""
    L = norm_log(gate_log)
    S = np.sort(L, 1)
    return -np.clip(S[:, -1] - S[:, -2], 0, 60)


def select_rows(score, frac, n):
    u"""Top-`frac` by score.  `int(round())` is L16's rule and is kept so the routed count matches
    the recorded ones exactly (0.30 * 7500 -> 2250)."""
    k = int(round(float(frac) * n))
    k = max(0, min(n, k))
    sel = np.zeros(n, dtype=bool)
    if k:
        sel[np.argsort(-np.asarray(score), kind='stable')[:k]] = True
    return sel, k


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()
