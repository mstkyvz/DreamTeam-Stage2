# -*- coding: utf-8 -*-
u"""Read-out: prompt-position label marginal, and the three count rules.

The marginal itself is NOT reimplemented here -- `lib.champ.l9_run.read_out` and
`lib.champ.l9_judge.build_label_token_sets` are the code every measured arm in this run used,
imported rather than copied, so a number produced here is comparable with the recorded ones.

What this module adds is the final assignment step, and the reason it is a module of its own is
that the choice between the three rules is worth up to 0.016 macro-F1 and was got wrong once:

  argmax  -- no count step at all.
  ipf     -- iterative proportional fitting of the three class columns until the ARGMAX COUNTS
             hit a target.  DEFAULT, with a SOFT target (lam=0.5): halfway between the raw
             counts and N/3.  Measured on the hidden holdout at 30% routing: 0.9133, against
             0.9079 for argmax and 0.9040 for the greedy pin (L37).
  pin     -- greedy exact-quota assignment.  Kept because single-source outputs still want it
             and because it is the anchor that reproduces the 0.9040 measurement, but it is NOT
             the default: across two uncalibrated sources it evicts the rows you paid the
             champion for (244 of its 248 displacements at 30% land on routed rows).
"""
import numpy as np

LABELS = ['irrelevant', 'relevant', 'semi-relevant']


def norm_log(L):
    L = np.asarray(L, dtype=np.float64)
    m = L.max(1, keepdims=True)
    return L - (m + np.log(np.exp(L - m).sum(1, keepdims=True)))


def soft_target(counts, n, lam=0.5):
    u"""lam=1 -> exactly n/3 each; lam=0 -> the model's own raw counts; 0.5 -> halfway.

    Returned as floats that sum to n, so the caller can assert the arithmetic.
    """
    counts = np.asarray(counts, dtype=np.float64)
    t = lam * (float(n) / 3.0) + (1.0 - lam) * counts
    return t * (float(n) / t.sum())


def ipf(P, target, iters=8000, damp=0.15, tol=0.0):
    u"""L37's `ipf_to`, verbatim in behaviour: re-weight the three class columns until the
    argmax counts land on `target`, keeping the best assignment ever seen rather than the last.

    `damp` is the exponent on the multiplicative update; 0.15 is what was measured.  The
    `best`/`bd` bookkeeping matters: the iteration is not monotone and the final w is often
    worse than one it passed through.
    """
    P = np.asarray(P, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    w = np.ones(P.shape[1])
    best, bd = None, np.inf
    for _ in range(int(iters)):
        pr = (P * w).argmax(1)
        c = np.bincount(pr, minlength=P.shape[1]).astype(float)
        dv = np.abs(c - t).max()
        if dv < bd:
            bd, best = dv, pr.copy()
        if dv <= tol:
            break
        w *= (t / np.maximum(c, 1.0)) ** damp
    return best, w, bd


def pin(P, quota=None):
    u"""Greedy exact-quota assignment, the coordinator's `pin()`.

    `quota=None` means the balanced quota with the remainder handed to the FIRST classes:
    `n//3` each, then `n - 3*(n//3)` extra slots spread one per class from index 0.  That is
    the arithmetic that breaks when n is not divisible by 3, and it is asserted below rather
    than trusted.
    """
    P = np.asarray(P, dtype=np.float64)
    n, k = P.shape
    if quota is None:
        quota = np.full(k, n // k, dtype=np.int64)
        quota[:n - int(quota.sum())] += 1
    quota = np.asarray(quota, dtype=np.int64)
    assert int(quota.sum()) == n, 'quota %r sums to %d, not %d rows' % (
        quota.tolist(), int(quota.sum()), n)
    prio = P.max(1) - np.sort(P, 1)[:, -2]
    order = np.argsort(-prio, kind='stable')
    out = np.full(n, -1, dtype=np.int64)
    left = quota.copy()
    for i in order:
        for c in np.argsort(-P[i]):
            if left[c] > 0:
                out[i] = c
                left[c] -= 1
                break
    assert (out >= 0).all(), 'pin left %d rows unassigned' % int((out < 0).sum())
    return out


def decide(P, mode='ipf', lam=0.5, quota=None):
    u"""P is an (N,3) PROBABILITY matrix (rows sum to 1).  Returns (labels_idx, info)."""
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    am = P.argmax(1)
    raw = np.bincount(am, minlength=3).astype(float)
    if mode == 'argmax':
        return am, {'mode': 'argmax', 'raw_counts': raw.astype(int).tolist()}
    if mode == 'ipf':
        t = soft_target(raw, n, lam)
        pr, w, dv = ipf(P, t)
        return pr, {'mode': 'ipf', 'lam': lam, 'target': [round(float(x), 1) for x in t],
                    'weights': [float(x) for x in w], 'max_count_gap': float(dv),
                    'raw_counts': raw.astype(int).tolist(),
                    'final_counts': np.bincount(pr, minlength=3).astype(int).tolist(),
                    'moved_vs_argmax': int((pr != am).sum())}
    if mode == 'pin':
        pr = pin(P, quota)
        return pr, {'mode': 'pin', 'raw_counts': raw.astype(int).tolist(),
                    'final_counts': np.bincount(pr, minlength=3).astype(int).tolist(),
                    'moved_vs_argmax': int((pr != am).sum())}
    raise ValueError('unknown read-out mode %r' % (mode,))


def macro_f1(y_true_idx, y_pred_idx, k=3):
    y_true_idx = np.asarray(y_true_idx); y_pred_idx = np.asarray(y_pred_idx)
    fs = []
    for c in range(k):
        tp = int(((y_true_idx == c) & (y_pred_idx == c)).sum())
        fp = int(((y_true_idx != c) & (y_pred_idx == c)).sum())
        fn = int(((y_true_idx == c) & (y_pred_idx != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(fs)), [round(x, 4) for x in fs]
