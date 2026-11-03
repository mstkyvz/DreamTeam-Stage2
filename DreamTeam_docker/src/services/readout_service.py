# -*- coding: utf-8 -*-
u"""Turning a composed (N, 3) probability matrix into labels.

The read-out is not argmax, and the reason is measured. The private test is balanced 30k/30k/30k,
so the prior is known exactly -- which makes it tempting to pin the output to those counts. That
is worse. Swept on saved probabilities from a 15%-routed 90,000-row run:

    argmax        0.8840
    IPF lam=0.25  0.8874
    IPF lam=0.50  0.8879      <- shipped
    IPF lam=0.75  0.8860
    IPF lam=1.00  0.8814
    hard pin      0.8812

Fitting the true distribution exactly costs 0.0067 against fitting it halfway. IPF moves the
class marginals toward the target by scaling columns, so a row only flips when it was close;
a hard pin reassigns whatever it must to hit the quota, including rows the model was sure about.
"""
from __future__ import annotations

import numpy as np

from core import readout as RO


class IpfReadout:
    u"""Iterative proportional fitting against a soft class target."""

    def __init__(self, mode: str = 'ipf', lam: float = 0.5) -> None:
        if mode not in ('ipf', 'argmax', 'pin'):
            raise ValueError('unknown read-out mode %r' % (mode,))
        self._mode = mode
        self._lam = lam
        self._info: dict[str, object] = {}

    def decide(self, matrix: np.ndarray) -> np.ndarray:
        probabilities = np.exp(RO.norm_log(matrix))
        indices, info = RO.decide(probabilities, mode=self._mode, lam=self._lam)
        self._info = info
        return indices

    @property
    def info(self) -> dict[str, object]:
        return dict(self._info)
