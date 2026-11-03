# -*- coding: utf-8 -*-
u"""Deciding which rows are worth the champion's time.

The cascade only pays off if the rows it escalates are the ones the gate would have got wrong.
That is a supervised question -- "does the gate disagree with the champion" -- whose label does
not exist at inference time, which is exactly what the cascade exists not to know yet.

Two routers, and the difference between them is measured, not assumed:

  learned   AUC 0.8892. Features read two encoder checkpoints (`tyonly_cal`, `s1_ty`) that are
            NOT in this image. Only their public_test outputs survive, so the router cannot be
            evaluated on private rows at all.
  margin    AUC 0.8184, zero extra cost: the gap between the gate's top two log-probabilities.
            Worth about -0.021 macro-F1 at 30% routing relative to the learned router.

This service ships the margin router and says so loudly. It does not silently produce a
router-shaped number from features it does not have -- a fabricated score would look like a
working router in the manifest and route rows at random.
"""
from __future__ import annotations

import numpy as np

from app.contracts import RoutingDecision, StageResult
from core import readout as RO


class MarginRouter:
    u"""Escalate the rows where the gate is least certain.

    Score = -(top1 - top2) over normalised log-probabilities, so higher means less certain means
    route it. Clipped at 60 nats: a saturated gap carries no more information than a large one,
    and leaving it unclipped lets a handful of extreme rows dominate the sort order.
    """

    source = 'gate_margin_fallback'

    def __init__(self, note: str = '') -> None:
        self._note = note or (
            'learned router unavailable in this image (encoder checkpoints absent); '
            'margin AUC 0.8184 vs learned 0.8892'
        )

    def route(self, gate: StageResult, frame: object, fraction: float) -> RoutingDecision:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError('routed fraction must be in [0, 1], got %r' % (fraction,))
        scores = RO_margin(gate.logprobs)
        n = len(scores)
        k = int(round(fraction * n))
        # Ties at the cut boundary are broken by position, which is deterministic given the input
        # order. argsort's default is stable, so two runs on the same input escalate the same set.
        order = np.argsort(-scores, kind='stable')
        escalate = np.sort(order[:k])
        return RoutingDecision(
            escalate_index=escalate,
            scores=scores,
            source=self.source,
            fraction=float(k) / float(n) if n else 0.0,
            note=self._note,
        )


def RO_margin(gate_logprobs: np.ndarray) -> np.ndarray:
    u"""The margin score itself, kept as a free function so tests can exercise it without a router."""
    normalised = RO.norm_log(gate_logprobs)
    ordered = np.sort(normalised, 1)
    return -np.clip(ordered[:, -1] - ordered[:, -2], 0, 60)
