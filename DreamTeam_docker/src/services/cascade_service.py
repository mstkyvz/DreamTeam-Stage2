# -*- coding: utf-8 -*-
u"""The orchestrator: gate over every row, champion over the hardest few, one composed answer.

    rows -> [gate engine] -> logprobs(N,3) -> [router] -> escalate_index
         -> [champion engine] -> logprobs(K,3)
         -> compose (champion where escalated, gate elsewhere)
         -> [read-out] -> labels -> submission

The sequencing is a hardware constraint, not a style choice. The gate is 24.75 GiB of weights and
the champion 30.63 GiB; 55.38 GiB does not fit on a 40 GiB card (co-residence was tried and vLLM
reports `Available KV cache memory: -21.61 GiB`). So the gate must be fully torn down before the
champion boots, which is why engines are context managers and why `close()` is in the port.

Composition is on NORMALISED log-probabilities from both stages, never on raw argmax. Mixing an
argmax from one model with a probability vector from another would make the read-out's class
marginals meaningless, and the read-out is worth +0.004 macro-F1 over argmax.

This service never imports an adapter. It is handed `InferencePort` factories, which is what lets
the whole cascade run on CPU in the test suite with no vLLM present.
"""
from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from app.contracts import InferencePort, LABELS, ReadoutPort, RouterPort, StageResult
from app.errors import OutputContractError

EngineFactory = Callable[[], InferencePort]


class CascadeService:
    u"""Composes a cheap gate and an expensive champion into one set of labels."""

    def __init__(
        self,
        gate_factory: EngineFactory | None,
        champion_factory: EngineFactory,
        router: RouterPort,
        readout: ReadoutPort,
        *,
        routed_fraction: float,
    ) -> None:
        self._gate_factory = gate_factory
        self._champion_factory = champion_factory
        self._router = router
        self._readout = readout
        self._fraction = 1.0 if gate_factory is None else routed_fraction
        self.manifest: dict[str, object] = {}

    def run(self, frame: pd.DataFrame) -> list[str]:
        u"""Return one label per input row, in input order."""
        started = time.time()
        rows: Sequence[dict] = frame.to_dict('records')
        n = len(rows)
        timings: dict[str, float] = {}
        stages: dict[str, object] = {}

        gate_result = None
        if self._gate_factory is not None:
            gate_result = self._run_stage(self._gate_factory, rows, 'gate', timings, stages)

        if gate_result is None:
            escalate = np.arange(n)
            routing_info = {'source': 'no_gate', 'fraction': 1.0,
                            'note': 'champion-only configuration: every row goes to the champion'}
        else:
            decision = self._router.route(gate_result, frame, self._fraction)
            escalate = decision.escalate_index
            routing_info = {'source': decision.source, 'fraction': decision.fraction,
                            'note': decision.note, 'escalated_rows': int(len(escalate))}
            if decision.source.endswith('fallback'):
                print('[cascade] WARNING: %s -- %s' % (decision.source, decision.note), flush=True)

        champion_result = None
        if len(escalate):
            routed_rows = [rows[i] for i in escalate]
            champion_result = self._run_stage(
                self._champion_factory, routed_rows, 'champion', timings, stages)

        composed, sources = self._compose(n, gate_result, champion_result, escalate)
        indices = self._readout.decide(composed)
        labels = [LABELS[i] for i in indices]

        if len(labels) != n:
            raise OutputContractError('produced %d labels for %d rows' % (len(labels), n))

        self.manifest = {
            'rows': n,
            'routing': routing_info,
            'stages': stages,
            'timings_s': timings,
            'readout': self._readout.info,
            'composition': {
                'from_champion': int((sources == 1).sum()),
                'from_gate': int((sources == 0).sum()),
            },
            'wall_clock_s': round(time.time() - started, 3),
        }
        return labels

    # ------------------------------------------------------------------ internals
    def _run_stage(
        self,
        factory: EngineFactory,
        rows: Sequence[dict],
        stage: str,
        timings: dict[str, float],
        stages: dict[str, object],
    ) -> StageResult:
        u"""Boot one engine, score, and free the GPU before returning.

        The `finally` in the context manager is what makes the next boot possible; a stage that
        raises must still release its weights or the champion will OOM on a card that looks free.
        """
        started = time.time()
        engine = factory()
        with engine:                                  # closes on every exit path, including raise
            result = engine.score(rows, stage=stage)
        for key, value in result.timings_s.items():
            timings['%s_%s' % (stage, key)] = value
        timings['%s_total_s' % stage] = round(time.time() - started, 3)
        stages[stage] = dict(result.engine_info)
        print('[cascade] %s: %d rows in %.1f s' % (stage, result.rows,
                                                   timings['%s_total_s' % stage]), flush=True)
        return result

    @staticmethod
    def _compose(
        n: int,
        gate: StageResult | None,
        champion: StageResult | None,
        escalate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        u"""Champion log-probabilities where it ran, gate log-probabilities everywhere else."""
        if gate is None and champion is None:
            raise OutputContractError('neither stage produced a result')
        width = len(LABELS)
        composed = (np.array(gate.logprobs, dtype=np.float64, copy=True) if gate is not None
                    else np.zeros((n, width), dtype=np.float64))
        sources = np.zeros(n, dtype=np.int8)
        if champion is not None and len(escalate):
            composed[escalate] = champion.logprobs
            sources[escalate] = 1
        return composed, sources
