# -*- coding: utf-8 -*-
u"""InferencePort with no GPU, so the orchestration can be tested for real.

This is not a stub that returns zeros. It produces a deterministic, input-dependent 3-vector, so
the tests exercise the properties that actually break in production: that composition puts the
champion's rows where the router asked, that the read-out sees a well-formed matrix, that row
order survives, and that engines are closed even when a stage raises.

`fail_on_stage` exists because "the gate raised and the champion then OOM'd on a card that looked
free" is a real failure mode, and the only way to test the `finally` that prevents it.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from app.contracts import LABELS, StageResult
from app.errors import EngineError


class MockInferenceAdapter:
    u"""Deterministic CPU stand-in for a vLLM engine."""

    def __init__(self, *, name: str = 'mock', sharpness: float = 1.0,
                 fail_on_stage: str | None = None) -> None:
        self._name = name
        self._sharpness = sharpness
        self._fail_on_stage = fail_on_stage
        self.closed = False
        self.scored_rows = 0

    def __enter__(self) -> 'MockInferenceAdapter':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def score(self, rows: Sequence[dict], *, stage: str) -> StageResult:
        if self._fail_on_stage == stage:
            raise EngineError('mock failure injected at stage %r' % stage)
        self.scored_rows += len(rows)
        matrix = np.vstack([self._row_logprobs(row) for row in rows])
        return StageResult(
            logprobs=matrix,
            rows=len(rows),
            prompt_tokens=0,
            timings_s={'generate_s': 0.0},
            engine_info={'profile': self._name, 'attention_impl': 'mock'},
        )

    def _row_logprobs(self, row: dict) -> np.ndarray:
        u"""Hash the row into a stable pseudo-posterior. Same row, same engine -> same vector."""
        seed = hashlib.sha256(
            ('%s|%s|%s' % (self._name, row.get('row_id', ''), row.get('query', ''))).encode()
        ).digest()
        raw = np.frombuffer(seed[:24], dtype=np.uint8)[:len(LABELS) * 8]
        values = raw.reshape(len(LABELS), 8).mean(axis=1) / 255.0
        logits = values * self._sharpness * 6.0
        return logits - np.log(np.exp(logits).sum())

    def close(self) -> None:
        self.closed = True

    @property
    def info(self) -> dict[str, object]:
        return {'profile': self._name, 'attention_impl': 'mock'}
