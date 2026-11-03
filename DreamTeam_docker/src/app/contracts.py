# -*- coding: utf-8 -*-
u"""The interfaces every service is written against, and the data that crosses them.

Ports are `typing.Protocol`, so an adapter satisfies one by shape alone -- no base class, no
import from the adapter into the domain. This is what lets the cascade orchestrator be tested on
CPU against `MockInferenceAdapter` with no vLLM installed, and it is why `services/` never
imports `adapters/`.

Every matrix that crosses a port is (N, 3) float, column order fixed by `LABELS`. That order is
load-bearing: it is the order the read-out, the router and the saved artifacts all assume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

# Fixed by the competition. Column order of every (N, 3) matrix in this codebase.
LABELS: tuple[str, str, str] = ('irrelevant', 'relevant', 'semi-relevant')

# The organizer's input schema.
REQUIRED_COLUMNS: tuple[str, ...] = (
    'row_id', 'query', 'title', 'category', 'brand', 'gender', 'age_group', 'attributes',
)


@dataclass(frozen=True)
class StageResult:
    u"""What one inference stage produced, plus what it cost.

    `logprobs` is the raw forced-token read-out: for each row, the marginal over vocabulary
    tokens that uniquely prefix each label. Not normalised -- the read-out service owns that.
    """

    logprobs: np.ndarray
    rows: int
    prompt_tokens: int
    timings_s: dict[str, float] = field(default_factory=dict)
    engine_info: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.logprobs.ndim != 2 or self.logprobs.shape[1] != len(LABELS):
            raise ValueError('logprobs must be (N, %d), got %r'
                             % (len(LABELS), self.logprobs.shape))


@dataclass(frozen=True)
class RoutingDecision:
    u"""Which rows escalate, and why that set and not another."""

    escalate_index: np.ndarray          # positional indices into the input frame
    scores: np.ndarray                  # per-row router score, higher = harder
    source: str                         # 'learned_model' | 'gate_margin_fallback' | 'precomputed'
    fraction: float
    note: str = ''


@dataclass(frozen=True)
class Attribution:
    u"""One field's contribution to the predictor's own decision.

    Produced by occluding the field and re-running the SAME forced-token read-out, so the number
    is a property of the predictor, not of a separate explainer.
    """

    field_name: str
    delta_logprob: tuple[float, float, float]
    occluded_label: str
    flipped: bool


@dataclass(frozen=True)
class Explanation:
    u"""Everything the UI shows about one row, all of it read out of the predictor."""

    row_id: str
    query: str
    label: str
    posterior: tuple[float, float, float]
    margin: float
    attributions: tuple[Attribution, ...]
    prior_shift: tuple[float, float, float]
    stage: str
    engine_info: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class InferencePort(Protocol):
    u"""A model that scores rows into a (N, 3) forced-token read-out.

    Implementations own their engine's whole lifecycle. `close()` must actually free the GPU:
    the gate (24.75 GiB) and the champion (30.63 GiB) do not co-reside in 40 GB, so the cascade
    depends on the first engine being gone before the second boots.
    """

    def score(self, rows: Sequence[dict], *, stage: str) -> StageResult: ...

    def close(self) -> None: ...

    @property
    def info(self) -> dict[str, object]: ...


@runtime_checkable
class RouterPort(Protocol):
    u"""Chooses which rows are worth the champion's time."""

    def route(self, gate: StageResult, frame: object, fraction: float) -> RoutingDecision: ...


@runtime_checkable
class ReadoutPort(Protocol):
    u"""Turns a composed (N, 3) matrix into final labels."""

    def decide(self, matrix: np.ndarray) -> np.ndarray: ...

    @property
    def info(self) -> dict[str, object]: ...
