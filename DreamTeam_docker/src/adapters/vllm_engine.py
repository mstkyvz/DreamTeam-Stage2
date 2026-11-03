# -*- coding: utf-8 -*-
u"""InferencePort over vLLM, wrapping the measured read-out path without reimplementing it.

WHY THIS IS AN ADAPTER AND NOT A REWRITE. The forced-token read-out in `core/` is the numerical
core: the assistant turn is prefilled with `{"etiket": "`, EXACTLY ONE token is decoded with
logprobs, and the 3-vector is the marginal over vocabulary tokens that uniquely prefix exactly
one label. Every score this project reports was measured through that code. Reimplementing it
behind a nicer interface would put the reported numbers and the shipped numbers on two different
code paths, which is the one thing a submission must never do. So the core is vendored unchanged
and this adapter gives it a typed boundary.

WHAT THE ADAPTER ADDS: engine lifecycle as a context manager (the gate at 24.75 GiB and the
champion at 30.63 GiB do not co-reside in 40 GB, so "freed before the next boots" is a
correctness property, not tidiness), boot timeouts, and translation of the core's positional
tuples into `StageResult`.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from app.config import CascadeSettings, EngineSettings
from app.contracts import StageResult
from app.errors import EngineBootTimeout, EngineError


class VllmInferenceAdapter:
    u"""One vLLM engine, booted on demand and guaranteed to release the GPU on exit."""

    def __init__(
        self,
        settings: EngineSettings,
        cascade: CascadeSettings,
        *,
        profile: str,
        prompt_spec: dict,
        sysmsg: str,
        shots_path: str,
    ) -> None:
        self._settings = settings
        self._cascade = cascade
        self._profile = profile
        self._prompt_spec = prompt_spec
        self._sysmsg = sysmsg
        self._shots_path = shots_path
        self._engine = None
        self._timings: dict[str, float] = {}
        self._info: dict[str, object] = {}

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> 'VllmInferenceAdapter':
        self.boot()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def boot(self) -> None:
        from core.engine import VllmJudge
        from core.reference_pipeline import BootTimeout, _with_timeout

        overrides = {
            'max_model_len': self._settings.max_model_len,
            'gpu_memory_utilization': self._settings.gpu_memory_utilization,
            'enforce_eager': self._settings.enforce_eager,
            'max_num_batched_tokens': self._settings.max_num_batched_tokens,
            'max_num_seqs': self._settings.max_num_seqs,
            'attention_backend': self._settings.attention_backend,
        }
        engine = VllmJudge(
            self._profile,
            model=self._settings.model_path,
            overrides=overrides,
            logprobs=self._cascade.logprobs,
        )
        started = time.time()
        try:
            _with_timeout(self._settings.boot_timeout_s, engine.boot,
                          'boot %s' % self._profile)
        except BootTimeout as exc:
            raise EngineBootTimeout('%s did not boot within %d s'
                                    % (self._profile, self._settings.boot_timeout_s)) from exc
        except Exception as exc:
            raise EngineError('%s failed to boot: %s' % (self._profile, exc)) from exc

        self._engine = engine
        self._timings['boot_s'] = round(time.time() - started, 3)
        self._info = {
            'profile': self._profile,
            'model_path': self._settings.model_path,
            'boot_s': self._timings['boot_s'],
            # Read from the live engine, not from the log line: the vLLM log's "GPU KV cache size"
            # counts one cache group and reads ~5.5x low on this model's heterogeneous layers.
            'attention_impl': getattr(engine, 'attn_impl', None),
        }

    def close(self) -> None:
        if self._engine is None:
            return
        started = time.time()
        try:
            self._engine.shutdown()
        finally:
            self._engine = None
            self._timings['shutdown_s'] = round(time.time() - started, 3)

    # ------------------------------------------------------------------ scoring
    def score(self, rows: Sequence[dict], *, stage: str) -> StageResult:
        if self._engine is None:
            raise EngineError('score() called before boot() on %s' % self._profile)
        from core.reference_pipeline import judge

        args = self._reference_args()
        timings: dict[str, float] = {}
        logprobs, coverage, stats = judge(
            self._engine, list(rows), self._sysmsg, self._prompt_spec,
            args, stage, timings, len(rows),
        )
        merged = dict(self._timings)
        merged.update(timings)
        info = dict(self._info)
        info.update(stats)
        info['label_mass_mean'] = stats.get('label_mass_mean')
        return StageResult(
            logprobs=np.asarray(logprobs, dtype=np.float64),
            rows=len(rows),
            prompt_tokens=int(stats.get('prompt_tokens_mean', 0) * len(rows)),
            timings_s=merged,
            engine_info=info,
        )

    def _reference_args(self) -> SimpleNamespace:
        u"""The namespace the vendored `judge()` reads. Built here so the core stays untouched."""
        return SimpleNamespace(
            prefix_align=self._cascade.prefix_align,
            sort_by_query=self._cascade.sort_by_query,
            max_attr=self._cascade.max_attr_chars,
            max_model_len=self._settings.max_model_len,
            logprobs=self._cascade.logprobs,
            shots=self._shots_path,
            allow_attention_fallback=False,
        )

    @property
    def info(self) -> dict[str, object]:
        return dict(self._info)
