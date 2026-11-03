# -*- coding: utf-8 -*-
u"""Composition root: the one place that knows both the services and the adapters.

Everything above this file depends on ports; everything below implements them. Keeping the
wiring here is what lets `services/` stay importable without vLLM, which is what makes the test
suite runnable on a laptop and the smoke test runnable in CI.
"""
from __future__ import annotations

import os
from typing import Callable

from app.config import Settings
from app.contracts import InferencePort
from services.cascade_service import CascadeService
from services.readout_service import IpfReadout
from services.routing_service import MarginRouter

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shot_block(settings: Settings) -> str:
    u"""The champion's global few-shot block, shipped as a rendered asset.

    The draw from train.parquet is deterministic (seed 4), so the rendered block ships with its
    SHA and the image needs no 500,000-row training file at inference. A silently different shot
    block is a silently different model, so a missing asset is fatal rather than defaulted.
    """
    path = settings.shots_path
    if not os.path.exists(path):
        raise FileNotFoundError('few-shot asset missing: %s' % path)
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def build_engine_factories(settings: Settings) -> tuple[Callable[[], InferencePort] | None,
                                                        Callable[[], InferencePort]]:
    u"""Return (gate_factory, champion_factory). Importing vLLM is deferred into the factories.

    DT_ENGINE=mock swaps in the CPU adapter so the full path -- CSV in, contract-verified CSV out,
    manifest written -- can be exercised with no GPU and no weights. The labels it produces are
    deterministic nonsense; the point is that everything AROUND the model is real. This is what
    the image's own smoke test runs.
    """
    if os.environ.get('DT_ENGINE') == 'mock':
        from adapters.mock_engine import MockInferenceAdapter
        champion = lambda: MockInferenceAdapter(name='champ', sharpness=3.0)
        if settings.cascade.routed_fraction >= 1.0:
            return None, champion
        return (lambda: MockInferenceAdapter(name='gate', sharpness=0.4)), champion

    from adapters.vllm_engine import VllmInferenceAdapter
    from core import prompts as PR
    from core.engine import PROFILES

    shots = _shot_block(settings)
    champion_profile = 'champ_int8_fi'
    gate_profile = 'gate_moe'

    champion_sys = PR.champion_sysmsg(shots)
    champion_spec = PROFILES[champion_profile]['prompt']

    def champion_factory() -> InferencePort:
        return VllmInferenceAdapter(
            settings.champion, settings.cascade,
            profile=champion_profile, prompt_spec=champion_spec,
            sysmsg=champion_sys, shots_path=settings.shots_path,
        )

    if settings.cascade.routed_fraction >= 1.0:
        return None, champion_factory                 # champion-only: one engine, one checkpoint

    # Asking for the cascade in an image built champion-only is a configuration error, and it must
    # surface here rather than as a vLLM download attempt on a network-disabled run.
    if not os.path.isdir(settings.gate.model_path):
        raise FileNotFoundError(
            'DT_FRAC=%.2f asks for the two-stage cascade, but the gate checkpoint is not at %s. '
            'The submitted image ships both checkpoints, so this means the image was rebuilt '
            'without the gate in its build context. Either restore it, or leave DT_FRAC at 1.0 '
            '(champion-only scores 0.9131 against the cascade\'s 0.8903 at 15%% routing anyway).'
            % (settings.cascade.routed_fraction, settings.gate.model_path))

    gate_spec = PROFILES[gate_profile]['prompt']
    gate_sys = PR.gate_sysmsg(gate_spec['kind'], shots)

    def gate_factory() -> InferencePort:
        return VllmInferenceAdapter(
            settings.gate, settings.cascade,
            profile=gate_profile, prompt_spec=gate_spec,
            sysmsg=gate_sys, shots_path=settings.shots_path,
        )

    return gate_factory, champion_factory


def build_cascade(settings: Settings) -> CascadeService:
    u"""The production object graph."""
    gate_factory, champion_factory = build_engine_factories(settings)
    return CascadeService(
        gate_factory=gate_factory,
        champion_factory=champion_factory,
        router=MarginRouter(),
        readout=IpfReadout(mode=settings.cascade.readout, lam=settings.cascade.lam),
        routed_fraction=settings.cascade.routed_fraction,
    )
