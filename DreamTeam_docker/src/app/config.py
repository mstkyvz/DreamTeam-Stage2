# -*- coding: utf-8 -*-
u"""The shipped configuration, in one place, with its provenance attached.

Every knob that can move the score or the wall clock lives here and nowhere else. Each default
carries the measurement it came from, because a value with no measurement behind it is a guess
that will be re-guessed by the next person to read it.

Environment overrides exist for the team's own A/B runs. Official judging sets none of them, so
the defaults below are exactly what the reported numbers were measured with.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class EngineSettings:
    u"""One vLLM engine's boot parameters.

    `gpu_memory_utilization` and `max_model_len` are not free choices. At 0.97 FlashInfer's
    workspace leaves its sampler no room and the engine OOMs during warm-up; at <=0.92 with
    max_model_len 6144 the KV reservation no longer fits. The longest prompt observed in
    public_test is 5,178 tokens, so 5,250 is the smallest safe max_model_len.
    """

    model_path: str
    max_model_len: int = 5250
    gpu_memory_utilization: float = 0.94
    enforce_eager: bool = True          # read-out is max_tokens=1 pure prefill; eager is 0.4% faster
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 256
    attention_backend: str | None = None
    expect_impl: tuple[str, ...] = ()
    weights_gib: float = 0.0
    boot_timeout_s: int = 2400


@dataclass(frozen=True)
class CascadeSettings:
    u"""The routing and read-out policy.

    routed_fraction -- 1.0 means champion-only: no gate, one engine, one checkpoint.
      Measured on the same 90,000 rows, one A100-SXM4-40GB, margin routing:

        fraction   macro-F1   wall clock   checkpoints
        0.05       0.8741     24.8 min     2  (59 GB)
        0.15       0.8903     29.4 min     2  (59 GB)
        1.00       0.9131     46.9 min     1  (32 GB)   <- shipped

      The cascade is a WALL-CLOCK optimisation, not an accuracy one, and at these fractions it
      costs 0.023-0.039 macro-F1. It only draws level with champion-only around 30% routing
      (0.9137 vs 0.9132 on public labels). Since the organizer allows 150 minutes and the
      champion-only run finishes in 46.9, there is no time pressure to buy accuracy against --
      so the gate is not shipped, and the image carries one checkpoint instead of two.

      Set DT_FRAC below 1.0 to re-enable the cascade. That requires the gate checkpoint to be
      present in the image; `wiring.build_engine_factories` refuses rather than guessing.

    readout / lam -- IPF against a soft class target. Swept on saved probabilities at 15%:
        argmax 0.8840 | lam 0.25 0.8874 | lam 0.50 0.8879 | lam 0.75 0.8860 | lam 1.00 0.8814
      lam=0.5 is the maximum. Pinning to the true 30k/30k/30k prior scores 0.8812 -- fitting the
      real distribution exactly is WORSE than fitting it halfway. Do not spend time here.

    router -- 'margin' is the only deployable option in this image. The learned router scores
      AUC 0.8892 against the margin's 0.8184, but its features read two encoder checkpoints that
      are not in this image and cannot be recomputed from the private rows.
    """

    routed_fraction: float = 1.0        # 1.0 = champion-only; see the class docstring
    readout: str = 'ipf'
    lam: float = 0.5
    router: str = 'margin'
    prefix_align: str = 'off'
    sort_by_query: bool = True          # token-identical; -5.8 s and eviction-proof
    logprobs: int = 48
    max_attr_chars: int = 220


@dataclass(frozen=True)
class Settings:
    gate: EngineSettings
    champion: EngineSettings
    cascade: CascadeSettings
    artifact_dir: str
    shots_path: str

    @property
    def summary(self) -> dict[str, object]:
        u"""What the manifest and the logs print, so a run is identifiable from its output alone."""
        return {
            'gate_model': self.gate.model_path,
            'champion_model': self.champion.model_path,
            'routed_fraction': self.cascade.routed_fraction,
            'readout': '%s(lam=%s)' % (self.cascade.readout, self.cascade.lam),
            'router': self.cascade.router,
            'prefix_align': self.cascade.prefix_align,
        }


# Baked model locations. COPY'd into the image at build time; never downloaded at runtime.
GATE_MODEL_DEFAULT = '/models/gate_moe26'
CHAMPION_MODEL_DEFAULT = '/models/champ_g31int8'


def load_settings(src_root: str) -> Settings:
    u"""Build the settings the process will run with, applying environment overrides."""
    gate = EngineSettings(
        model_path=_env_str('DT_GATE_MODEL', GATE_MODEL_DEFAULT),
        max_model_len=5250,
        gpu_memory_utilization=_env_float('DT_GATE_UTIL', 0.90),
        attention_backend=None,             # forcing FlashInfer here is 1.56x SLOWER (measured)
        expect_impl=('Triton',),
        weights_gib=24.75,
        boot_timeout_s=_env_int('DT_BOOT_TIMEOUT', 2400),
    )
    champion = EngineSettings(
        model_path=_env_str('DT_CHAMP_MODEL', CHAMPION_MODEL_DEFAULT),
        max_model_len=5250,
        gpu_memory_utilization=_env_float('DT_CHAMP_UTIL', 0.94),
        attention_backend='FLASHINFER',
        expect_impl=('FlashInfer',),
        weights_gib=30.63,
        boot_timeout_s=_env_int('DT_BOOT_TIMEOUT', 2400),
    )
    cascade = CascadeSettings(
        routed_fraction=_env_float('DT_FRAC', 1.0),
        readout=_env_str('DT_READOUT', 'ipf'),
        lam=_env_float('DT_LAM', 0.5),
        router=_env_str('DT_ROUTER', 'margin'),
        prefix_align=_env_str('DT_PREFIX_ALIGN', 'off'),
    )
    return Settings(
        gate=gate,
        champion=champion,
        cascade=cascade,
        artifact_dir=_env_str('DT_ARTIFACT_DIR', '/tmp/dt_artifacts'),
        shots_path=os.path.join(src_root, 'assets', 'shots_gfs4.txt'),
    )


def with_fraction(settings: Settings, fraction: float) -> Settings:
    u"""Return settings with a different routed fraction. Used by the A/B tooling, not by judging."""
    return replace(settings, cascade=replace(settings.cascade, routed_fraction=fraction))
