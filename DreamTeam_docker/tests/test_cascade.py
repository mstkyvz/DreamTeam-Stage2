# -*- coding: utf-8 -*-
u"""CPU tests for the orchestration layer. No GPU, no vLLM, no model weights.

These cover the properties that silently break a submission rather than crashing it: rows coming
back in a different order, the champion's answers landing on the wrong rows, an engine surviving
a failed stage and starving the next one, and the output contract drifting.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, SRC)

from adapters.mock_engine import MockInferenceAdapter          # noqa: E402
from app.contracts import LABELS, REQUIRED_COLUMNS, StageResult  # noqa: E402
from app.errors import EngineError, InputContractError, OutputContractError  # noqa: E402
from services.cascade_service import CascadeService            # noqa: E402
from services.io_service import read_input, verify_submission, write_submission  # noqa: E402
from services.readout_service import IpfReadout                # noqa: E402
from services.routing_service import MarginRouter, RO_margin   # noqa: E402


def make_frame(n=40):
    return pd.DataFrame({
        'row_id': ['r%03d' % i for i in range(n)],
        'query': ['sorgu %d' % i for i in range(n)],
        'title': ['urun %d' % i for i in range(n)],
        'category': ['kategori'] * n,
        'brand': ['marka'] * n,
        'gender': ['kadin'] * n,
        'age_group': ['yetiskin'] * n,
        'attributes': ['renk: mavi, beden: M'] * n,
    })


def build(fraction=0.25, gate=None, champion=None):
    gate = gate if gate is not None else MockInferenceAdapter(name='gate', sharpness=0.4)
    champion = champion if champion is not None else MockInferenceAdapter(name='champ', sharpness=3.0)
    service = CascadeService(
        gate_factory=(lambda: gate),
        champion_factory=(lambda: champion),
        router=MarginRouter(),
        readout=IpfReadout('ipf', 0.5),
        routed_fraction=fraction,
    )
    return service, gate, champion


# ----------------------------------------------------------------------------- contract shape
def test_one_label_per_row_in_input_order():
    frame = make_frame(40)
    service, _, _ = build()
    labels = service.run(frame)
    assert len(labels) == len(frame)
    assert set(labels) <= set(LABELS)


def test_routed_fraction_is_respected():
    frame = make_frame(100)
    service, _, champion = build(fraction=0.15)
    service.run(frame)
    assert champion.scored_rows == 15
    assert service.manifest['composition']['from_champion'] == 15
    assert service.manifest['composition']['from_gate'] == 85


def test_champion_only_when_fraction_is_one():
    frame = make_frame(20)
    champion = MockInferenceAdapter(name='champ')
    service = CascadeService(None, (lambda: champion), MarginRouter(),
                             IpfReadout('argmax'), routed_fraction=1.0)
    labels = service.run(frame)
    assert champion.scored_rows == 20
    assert len(labels) == 20
    assert service.manifest['routing']['source'] == 'no_gate'


# --------------------------------------------------------------- composition puts rows correctly
def test_composition_places_champion_rows_at_their_own_indices():
    u"""The bug this guards: champion answers scattered back by rank instead of by row index."""
    n = 30
    gate = np.tile(np.array([0.0, -1.0, -2.0]), (n, 1))
    champ_rows = np.array([3, 11, 25])
    champ = np.tile(np.array([-5.0, -5.0, 0.0]), (len(champ_rows), 1))
    composed, sources = CascadeService._compose(
        n,
        StageResult(logprobs=gate, rows=n, prompt_tokens=0),
        StageResult(logprobs=champ, rows=len(champ_rows), prompt_tokens=0),
        champ_rows,
    )
    assert composed.shape == (n, 3)
    assert sources.sum() == len(champ_rows)
    for i in champ_rows:
        assert composed[i].argmax() == 2
    for i in set(range(n)) - set(champ_rows.tolist()):
        assert composed[i].argmax() == 0


def test_engine_is_closed_even_when_a_stage_raises():
    u"""55.38 GiB of weights on a 40 GiB card: a leaked engine starves the next boot."""
    gate = MockInferenceAdapter(name='gate', fail_on_stage='gate')
    service, _, _ = build(gate=gate)
    with pytest.raises(EngineError):
        service.run(make_frame(10))
    assert gate.closed, 'gate engine was not released after a failed stage'


# ------------------------------------------------------------------------------------- router
def test_margin_scores_uncertain_rows_higher():
    certain = np.array([[0.0, -9.0, -9.0]])
    unsure = np.array([[0.0, -0.05, -9.0]])
    assert RO_margin(unsure)[0] > RO_margin(certain)[0]


def test_routing_is_deterministic_across_runs():
    frame = make_frame(60)
    first, _, _ = build(fraction=0.2)
    second, _, _ = build(fraction=0.2)
    assert first.run(frame) == second.run(frame)


# ------------------------------------------------------------------------------------ read-out
def test_ipf_moves_marginals_toward_the_target_without_pinning():
    rng = np.random.default_rng(0)
    skewed = rng.normal(size=(600, 3)) + np.array([3.0, 0.0, 0.0])
    argmax_counts = np.bincount(skewed.argmax(1), minlength=3)
    ipf_counts = np.bincount(IpfReadout('ipf', 0.5).decide(skewed), minlength=3)
    # closer to balanced than argmax, but not pinned to exactly n/3
    assert ipf_counts.std() < argmax_counts.std()
    assert ipf_counts.std() > 0


# ------------------------------------------------------------------------------------ CSV I/O
def test_read_input_rejects_duplicate_row_ids(tmp_path):
    frame = make_frame(4)
    frame.loc[3, 'row_id'] = frame.loc[0, 'row_id']
    path = tmp_path / 'dupes.csv'
    frame.to_csv(path, index=False)
    with pytest.raises(InputContractError):
        read_input(path)


def test_read_input_rejects_missing_columns(tmp_path):
    frame = make_frame(4).drop(columns=['attributes'])
    path = tmp_path / 'short.csv'
    frame.to_csv(path, index=False)
    with pytest.raises(InputContractError):
        read_input(path)


def test_row_ids_with_commas_quotes_and_newlines_survive(tmp_path):
    u"""The rules warn that text fields carry commas, quotes and embedded newlines."""
    frame = make_frame(3)
    frame.loc[0, 'title'] = 'gomlek, "mavi"\nikinci satir'
    frame.loc[1, 'query'] = 'a,b,"c"'
    src = tmp_path / 'in.csv'
    frame.to_csv(src, index=False)
    out = tmp_path / 'out.csv'
    write_submission(read_input(src)['row_id'], ['relevant'] * 3, out)
    report = verify_submission(src, out)
    assert report['rows'] == 3


def test_leading_zero_row_ids_are_not_renumbered(tmp_path):
    u"""row_id must be copied unchanged: `007` must not become `7`."""
    frame = make_frame(3)
    frame['row_id'] = ['007', '010', '100']
    src = tmp_path / 'in.csv'
    frame.to_csv(src, index=False)
    out = tmp_path / 'out.csv'
    write_submission(read_input(src)['row_id'], ['relevant'] * 3, out)
    assert pd.read_csv(out, dtype=str)['row_id'].tolist() == ['007', '010', '100']


def test_write_submission_rejects_labels_outside_the_fixed_set(tmp_path):
    with pytest.raises(OutputContractError):
        write_submission(pd.Series(['a', 'b']), ['relevant', 'RELEVANT'], tmp_path / 'x.csv')


def test_verify_detects_a_truncated_output(tmp_path):
    frame = make_frame(5)
    src = tmp_path / 'in.csv'
    frame.to_csv(src, index=False)
    out = tmp_path / 'out.csv'
    write_submission(frame['row_id'][:4], ['relevant'] * 4, out)
    with pytest.raises(OutputContractError):
        verify_submission(src, out)


def test_end_to_end_writes_a_valid_submission(tmp_path):
    frame = make_frame(50)
    src = tmp_path / 'in.csv'
    frame.to_csv(src, index=False)
    service, _, _ = build(fraction=0.1)
    labels = service.run(read_input(src))
    out = tmp_path / 'sub.csv'
    write_submission(frame['row_id'], labels, out)
    report = verify_submission(src, out)
    assert report['rows'] == 50
    assert set(report['label_counts']) <= set(LABELS)


# -------------------------------------------------------------------- real prompt construction
# These need no GPU and no weights, but they DO exercise `core/`, which the mock engine path
# skips entirely. That gap let a missing vendored module (`l10_run`) ship undetected: every test
# and the smoke test passed while the real champion prompt could not be built at all.
def test_the_shipped_few_shot_asset_is_present_and_nonempty():
    from app.config import load_settings
    from app.wiring import _shot_block
    shots = _shot_block(load_settings(SRC))
    assert len(shots) > 1000, 'few-shot asset looks truncated: %d chars' % len(shots)


def test_champion_system_prompt_builds_from_the_vendored_core():
    u"""Guards the vendored dependency graph: core.prompts reaches into core/champ at import time."""
    sys.path.insert(0, os.path.join(SRC, 'core', 'champ'))
    from app.config import load_settings
    from app.wiring import _shot_block
    from core import prompts as PR

    settings = load_settings(SRC)
    sysmsg = PR.champion_sysmsg(_shot_block(settings))
    assert len(sysmsg) > 5000, 'champion system message is suspiciously short: %d' % len(sysmsg)
    for label in LABELS:
        assert label in sysmsg, 'label %r missing from the rubric' % label


def test_readout_label_order_matches_the_contract():
    u"""LABELS order is load-bearing: every (N,3) matrix, the router and the artifacts assume it."""
    from core import readout as RO
    assert tuple(RO.LABELS) == LABELS
