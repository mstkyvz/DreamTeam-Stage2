# -*- coding: utf-8 -*-
u"""Reading the organizer's CSV and writing the submission, with the contract enforced at both ends.

Two rules from the competition text drive everything here:

* "Text fields may contain commas, quotes, and embedded newlines. Teams must use a CSV parser
  that supports quoted multiline fields; line-by-line splitting is not valid." -- so pandas with
  the default C parser, never a hand-rolled split.
* "row_id is a unique submission key. It must be copied unchanged." -- so row_id is read as `str`
  with `keep_default_na=False` and never passed through a numeric dtype, which would silently
  turn `007` into `7` and `NA` into NaN.

The write is atomic: a temporary file in the same directory, then `replace()`. A judge that reads
the output while we are still writing must never see a half-file, and exit code 0 must mean the
whole file is there.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.contracts import LABELS, REQUIRED_COLUMNS
from app.errors import InputContractError, OutputContractError


def read_input(path: Path) -> pd.DataFrame:
    u"""Load and validate the test CSV. Raises InputContractError on anything the contract forbids."""
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:                      # malformed CSV is an input-contract failure
        raise InputContractError('cannot parse %s as CSV: %s' % (path, exc)) from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise InputContractError('input CSV is missing required columns: %r' % (missing,))
    if frame['row_id'].duplicated().any():
        dupes = int(frame['row_id'].duplicated().sum())
        raise InputContractError('input CSV contains %d duplicate row_id values' % dupes)
    if frame['row_id'].eq('').any():
        raise InputContractError('input CSV contains empty row_id values')
    if frame.empty:
        raise InputContractError('input CSV has no rows')
    return frame


def write_submission(row_ids: pd.Series, labels: list[str], path: Path) -> None:
    u"""Write `row_id,real_label` atomically, validating before the file becomes visible."""
    if len(row_ids) != len(labels):
        raise OutputContractError('have %d row_ids and %d labels' % (len(row_ids), len(labels)))
    invalid = sorted(set(labels) - set(LABELS))
    if invalid:
        raise OutputContractError('labels outside the fixed set: %r' % (invalid,))

    out = pd.DataFrame({'row_id': row_ids.to_numpy(copy=True), 'real_label': labels})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.partial' % path.name)
    out.to_csv(tmp, index=False)
    tmp.replace(path)


def verify_submission(input_path: Path, output_path: Path) -> dict[str, object]:
    u"""Re-read both files from disk and prove the output satisfies the contract.

    This runs after the write, deliberately re-reading rather than trusting the in-memory frame,
    so the check covers truncation, encoding damage and partial writes -- not just our own logic.
    """
    src = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    got = pd.read_csv(output_path, dtype=str, keep_default_na=False)

    if list(got.columns) != ['row_id', 'real_label']:
        raise OutputContractError('output columns are %r, contract requires [row_id, real_label]'
                                  % (list(got.columns),))
    if len(got) != len(src):
        raise OutputContractError('output has %d rows, input has %d' % (len(got), len(src)))
    if got['row_id'].duplicated().any():
        raise OutputContractError('output contains duplicate row_id values')

    src_ids, got_ids = set(src['row_id']), set(got['row_id'])
    if src_ids != got_ids:
        raise OutputContractError('row_id mismatch: %d missing, %d unknown'
                                  % (len(src_ids - got_ids), len(got_ids - src_ids)))
    bad = sorted(set(got['real_label']) - set(LABELS))
    if bad:
        raise OutputContractError('output contains labels outside the fixed set: %r' % (bad,))

    return {
        'rows': int(len(got)),
        'columns': list(got.columns),
        'label_counts': {k: int(v) for k, v in got['real_label'].value_counts().items()},
    }
