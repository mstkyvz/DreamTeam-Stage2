#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Solution mode. The organizer's contract, and nothing else.

    main.py INPUT_CSV OUTPUT_CSV     ->  CSV with exactly `row_id,real_label`

Exit codes are part of the contract: 0 means the complete, verified output is on disk; 2 means
the invocation or the input was wrong; 1 means we failed while producing it. Nothing here
degrades a failure into a plausible-looking CSV -- a wrong submission that exits 0 is worse than
a run that exits nonzero, because only one of the two is visible to the team.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
# The vendored champion prompt modules import each other by bare name.
sys.path.insert(0, os.path.join(SRC_ROOT, 'core', 'champ'))

from app.config import load_settings                                    # noqa: E402
from app.errors import PipelineError, UsageError                        # noqa: E402
from app.wiring import build_cascade                                    # noqa: E402
from services.io_service import read_input, verify_submission, write_submission   # noqa: E402


def run(input_csv: Path, output_csv: Path) -> dict[str, object]:
    settings = load_settings(SRC_ROOT)
    print('[main] configuration: %s' % json.dumps(settings.summary), flush=True)

    frame = read_input(input_csv)
    print('[main] %d rows from %s' % (len(frame), input_csv), flush=True)

    cascade = build_cascade(settings)
    labels = cascade.run(frame)

    write_submission(frame['row_id'], labels, output_csv)
    report = verify_submission(input_csv, output_csv)

    manifest = dict(cascade.manifest)
    manifest['settings'] = settings.summary
    manifest['output'] = report
    _write_manifest(settings.artifact_dir, manifest)
    return report


def _write_manifest(artifact_dir: str, manifest: dict[str, object]) -> None:
    u"""Best-effort run record. A manifest that cannot be written must not fail a good run."""
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, 'run_manifest.json'), 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh, indent=1, ensure_ascii=False, default=str)
    except OSError as exc:
        print('[main] note: manifest not written (%s)' % exc, flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    started = time.time()
    try:
        if len(argv) != 2:
            raise UsageError('usage: main.py INPUT_CSV OUTPUT_CSV')
        input_csv, output_csv = Path(argv[0]), Path(argv[1])
        if not input_csv.is_file():
            raise UsageError('input CSV does not exist: %s' % input_csv)
        if output_csv.exists() and input_csv.resolve() == output_csv.resolve():
            raise UsageError('input and output paths must differ')

        report = run(input_csv, output_csv)
    except PipelineError as exc:
        sys.stderr.write('[main] %s: %s\n' % (type(exc).__name__, exc))
        return exc.exit_code
    except Exception:
        traceback.print_exc()
        return 1

    print('[main] wrote %d rows to %s in %.1f s'
          % (report['rows'], output_csv, time.time() - started), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
