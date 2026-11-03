# -*- coding: utf-8 -*-
u"""Explainability mode: a local UI over the SAME predictor that writes the submission.

    server.py PUBLIC_TEST_CSV --host 127.0.0.1 --port 7860

WHICH MODEL THIS SERVES, AND WHY ONLY ONE. The cascade uses two checkpoints that cannot co-reside
in 40 GB (24.75 + 30.63 GiB). Solution mode gets around that by loading them sequentially, which
an interactive UI cannot do -- swapping engines per request would cost ~40 s of boot each time.
So the UI boots the CHAMPION, which is the model that decides every escalated row and the one
whose rationale a reviewer is actually asking about. The page says so; nothing here implies the
gate was consulted for a row it never saw.

WHAT IS AND IS NOT REPRODUCED HERE. The posterior, the margin and the field attributions are
exactly what the predictor computes in solution mode. The final IPF read-out is NOT reproducible
per row -- it is a global operation over all 90,000 rows -- so the UI shows the raw posterior and
names the read-out rather than inventing a per-row adjusted number.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, os.path.join(SRC_ROOT, 'core', 'champ'))

import numpy as np                                                      # noqa: E402
from fastapi import FastAPI, HTTPException                              # noqa: E402
from fastapi.responses import HTMLResponse                              # noqa: E402
from pydantic import BaseModel, Field                                   # noqa: E402

from api.ui import render_page                                          # noqa: E402
from app.config import load_settings                                    # noqa: E402
from app.contracts import LABELS, REQUIRED_COLUMNS                      # noqa: E402
from app.wiring import build_engine_factories                           # noqa: E402
from services.explain_service import ExplainService                     # noqa: E402
from services.io_service import read_input                              # noqa: E402

PIPELINE_NOTE = (
    'Gönderilen etiket iki aşamalı bir kademeden çıkar: ucuz bir <b>kapıcı</b> (MoE, 24,75 GiB) '
    'her satırı puanlar, en belirsiz %{frac:.0f}\'ı <b>şampiyona</b> (31B INT8, 30,63 GiB) '
    'yükseltilir, sonra tüm satırlar üzerinde küresel bir <b>IPF</b> okuması (λ={lam}) sınıf '
    'marjinallerini yumuşak bir hedefe çeker. İki kontrol noktası 40 GiB\'a birlikte sığmadığı '
    'için bu arayüz yalnızca <b>şampiyonu</b> yükler — yükseltilen her satıra karar veren model '
    'odur. IPF küresel olduğundan tek satırda yeniden üretilemez; burada gösterilen ham '
    'sonsal olasılıktır.'
)


class ExplainRequest(BaseModel):
    row_id: str = Field(min_length=1)
    query: str = ''


def create_app(input_csv: Path) -> FastAPI:
    state: dict = {}
    settings = load_settings(SRC_ROOT)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        frame = read_input(input_csv)
        state['frame'] = frame.set_index('row_id', drop=False)
        state['row_ids'] = frame['row_id'].tolist()

        _, champion_factory = build_engine_factories(settings)
        engine = champion_factory()
        engine.boot()
        state['engine'] = engine
        state['explainer'] = ExplainService(engine, stage_name='champion',
                                            lam=settings.cascade.lam)
        # One engine, one request at a time: vLLM's LLM object is not re-entrant, and an
        # interactive UI has no batching to gain from concurrency anyway.
        state['lock'] = threading.Lock()
        state['ready'] = True
        print('[serve] explainability ready: %d rows, champion %s'
              % (len(frame), settings.champion.model_path), flush=True)
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(title='Relevance Predictor Explainability', lifespan=lifespan)

    @app.get('/', response_class=HTMLResponse)
    def index() -> str:
        return render_page()

    @app.get('/health')
    def health() -> dict:
        u"""The entrypoint polls this before starting ngrok; it must only pass once the engine is up."""
        if not state.get('ready'):
            raise HTTPException(status_code=503, detail='engine still loading')
        return {'status': 'ok', 'rows': len(state['frame'])}

    @app.get('/api/meta')
    def meta() -> dict:
        info = state['engine'].info
        return {
            'rows': len(state['frame']),
            'stage': 'champion',
            'model': os.path.basename(settings.champion.model_path),
            'attention_impl': info.get('attention_impl'),
            'routed_fraction': settings.cascade.routed_fraction,
            'pipeline_note': PIPELINE_NOTE.format(
                frac=settings.cascade.routed_fraction * 100, lam=settings.cascade.lam),
        }

    @app.get('/api/rows')
    def rows() -> dict:
        return {'row_ids': state['row_ids']}

    @app.get('/api/item/{row_id}')
    def item(row_id: str) -> dict:
        if row_id not in state['frame'].index:
            raise HTTPException(status_code=404, detail='bilinmeyen row_id')
        row = state['frame'].loc[row_id]
        return {c: ('' if c not in row else str(row[c])) for c in REQUIRED_COLUMNS}

    @app.post('/api/explain')
    def explain(request: ExplainRequest) -> dict:
        if request.row_id not in state['frame'].index:
            raise HTTPException(status_code=404, detail='bilinmeyen row_id')
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail='sorgu boş olamaz')

        row = state['frame'].loc[request.row_id].to_dict()
        with state['lock']:
            result = state['explainer'].explain(row, query_override=query)

        predicted = LABELS.index(result.label)
        return {
            'row_id': result.row_id,
            'query': result.query,
            'label': result.label,
            'posterior': list(result.posterior),
            'margin': result.margin,
            'stage': result.stage,
            'attributions': [
                {
                    'field_name': a.field_name,
                    'delta_logprob': list(a.delta_logprob),
                    # The number the waterfall draws: how much this field supported the class the
                    # model actually chose. Positive = removing the field would weaken that class.
                    'delta_for_predicted': float(a.delta_logprob[predicted]),
                    'occluded_label': a.occluded_label,
                    'flipped': a.flipped,
                }
                for a in result.attributions
            ],
        }

    return app


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    import uvicorn
    args = parse_args(argv)
    uvicorn.run(create_app(args.input_csv), host=args.host, port=args.port, log_level='info')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
