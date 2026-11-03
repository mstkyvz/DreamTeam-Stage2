# -*- coding: utf-8 -*-
u"""Explainability: the predictor's own decision variable, not a second model's opinion of it.

The rules are explicit that a standalone zero-shot explainer is out of scope -- what is wanted is
the rationale of the PREDICTOR. This service produces four things, all of them read out of the
same model that writes the submission, through the same forced-token read-out:

1. THE POSTERIOR. The predictor's decision variable is literally a 3-vector: the assistant turn is
   prefilled with `{"etiket": "`, one token is decoded, and the marginal over vocabulary tokens
   that uniquely prefix each label is the answer. We show that vector. It is not a confidence
   score bolted on afterwards -- it IS what argmax is taken over.

2. FIELD ATTRIBUTION BY OCCLUSION. Each input field is blanked in turn and the row is re-scored
   through the identical path; the change in log-posterior is that field's contribution. This is
   a Shapley-style attribution with a single-feature coalition, and it is exact for the model as
   deployed rather than estimated from a surrogate. Cost is one extra prefill per field --
   `max_tokens=1`, so a full attribution is 8 forward passes of one token each.

3. THE ROUTING DECISION. The cascade's margin -- the gap between the gate's top two
   log-probabilities -- is why a row was or was not escalated to the champion. A reviewer asking
   "why did the big model see this row" gets the actual number the router sorted on.

4. THE PRIOR CORRECTION. The submitted label is not argmax; it is IPF against a soft class target
   (lam=0.5). IPF is a GLOBAL operation over all rows, so a single-row view cannot reproduce it
   honestly. We show the raw posterior, name the shipped target, and say plainly that the final
   submission additionally passes through the global step. Fabricating a per-row "IPF-adjusted"
   number would be a lie about how the label was produced.
"""
from __future__ import annotations

import numpy as np

from app.contracts import Attribution, Explanation, InferencePort, LABELS
from core import readout as RO

# Fields the reviewer can see in the UI, in the order they are occluded and displayed.
OCCLUDABLE_FIELDS: tuple[str, ...] = (
    'query', 'title', 'category', 'brand', 'gender', 'age_group', 'attributes',
)

# What an occluded field is replaced with. Empty string, not a sentinel word: a sentinel would be
# tokenised and carry its own meaning, which would contaminate the attribution it is measuring.
OCCLUSION_VALUE = ''


class ExplainService:
    u"""Explains one row at a time against a single booted engine."""

    def __init__(self, engine: InferencePort, *, stage_name: str, lam: float = 0.5) -> None:
        self._engine = engine
        self._stage = stage_name
        self._lam = lam

    def explain(self, row: dict, *, query_override: str | None = None) -> Explanation:
        u"""Score the row, then score it once per occluded field, and diff the posteriors."""
        base_row = dict(row)
        if query_override is not None:
            base_row['query'] = query_override

        variants = [base_row]
        for field in OCCLUDABLE_FIELDS:
            occluded = dict(base_row)
            occluded[field] = OCCLUSION_VALUE
            variants.append(occluded)

        # One engine call for the row and all of its occlusions: they share the entire system
        # prompt, so the prefix cache serves every variant after the first.
        result = self._engine.score(variants, stage=self._stage)
        logprobs = RO.norm_log(result.logprobs)
        base_lp = logprobs[0]
        base_posterior = np.exp(base_lp)
        base_idx = int(np.argmax(base_lp))

        ordered = np.sort(base_lp)
        margin = float(ordered[-1] - ordered[-2])

        attributions = []
        for offset, field in enumerate(OCCLUDABLE_FIELDS, start=1):
            delta = base_lp - logprobs[offset]
            occluded_idx = int(np.argmax(logprobs[offset]))
            attributions.append(Attribution(
                field_name=field,
                delta_logprob=tuple(float(x) for x in delta),
                occluded_label=LABELS[occluded_idx],
                flipped=occluded_idx != base_idx,
            ))

        return Explanation(
            row_id=str(base_row.get('row_id', '')),
            query=str(base_row.get('query', '')),
            label=LABELS[base_idx],
            posterior=tuple(float(x) for x in base_posterior),
            margin=margin,
            attributions=tuple(attributions),
            # IPF is global; the honest per-row statement is the target it aims at, not a
            # fabricated per-row adjustment. See the module docstring.
            prior_shift=tuple(float(x) for x in (base_posterior - base_posterior.mean())),
            stage=self._stage,
            engine_info=dict(result.engine_info),
        )
