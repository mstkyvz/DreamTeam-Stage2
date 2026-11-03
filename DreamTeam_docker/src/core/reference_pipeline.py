#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Trendyol stage-2 multiclass -- the delivered system, end to end, one command.

    python3 predict.py --input public_test.csv --out preds.csv

    load
      -> ENGINE(s): gate over every row, champion over the routed rows
      -> router features + router score + top-frac selection
      -> compose (champion where routed, gate elsewhere)
      -> read-out (IPF soft count balance)        -> preds.csv + run_manifest.json

THE THREE THINGS THIS SCRIPT IS CAREFUL ABOUT, because the run got each of them wrong once:

* **Roughly half the wall clock is not inference.**  L33 measured a warm sequential cascade at
  ~177 s of which only 86.1 s is `generate()`: engine setup x2 41.6 s, prompt tokenisation x2
  31.7 s, import 10.7 s, teardown 6.7 s.  So this script shares the tokenisation between the two
  stages when the prompt and the tokenizer allow it, runs both stages in ONE process, and times
  boot, tokenisation, generate and teardown into separate manifest lines.  The gate choice decides
  how many boots there are:
  `c3p_t` shares the champion's checkpoint (ONE boot); the MoE does not, and 24.75 + 30.63 =
  55.38 GiB of weights do not co-reside in 40 GiB (measured: the champion then reports
  `Available KV cache memory: -21.61 GiB`), so that path is load-run-free-load-run, TWO boots.
  Measured eager boots: gate 63.4 s first-on-box / 19.5 s warm, champion 23.6 / 22.2 s.
* **The read-out is IPF with a soft target, not a hard quota pin.**  Measured at 30% routing:
  IPF(lam=0.5) 0.9133, plain argmax 0.9079, greedy pin 0.9040.  `--pin` still exists because
  single-source output wants it; it is not the default and it warns on a mixed-source matrix.
* **The router cannot be fitted at inference time** -- its label is "does the gate disagree with
  the champion", which is what the cascade exists not to know yet.  `resolve_router` requires a
  pre-fitted model or a precomputed score, and falls back to the gate margin loudly rather than
  producing a router-shaped number out of nothing.

Read-out mechanics, unchanged from every measured arm in this run: the assistant turn is prefilled
with `{"etiket": "`, EXACTLY ONE token is decoded with `logprobs`, and the 3-vector is the
marginal over every returned vocabulary token that is a unique prefix of exactly one label.
`max_tokens=1`; nothing is ever generated.
"""
import os, sys, json, time, argparse, signal, hashlib
_T_IMPORT = time.time()
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'lib', 'champ'))

from lib import readout as RO
from lib import routing as RT
from lib import prompts as PR
from lib.engine import PROFILES, VllmJudge, MockJudge, MockTokenizer

IMPORT_S = time.time() - _T_IMPORT     # L33 measured 10.7 s of this in a 177 s cascade
LABELS = RO.LABELS
NEEDED_COLS = ['row_id', 'query', 'title', 'category', 'brand', 'gender', 'age_group',
               'attributes']

# name          -> (gate profile, champion profile, routed fraction)
CONFIGS = {
    'c3pt30':     ('gate_c3p_t', 'champ_int8_fi',     0.30),
    'c3pt20':     ('gate_c3p_t', 'champ_int8_fi',     0.20),
    'moe30':      ('gate_moe',   'champ_int8_fi',     0.30),
    'moe25':      ('gate_moe',   'champ_int8_fi',     0.25),
    'moe20':      ('gate_moe',   'champ_int8_fi',     0.20),
    'champ-only': (None,         'champ_int8_fi',     1.00),   # 269 s, no cascade, one engine
    'champ-safe': (None,         'champ_int8_triton', 1.00),   # bit-identical to L20's INT8
    'champ-fp8':  (None,         'champ_fp8',         1.00),   # the 0.9133 accuracy reference
}
DEFAULT_CONFIG = 'moe30'


# ------------------------------------------------------------------------------------- helpers
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


class Timer(object):
    def __init__(self, store, key):
        self.store, self.key = store, key

    def __enter__(self):
        self.t0 = time.time(); return self

    def __exit__(self, *a):
        self.store[self.key] = round(time.time() - self.t0, 3)


class BootTimeout(Exception):
    pass


def _with_timeout(seconds, fn, what):
    u"""FRAGILITY 1 -- engine boot hangs.  A stalled vLLM boot (half-downloaded shard, wedged NCCL
    init, compile cache rebuild) hangs silently and forever, which on a rented box is money and a
    lost run.  SIGALRM turns it into an error with a name.  Absent SIGALRM this degrades to no
    timeout, never to a wrong answer."""
    if not seconds or not hasattr(signal, 'SIGALRM'):
        return fn()

    def _h(signum, frame):
        raise BootTimeout('%s did not come up within %d s (raise --boot-timeout, or look for a '
                          'partial HF download / a stuck compile cache)' % (what, seconds))
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(int(seconds))
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def load_rows(path):
    u"""FRAGILITY 8 -- input surprises.  A duplicated row_id silently corrupts every keyed lookup
    downstream; a missing optional column silently changes the prompt, because `render_item` omits
    empty fields rather than printing them blank."""
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    assert 'row_id' in df.columns, 'input has no row_id column: %r' % (list(df.columns),)
    missing = [c for c in NEEDED_COLS if c not in df.columns]
    for c in missing:
        df[c] = ''
    if missing:
        print('[L38] WARNING: input is missing %r -- filled with empty strings. This CHANGES the '
              'prompt and therefore the answer.' % missing, flush=True)
    dup = int(df.row_id.duplicated().sum())
    assert dup == 0, '%d duplicated row_id values; every keyed lookup downstream assumes unique' % dup
    return df.reset_index(drop=True), missing


def load_shot_block(args):
    u"""The champion arm is `gfs4/champ3`: rubric `champ3` plus a GLOBAL few-shot block drawn from
    train.parquet with seed 4.  The draw is deterministic, so the RENDERED block ships as an asset
    with its SHA and the deliverable needs no 500,000-row training file at inference.  With
    `--train` the block is rebuilt and compared; a mismatch is fatal, because a silently different
    shot block is a silently different model."""
    import l10_run as L10
    built = None
    if args.train:
        built = L10.build_global_shots(pd.read_parquet(args.train), n_pair=4, n_extra=3,
                                       seed=4, order='label')
    if args.shots and os.path.exists(args.shots):
        txt = open(args.shots, encoding='utf-8').read()
        if built is not None and built != txt:
            raise RuntimeError(
                'shot block rebuilt from %s does not match cached %s (sha %s vs %s). One of the '
                'two is not the champion arm.' % (
                    args.train, args.shots, hashlib.sha256(built.encode()).hexdigest()[:12],
                    hashlib.sha256(txt.encode()).hexdigest()[:12]))
        return txt, args.shots, hashlib.sha256(txt.encode()).hexdigest()
    if built is None:
        raise RuntimeError('no shot block: pass --shots <cached file> or --train <train.parquet>')
    return built, None, hashlib.sha256(built.encode()).hexdigest()


# ------------------------------------------------------------------------------- one judge pass
def judge(engine, recs, sysmsg, prompt_spec, args, stage, timings, N_total, cache=None):
    u"""One forward pass: build prompts, decode one token, read the label marginal out."""
    from l9_judge import build_label_token_sets, assert_label_tokens
    from l9_run import read_out

    tok = engine.get_tokenizer()
    with Timer(timings, '%s_labeltokens_s' % stage):
        idx, cls = build_label_token_sets(tok)
        assert_label_tokens(tok, idx, cls)     # refuses to run if the read position is unsound
    id2cls = dict(zip(idx.tolist(), cls.tolist()))

    row_ids = [r['row_id'] for r in recs]
    enc_fn = (PR.encode_champion if prompt_spec['fields'] is None
              else (lambda t, sm, rs, ma, mml: PR.encode_gate(t, sm, rs, prompt_spec['fields'],
                                                              ma, mml)))

    # ---- cache-block alignment of the shared prefix (F190) ---------------------------------
    # The shared prefix is hashed in fixed-size blocks from position 0.  If it does not end on a
    # block boundary, its tail shares a block with row-specific body tokens and that block can
    # never be cached -- 4.6% of every computed token, for nothing.  The target is
    # `prefix + constant_body_head ≡ 0 (mod G)`.  Both the head length and the achieved
    # alignment are MEASURED; nothing here is assumed, because being one token off turns a
    # -6.2% win into a +1.8% loss.
    align = {'mode': args.prefix_align}
    if args.prefix_align != 'off':
        head = PR.constant_body_head_tokens(tok, sysmsg, recs, enc_fn, args.max_attr,
                                            args.max_model_len)
        measure = lambda pad: enc_fn(tok, sysmsg + pad, recs[:1], args.max_attr,
                                     args.max_model_len)[1]
        align['body_head_tokens'] = int(head)
        if args.prefix_align == 'pad':
            akey = (prompt_spec['kind'], tuple(prompt_spec['fields'] or ()), len(sysmsg), head)
            if cache is not None and akey in getattr(cache, 'aligns', {}):
                pad, before, after = cache.aligns[akey]
            else:
                pad, before, after = PR.find_prefix_pad(measure, head)
                if cache is not None:
                    cache.aligns[akey] = (pad, before, after)
            sysmsg = sysmsg + pad
            align.update(pad_units=len(pad), prefix_before=int(before), prefix_after=int(after),
                         granularity=PR.ALIGN_GRANULARITY)
        else:                                            # 'assert': measure, do not change
            align.update(prefix_before=int(measure(u'')), granularity=PR.ALIGN_GRANULARITY)

    ck = PR.PromptCache.key(prompt_spec['kind'], prompt_spec['fields'], sysmsg,
                            PR.tokenizer_fingerprint(tok))
    with Timer(timings, '%s_prompt_build_s' % stage):
        hit = cache.get(ck, row_ids) if cache is not None else None
        if hit is not None:
            ids_list, Pn, plens = hit
            print('[L38] %-7s prompt cache HIT -- the gate already tokenised these rows with an '
                  'identical prompt and an identical tokenizer' % stage, flush=True)
        else:
            if prompt_spec['fields'] is None:
                ids_list, Pn, plens = PR.encode_champion(tok, sysmsg, recs, args.max_attr,
                                                         args.max_model_len)
            else:
                ids_list, Pn, plens = PR.encode_gate(tok, sysmsg, recs, prompt_spec['fields'],
                                                     args.max_attr, args.max_model_len)
            if cache is not None:
                cache.put(ck, row_ids, ids_list, Pn)
    print('[L38] %-7s %5d rows | shared prefix %d tok | body mean %.0f | max %d'
          % (stage, len(recs), Pn, float(np.mean(plens)) - Pn, max(plens)), flush=True)

    # THE ASSERT F190 ASKS FOR.  The prefix length is checked against what the alignment step
    # thought it produced, and against the congruence itself -- a builder that drifted from the
    # measuring helper, or a cache entry from a differently-padded prompt, both land here.
    if args.prefix_align != 'off':
        G, head = PR.ALIGN_GRANULARITY, align.get('body_head_tokens', 0)
        align['prefix_actual'] = int(Pn)
        align['residue'] = int(Pn % G)
        align['aligned'] = (Pn + head) % G == 0
        if args.prefix_align == 'pad' and not align['aligned']:
            raise AssertionError(
                'prefix alignment FAILED: prompts were built with a %d-token shared prefix, '
                '%d mod %d, and the constant body head is %d, so the first body block still '
                'starts mid-block. Expected residue %d. One token off is +4.8 s at a tight cache '
                'cap -- refusing to run a misaligned prefix that was believed to be aligned.'
                % (Pn, Pn % G, G, head, (-head) % G))
        print('[L38] %-7s prefix alignment: %d tok = %d mod %d, body head %d -> %s'
              % (stage, Pn, Pn % G, G, head,
                 'ALIGNED (first body block starts on a boundary)' if align['aligned']
                 else 'MISALIGNED, %d tokens/row stranded' % ((-(Pn + head)) % G)), flush=True)
    timings.setdefault('_align', {})[stage] = align
    # the pool has to be checked against the prefix that was actually built, so it happens here
    # rather than at boot -- still before a single row is judged.
    if getattr(engine, 'check_headroom', None):
        align['kv'] = engine.check_headroom(Pn)

    # ---- submission order: query-first (F190) -----------------------------------------------
    # Rows that share a query share their longest body prefix, so submitting them adjacently
    # keeps those blocks resident instead of letting an unrelated row evict them in between.
    # This changes NO TOKEN -- only the order rows arrive in -- so it cannot move the score, and
    # it makes the pass eviction-proof (flat at 148.98 tokens/row from a 400-block cap to
    # infinity).  F170's "row ordering is worth exactly 0.00%" holds only at INFINITE cache; at
    # every real pool size it is worth -5.8 to -6.7 s.  On by default for that reason.
    # The results come back in SUBMISSION order and are scattered back below; the mock engine is
    # keyed by row_id, so a wrong scatter fails the self-test loudly rather than silently.
    if args.sort_by_query:
        qs = [str(r.get('query') or '') for r in recs]
        sub = sorted(range(len(recs)), key=lambda i: (qs[i], i))     # stable, deterministic
    else:
        sub = list(range(len(recs)))
    qall = [str(r.get('query') or '') for r in recs]
    from collections import Counter
    qc = Counter(qall)
    timings.setdefault('_order', {})[stage] = {
        'sorted_by_query': bool(args.sort_by_query), 'rows': len(recs),
        'distinct_queries': len(qc),
        'rows_sharing_a_query': int(sum(1 for q in qall if qc[q] > 1)),
        'largest_group': int(max(qc.values()) if qc else 0)}

    if getattr(engine, 'set_rows', None):
        engine.set_rows([row_ids[i] for i in sub])
    if getattr(engine, 'llm', None) is not None:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=ids_list[i]) for i in sub]
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=args.logprobs)
    else:
        prompts, sp = [ids_list[i] for i in sub], None

    t0 = time.time()
    outs = engine.generate(prompts, sp)
    wall = time.time() - t0
    timings['%s_generate_s' % stage] = round(wall, 3)

    LGs, covs = read_out(outs, id2cls, len(recs))
    # scatter submission order back to record order
    LG = np.zeros_like(LGs); cov = np.zeros_like(covs)
    for j, i in enumerate(sub):
        LG[i] = LGs[j]; cov[i] = covs[j]
    # FRAGILITY 6 -- the read position moves silently.  A different tokenizer, a truncated prompt
    # or a changed chat template turns the 3-vector into noise while the run still "succeeds".
    # Every model in the recorded ladder scored label mass 0.9982-1.0000.
    assert cov.mean() > 0.5, 'read position wrong: label mass %.4f' % cov.mean()
    if cov.mean() < 0.99:
        print('[L38] WARNING: label mass %.4f -- every recorded model scored 0.998-1.000. The '
              'read-out is losing mass outside the label set.' % cov.mean(), flush=True)
    stats = {'rows': len(recs), 'wall_s': round(wall, 2),
             'rows_per_s': round(len(recs) / wall, 2) if wall > 0 else None,
             'est_full_split_s': round(N_total / (len(recs) / wall), 1) if wall > 0 else None,
             'shared_prefix_tok': int(Pn),
             'prompt_tokens_mean': round(float(np.mean(plens)), 1),
             'prompt_tokens_max': int(max(plens)),
             'label_mass_mean': round(float(cov.mean()), 4),
             'engine_boot_s': round(engine.boot_s, 2) if engine.boot_s is not None else None,
             'attention_impl': getattr(engine, 'attn_impl', None)}
    return RO.norm_log(LG), cov, stats


# --------------------------------------------------------------------------------- the router
def resolve_router(args, df, gate_log, timings, manifest):
    u"""FRAGILITY 3 -- the router's inputs are not all here, and it must not pretend they are.

    The learned router's label is `y = (gate argmax != champion argmax)`, which at inference does
    not exist yet.  Its features read a `tyonly_cal` encoder (and, in the feature set that was
    actually scored, an `s1_ty` encoder too) whose CHECKPOINTS are not in this repository -- only
    their `public_test` outputs are.  Resolution order, each step recorded in the manifest:

        1. --router-scores   a score vector, aligned by position          (repro / anchor)
        2. --router-model    a fitted bundle + encoder matrices           (deployment)
        3. --router-fit-oof  fit in-process against champion logits for
                             EVERY row -- only possible where those
                             already exist, i.e. never at real inference  (anchor only)
        4. gate margin       no features, no fit, always available.  AUC 0.8184 against the
                             learned router's 0.8892; about -0.021 macro-F1 at 30% routing.
    """
    n = len(df)
    if args.router_scores:
        if args.router_scores.endswith('.npy'):
            s = np.load(args.router_scores)
        else:
            d = pd.read_csv(args.router_scores)
            s = d.set_index('row_id').loc[df.row_id.values, d.columns[1]].values.astype(float)
        assert len(s) == n, 'router scores are %d long, input is %d rows' % (len(s), n)
        manifest['router'] = {'source': 'precomputed', 'path': args.router_scores,
                              'sha256': sha256(args.router_scores), 'feature_cost_s': 0.0,
                              'note': 'aligned by POSITION for .npy; verify the input row order'}
        return np.asarray(s, dtype=float)

    spec = RT.FEATURE_SETS[args.router_features]
    enc, enc_paths = {}, {}
    for k, opt in (('e1', args.enc_e1), ('e2', args.enc_e2)):
        if k in spec['need'] and opt and os.path.exists(opt):
            with Timer(timings, 'router_enc_%s_load_s' % k):
                M = np.load(opt)
                if M.shape[0] != n:
                    raise RuntimeError('%s has %d rows, input has %d' % (opt, M.shape[0], n))
                enc[k] = RT.enc_from_probs(M) if float(M.min()) >= 0 else RO.norm_log(M)
            enc_paths[k] = {'path': opt, 'sha256': sha256(opt)}
        elif k in spec['need']:
            print('[L38] WARNING: encoder matrix for block %s not available (%r)' % (k, opt),
                  flush=True)

    have_all = all(k in enc for k in spec['need'])
    if args.router_features != 'margin' and have_all and (args.router_model or args.router_fit_oof):
        with Timer(timings, 'router_features_s'):
            X = RT.select_columns(RT.build_features(df, gate_log, enc), args.router_features)
        base = {'feature_set': args.router_features, 'feature_cost_s': spec['K'],
                'feature_cost_note': ('the set filed as "e1 / 5.5 s" also reads the SECOND '
                                      'encoder through its eb_* columns; its true cost is 21.2 s'
                                      if args.router_features in ('shipped', 'e1e2') else ''),
                'encoders': enc_paths, 'columns': int(X.shape[1])}
        if args.router_model:
            import joblib
            b = joblib.load(args.router_model)
            X = X[b['columns']]                      # positional model: column order is load-bearing
            s = RT.apply_models(b['models'], X.values)
            base.update(source='fitted_model', path=args.router_model,
                        sha256=sha256(args.router_model))
            manifest['router'] = base
            return s
        champ_all = np.load(args.router_fit_oof)
        assert champ_all.shape == (n, 3), 'the OOF fit needs champion logits for every row'
        y = (RO.norm_log(gate_log).argmax(1) != RO.norm_log(champ_all).argmax(1)).astype(int)
        with Timer(timings, 'router_fit_s'):
            s = RT.fit_oof_scores(X.values, y)
        base.update(source='oof_fit_ANCHOR_ONLY', champion_logits=args.router_fit_oof,
                    disagreements=int(y.sum()),
                    warning='fitted against the champion\'s answer for every row; this reproduces '
                            'a recorded measurement and is NOT a deployable configuration')
        manifest['router'] = base
        return s

    why = ('--router-features margin' if args.router_features == 'margin'
           else 'neither --router-model nor --router-fit-oof was given' if have_all
           else 'encoder matrices missing for %r' % (tuple(k for k in spec['need']
                                                           if k not in enc),))
    print('[L38] WARNING: GATE MARGIN routing (%s). Measured AUC 0.8184 vs the learned router\'s '
          '0.8892; worth about -0.021 macro-F1 at 30%% routing.' % why, flush=True)
    manifest['router'] = {'source': 'gate_margin_FALLBACK', 'reason': why, 'feature_cost_s': 0.0}
    return RT.margin_score(gate_log)


# ---------------------------------------------------------------------------------------- main
def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='public_test.csv or .parquet')
    ap.add_argument('--out', default='preds.csv')
    ap.add_argument('--outdir', default=None, help='artifact dir (default: next to --out)')
    ap.add_argument('--config', default=DEFAULT_CONFIG, choices=sorted(CONFIGS))
    ap.add_argument('--gate', default=None, choices=['gate_c3p_t', 'gate_moe', 'none'],
                    help='override the config\'s gate. Which gate wins is UNDECIDED: priced on '
                         'one box c3p_t is 102.5 s and the MoE 104.2 s, and at 20%% routing '
                         'c3p_t scores 0.8987 against the MoE\'s 0.8767 -- but that is an audit '
                         'of two cross-box conversions, not one measurement.')
    ap.add_argument('--champion', default=None,
                    choices=['champ_int8_fi', 'champ_int8_triton', 'champ_fp8'])
    ap.add_argument('--frac', type=float, default=None, help='override the routed fraction')
    ap.add_argument('--readout', default='ipf', choices=['ipf', 'argmax', 'pin'])
    ap.add_argument('--lam', type=float, default=0.5,
                    help='IPF target softness: 1.0 = hard N/3, 0.0 = the model\'s own counts')
    ap.add_argument('--pin', action='store_true', help='alias for --readout pin')
    ap.add_argument('--argmax', action='store_true', help='alias for --readout argmax')
    ap.add_argument('--gate-model', default=None)
    ap.add_argument('--champ-model', default=None)
    ap.add_argument('--gpu-util', type=float, default=None)
    ap.add_argument('--max-model-len', type=int, default=5250)
    ap.add_argument('--max-attr', type=int, default=220)
    ap.add_argument('--logprobs', type=int, default=48)
    ap.add_argument('--boot-timeout', type=int, default=1800)
    ap.add_argument('--sort-by-query', dest='sort_by_query', action='store_true', default=True,
                    help='submit rows grouped by query (default ON). Token-identical, so it '
                         'cannot move the score; worth -5.8 s (-2.2%%) at the A100\'s real pool '
                         'and it makes the pass eviction-proof.')
    ap.add_argument('--no-sort-by-query', dest='sort_by_query', action='store_false',
                    help='submit in input order -- for an A/B of the ordering lever only')
    ap.add_argument('--prefix-align', default='pad', choices=['pad', 'assert', 'off'],
                    help='pad: extend the shared prefix with inert whitespace until the first '
                         'body block starts on a cache-block boundary (-16.8 s / -6.2%%, DERIVED '
                         'from a model, not yet verified on a GPU). assert: measure and report '
                         'the alignment without changing the prompt. off: neither.')
    ap.add_argument('--allow-attention-fallback', action='store_true')
    ap.add_argument('--shots', default=os.path.join(HERE, 'assets', 'shots_gfs4.txt'))
    ap.add_argument('--train', default=None, help='train.parquet, to rebuild and verify the shots')
    ap.add_argument('--router-features', default='shipped', choices=sorted(RT.FEATURE_SETS))
    ap.add_argument('--router-scores', default=None)
    ap.add_argument('--router-model', default=None)
    ap.add_argument('--router-fit-oof', default=None, metavar='CHAMPION_LOGITS.npy')
    ap.add_argument('--enc-e1', default=None, help='tyonly_cal (N,3) probability matrix')
    ap.add_argument('--enc-e2', default=None, help='s1_ty (N,3) probability matrix')
    ap.add_argument('--mock', action='store_true', help='CPU end-to-end with replayed matrices')
    ap.add_argument('--mock-gate', default=None)
    ap.add_argument('--mock-champ', default=None)
    ap.add_argument('--resume', action='store_true',
                    help='reuse stage matrices already on disk rather than re-running an engine')
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.pin:
        args.readout = 'pin'
    if args.argmax:
        args.readout = 'argmax'

    outdir = args.outdir or (os.path.dirname(os.path.abspath(args.out)) or '.')
    os.makedirs(outdir, exist_ok=True)
    timings = {}
    manifest = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'argv': sys.argv[1:], 'config': args.config, 'mock': bool(args.mock)}
    T0 = time.time()

    gate_profile, champ_profile, default_frac = CONFIGS[args.config]
    if args.gate:
        gate_profile = None if args.gate == 'none' else args.gate
    if args.champion:
        champ_profile = args.champion
    frac = default_frac if args.frac is None else args.frac
    if gate_profile is None:
        frac = 1.0
    manifest.update(gate_profile=gate_profile, champion_profile=champ_profile,
                    routed_fraction=frac, readout=args.readout)

    # -------------------------------------------------------------------------------- 1. load
    with Timer(timings, 'load_s'):
        df, missing = load_rows(args.input)
    N = len(df)
    recs = df.to_dict('records')
    manifest['rows'] = N
    manifest['input'] = {'path': args.input, 'sha256': sha256(args.input),
                         'filled_columns': missing}
    print('[L38] %d rows from %s' % (N, args.input), flush=True)

    shot_block, shot_path, shot_sha = load_shot_block(args)
    champ_sys = PR.champion_sysmsg(shot_block)
    manifest['prompt'] = {'champion_spec': 'gfs4/champ3', 'shots_path': shot_path,
                          'shots_sha256': shot_sha, 'shots_chars': len(shot_block),
                          'champion_sysmsg_chars': len(champ_sys)}
    gate_spec = PROFILES[gate_profile]['prompt'] if gate_profile else None
    gate_sys = (PR.gate_sysmsg(gate_spec['kind'], shot_block) if gate_profile else None)
    champ_spec = PROFILES[champ_profile]['prompt']

    # ------------------------------------------- 2. how many engines does this configuration need
    same_ckpt = bool(gate_profile) and PROFILES[gate_profile]['same_checkpoint_as_champion']
    n_engines = 0 if not gate_profile else (1 if same_ckpt else 2)
    manifest['engine_plan'] = {
        'gate_shares_champion_checkpoint': same_ckpt,
        'engine_loads': max(1, n_engines),
        'why': ('gate and champion are the same weights, one engine serves both prompts'
                if same_ckpt else
                'gate and champion are different checkpoints; %.1f + %.1f GiB of weights do not '
                'co-reside in 40 GiB, so the engine is torn down and reloaded'
                % (PROFILES[gate_profile]['weights_gib'] if gate_profile else 0.0,
                   PROFILES[champ_profile]['weights_gib']) if gate_profile else
                'no gate: one engine')}

    # A default that quietly downloads a second 29 GB checkpoint is a footgun, and the thing it
    # buys is small: measured on the full public labels, `moe30` scores 0.9137 against
    # `champ-only`'s 0.9132 -- a 0.0005 gap, well inside the MoE gate's own +-0.01 replicate
    # slack. The cascade's real value is wall clock (~25% at 7,500 rows, ~29% at 90,000), not
    # score. So say so, once, before spending anything.
    if (gate_profile and not same_ckpt and not args.mock and not args.gate_model
            and not os.path.isdir(str(PROFILES[gate_profile]['model']))):
        print('[L38] NOTE: --config %s needs a SECOND checkpoint (%s, ~29 GB) and will fetch it. '
              'It buys +0.0005 macro-F1 over --config champ-only (0.9137 vs 0.9132, inside the '
              'gate\'s own replicate slack) and about 25%% of the wall clock. If you want the '
              'wall clock, continue; if you want the score, --config champ-only is one engine, '
              'one checkpoint and nothing that can silently degrade.'
              % (args.config, PROFILES[gate_profile]['model']), flush=True)

    # FRAGILITY 2 -- OOM, raised at the right moment.  Refusing here costs a second; refusing
    # during vLLM's memory profiling costs a boot; refusing during generate costs the run.
    try:
        import torch
        tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        manifest['gpu'] = {'name': torch.cuda.get_device_name(0), 'total_gib': round(tot, 1)}
        need = max(PROFILES[p]['weights_gib'] for p in (gate_profile, champ_profile) if p)
        if tot < need + 3.0:
            raise SystemExit('[L38] REFUSING: largest checkpoint needs %.1f GiB of weights and '
                             'this card has %.1f GiB total.' % (need, tot))
        if n_engines == 2:
            both = PROFILES[gate_profile]['weights_gib'] + PROFILES[champ_profile]['weights_gib']
            print('[L38] %.2f GiB of weights for two checkpoints on a %.0f GiB card -> '
                  'SEQUENTIAL load, TWO engine boots. Measured eager: gate 63.4 s first-on-box / '
                  '19.5 s warm, champion 23.6 / 22.2 s. Co-residence was tried and gives '
                  '"Available KV cache memory: -21.61 GiB".' % (both, tot), flush=True)
    except SystemExit:
        raise
    except Exception:
        manifest['gpu'] = None
        if not args.mock:
            print('[L38] WARNING: no CUDA device visible; this will fail at engine boot.',
                  flush=True)

    mock_tok = None
    if args.mock:
        # The mock tokenizer is NOT gemma's, so its token counts are not comparable and the real
        # 5,250 budget would fire spuriously.  The real numbers, for the record: 5,096.8 mean and
        # 5,178 max prompt tokens under gemma-4's tokenizer, i.e. 1.4% of headroom at mml 5250.
        if '--max-model-len' not in sys.argv:
            args.max_model_len = 32768
            print('[L38] mock: max_model_len relaxed to %d because the mock tokenizer is not '
                  "gemma's. The REAL budget is 5,250 for a 5,178-token longest prompt."
                  % args.max_model_len, flush=True)
        words = set()
        for c in ('query', 'title', 'category', 'brand', 'attributes'):
            for v in df[c].astype(str):
                for w in v.replace('/', ' ').replace(',', ' ').replace(':', ' ').lower().split():
                    if w.isalnum():
                        words.add(w)
        for w in (champ_sys + ' ' + (gate_sys or '')).replace('\n', ' ').lower().split():
            if w.isalnum():
                words.add(w)
        # the three label words must NOT be single vocabulary pieces, or the mock's read position
        # stops looking like gemma's (`ir` + `relevant`, `semi` + `-` + `relevant`)
        words -= set(LABELS) | {'irrelevant', 'relevant', 'semi'}
        mock_tok = MockTokenizer(sorted(words))

    def make_engine(profile, mock_matrix):
        if args.mock:
            rep = {rid: v for rid, v in zip(df.row_id.values,
                                            RO.norm_log(np.load(mock_matrix)))}
            return MockJudge(profile, rep, mock_tok, logprobs=args.logprobs)
        ov = {}
        if args.gpu_util is not None:
            ov['gpu_memory_utilization'] = args.gpu_util
        if args.max_model_len:
            ov['max_model_len'] = args.max_model_len
        if args.allow_attention_fallback:
            ov['expect_impl'] = ()
            ov['attention_backend'] = None   # N4: else vLLM refuses before the check
            ov['attention_backend'] = None   # N4: else vLLM refuses before the check
        model = (args.gate_model if profile == gate_profile else args.champ_model)
        if PROFILES[profile]['model'] is None:                # c3p_t inherits the champion's
            model = model or args.champ_model or PROFILES[champ_profile]['model']
        return VllmJudge(profile, model=model, overrides=ov, logprobs=args.logprobs)

    pcache = PR.PromptCache()
    gate_npy = os.path.join(outdir, 'logits_gate_%s_log.npy' % (gate_profile or 'none'))
    champ_npy = os.path.join(outdir, 'logits_champ_%s_log.npy' % champ_profile)

    # ------------------------------------------------- 3. gate pass (skipped when there is no gate)
    GATE = None
    engine = None
    if gate_profile and frac < 1.0:
        if args.resume and os.path.exists(gate_npy):
            GATE = RO.norm_log(np.load(gate_npy))
            print('[L38] resume: reusing %s' % gate_npy, flush=True)
            timings['gate_boot_s'] = 0.0
        else:
            # when the gate shares the champion's checkpoint the engine is created ONCE here and
            # reused below; that is the whole point of that gate.
            engine = make_engine(gate_profile if not same_ckpt else champ_profile, args.mock_gate)
            with Timer(timings, 'gate_boot_s'):
                _with_timeout(args.boot_timeout, engine.boot, 'gate engine')
            GATE, _, gstats = judge(engine, recs, gate_sys, gate_spec, args, 'gate', timings, N,
                                    cache=pcache)
            np.save(gate_npy, GATE.astype(np.float32))
            manifest['gate'] = gstats
            if not same_ckpt:
                with Timer(timings, 'gate_shutdown_s'):
                    engine.shutdown()
                engine = None

    # ------------------------------------------------------------------------------ 4. routing
    if frac >= 1.0:
        sel, k = np.ones(N, bool), N
        manifest.setdefault('router', {'source': 'none (no cascade)', 'feature_cost_s': 0.0})
    elif frac <= 0.0:
        sel, k = np.zeros(N, bool), 0
        manifest.setdefault('router', {'source': 'none (gate only)', 'feature_cost_s': 0.0})
    else:
        score = resolve_router(args, df, GATE, timings, manifest)
        sel, k = RT.select_rows(score, frac, N)
        np.save(os.path.join(outdir, 'router_score.npy'), np.asarray(score))
    print('[L38] routing %d of %d rows (%.2f%%) to the champion' % (k, N, 100.0 * k / max(N, 1)),
          flush=True)
    manifest['routed_rows'] = int(k)

    # ---------------------------------------------------------------------------- 5. champion
    CH = None
    if k > 0:
        if args.resume and os.path.exists(champ_npy):
            full = np.load(champ_npy)
            CH = RO.norm_log(full[sel] if full.shape[0] == N else full)
            print('[L38] resume: reusing %s' % champ_npy, flush=True)
            timings['champ_boot_s'] = 0.0
        else:
            sub = [r for r, s in zip(recs, sel) if s]
            if engine is None:
                engine = make_engine(champ_profile, args.mock_champ)
                with Timer(timings, 'champ_boot_s'):
                    _with_timeout(args.boot_timeout, engine.boot, 'champion engine')
            else:
                timings['champ_boot_s'] = 0.0        # shared with the gate: no second boot
                if args.mock:
                    engine.replay = {rid: v for rid, v in zip(
                        df.row_id.values, RO.norm_log(np.load(args.mock_champ)))}
            CH, _, cstats = judge(engine, sub, champ_sys, champ_spec, args, 'champ', timings,
                                  N, cache=pcache)
            manifest['champion'] = cstats
            full = np.full((N, 3), np.nan, dtype=np.float32)
            full[sel] = CH
            np.save(champ_npy, full)
        if engine is not None:
            with Timer(timings, 'champ_shutdown_s'):
                engine.shutdown()
            engine = None

    # ----------------------------------------------------------------------------- 6. compose
    # FRAGILITY 5 -- row order.  Everything here is positional, so the scatter is asserted.
    if frac >= 1.0:
        P = np.exp(CH)
    else:
        P = np.exp(GATE).copy()
        if k > 0:
            P[sel] = np.exp(CH)
    assert P.shape == (N, 3), 'composed matrix is %r, expected (%d, 3)' % (P.shape, N)
    assert np.isfinite(P).all(), 'non-finite probabilities in the composed matrix'
    P = P / P.sum(1, keepdims=True)
    np.save(os.path.join(outdir, 'composed_probs.npy'), P.astype(np.float32))

    # ----------------------------------------------------------------------------- 7. read-out
    if args.readout == 'pin' and 0 < k < N:
        print('[L38] WARNING: --pin over a MIXED-SOURCE matrix. Measured cost against IPF: up to '
              '0.016 macro-F1, because 244 of its 248 displacements at 30%% routing land on the '
              'rows the champion was paid for. This is the anchor, not the recommendation.',
              flush=True)
    with Timer(timings, 'readout_s'):
        pred, info = RO.decide(P, args.readout, lam=args.lam)
    manifest['readout_detail'] = info

    out = pd.DataFrame({'row_id': df.row_id.values, 'real_label': [LABELS[i] for i in pred]})
    assert list(out.row_id.values) == list(df.row_id.values), 'output row order drifted'
    out.to_csv(args.out, index=False)

    # -------------------------------------------------------------------------- 8. accounting
    timings['total_s'] = round(time.time() - T0, 3)
    boots = sum(timings.get(k_, 0.0) or 0.0 for k_ in ('gate_boot_s', 'champ_boot_s'))
    gen = sum(timings.get(k_, 0.0) or 0.0 for k_ in ('gate_generate_s', 'champ_generate_s'))
    manifest['timings_s'] = timings
    manifest['prefix_alignment'] = timings.pop('_align', {})
    manifest['submission_order'] = timings.pop('_order', {})
    manifest['prompt_cache'] = {'hits': pcache.hits, 'misses': pcache.misses,
                                'note': 'a HIT means the champion reused the gate\'s token ids. '
                                        'It happens when the gate wears the champion prompt and '
                                        'the two checkpoints share a tokenizer, which the gemma-4 '
                                        'family does. L33 measured tokenisation at 31.7 s of a '
                                        '177 s sequential cascade; a hit removes half of it.'}
    manifest['accounting'] = {
        'engine_boots_s': round(boots, 1),
        'engine_teardown_s': round(sum(timings.get(k_, 0.0) or 0.0
                                       for k_ in ('gate_shutdown_s', 'champ_shutdown_s')), 1),
        'prompt_tokenisation_s': round(sum(timings.get(k_, 0.0) or 0.0
                                           for k_ in ('gate_prompt_build_s',
                                                      'champ_prompt_build_s')), 1),
        'import_s': round(IMPORT_S, 1),
        'engine_loads': manifest['engine_plan']['engine_loads'],
        'generate_s': round(gen, 1),
        'router_feature_cost_s_charged': manifest.get('router', {}).get('feature_cost_s', 0.0),
        'everything_else_s': round(timings['total_s'] - boots - gen, 1),
        'total_s': timings['total_s'],
        'note': 'engine_boots_s is the line no recorded table contains. Measured eager on an '
                'A100 40GB: gate 63.4 s first-on-box / 19.5 s warm, champion 23.6 / 22.2 s. '
                'Router feature cost '
                'is CHARGED here but not spent when the encoder outputs are precomputed -- read '
                'RUNBOOK before quoting a total.'}
    manifest['counts'] = {LABELS[i]: int(c) for i, c in enumerate(np.bincount(pred, minlength=3))}
    with open(os.path.join(outdir, 'run_manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)

    print('[L38] wrote %s   counts %r' % (args.out, manifest['counts']), flush=True)
    print('[L38] %d engine load(s): boots %.1fs + generate %.1fs + rest %.1fs = %.1fs total'
          % (manifest['engine_plan']['engine_loads'], boots, gen,
             manifest['accounting']['everything_else_s'], timings['total_s']), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
