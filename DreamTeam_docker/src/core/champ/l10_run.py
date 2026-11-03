#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""L10 -- run the semi-widening rubric ladder through L9's zero-generation read-out.

The read-out, the tokenisation guards, the label-token marginal and the IPF count-balancer are
L9's, imported rather than reimplemented, so the ONLY thing that differs between an L10 arm and
L9's measured arms is the rubric text.  Anything else would confound the measurement the brief
asked for.

One process loads the 31B once and walks the whole ladder; every arm's four output files are
flushed to disk the moment that arm finishes (and a `.done` marker written), because this run has
already lost a completed pass to a box that vanished.  A watcher on the laptop pulls and pushes
on the marker.

Spec grammar:  MODE/RUBRIC[:TAG]
  plain/<r>   -- one shared system message, SYSTEMS[r]
  fs/<r>      -- same, plus L3's ten hand-written shots (only meaningful for `soft`)
  catfs/<r>   -- L9's per-category contrastive few-shot with CATFS_HEADS[r] as the head
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts_l3 as P
import prompts_l10 as PS
from l9_judge import (LABELS, DATA, ANSWER_PREFIX, SENTINEL, _clean, render_item,
                      user_block, shots_text, build_label_token_sets, assert_label_tokens)
from l9_run import macro_f1, balance, read_out
from l9_catfs import catkey, build_cat_shots, shot_json
import prompts_l3 as _P3


def build_global_shots(tr, n_pair=4, n_extra=3, seed=0, order='label'):
    """A GLOBAL few-shot block, drawn from train with a seed.

    Route 2 of the brief: the `hash()` bug made shot selection accidental, and fixing it exposed
    that it was never a chosen quantity at all. `fs` (L3's ten hand-written shots) produced the
    run's best calibration -- L1 158 for `wide3_fs` -- so the shots are doing real work and a
    better draw may do more. This samples CONTRASTIVE pairs (the same train query appearing once
    as `relevant` and once as `semi-relevant`, 7,949 such queries exist) from across the whole
    catalogue rather than per-category, spread over distinct level-2 departments so one seed does
    not draw ten shots from one aisle. The three authored `irrelevant` shots are appended
    unchanged: train.parquet contains no gold `irrelevant` row, so inventing one per seed would be
    manufacturing evidence rather than sampling it.
    """
    rng = np.random.default_rng(20260828 + seed)
    tr = tr.copy()
    tr['_ck'] = tr.category.map(lambda c: catkey(c, 2))
    nq = tr.groupby('query').real_label.nunique()
    both = nq[nq == 2].index
    d = tr[tr['query'].isin(set(both))]
    # one contrastive pair per department, departments shuffled by the seed
    cks = list(d._ck.unique()); rng.shuffle(cks)
    blocks = []
    for ck in cks:
        if len(blocks) >= 2 * n_pair:
            break
        dc = d[d._ck == ck]
        # a contrastive query can straddle departments, so "has both labels" must be re-checked
        # INSIDE the department -- checking it globally is what made the first run IndexError.
        nq_c = dc.groupby('query').real_label.nunique()
        qs = list(nq_c[nq_c == 2].index)
        if not qs:
            continue
        rng.shuffle(qs)
        dq = dc[dc['query'] == qs[0]]
        blocks.append(shot_json(dq[dq.real_label == 'relevant'].iloc[0], 'relevant', order))
        blocks.append(shot_json(dq[dq.real_label == 'semi-relevant'].iloc[0], 'semi-relevant', order))
    assert len(blocks) == 2 * n_pair, 'only %d contrastive shots found' % (len(blocks) // 2)
    # independent top-ups, also seed-dependent
    for lab in ('relevant', 'semi-relevant'):
        dl = tr[tr.real_label == lab]
        take = dl.iloc[rng.choice(len(dl), size=n_extra, replace=False)]
        for _, r in take.iterrows():
            blocks.append(shot_json(r, lab, order))
    authored = []
    for r in _P3.SHOTS_IRRELEVANT:
        ans = ({'etiket': r['etiket'], 'gerekce': r['gerekce']} if order == 'label'
               else {'gerekce': r['gerekce'], 'etiket': r['etiket']})
        authored.append(user_block(r) + "\nCEVAP: " + json.dumps(ans, ensure_ascii=False))
    merged = []
    for i, b in enumerate(blocks):
        merged.append(b)
        if i % 3 == 2 and authored:
            merged.append(authored[(i // 3) % len(authored)])
    for a in authored:
        if a not in merged:
            merged.append(a)
    return "\n\n---\n\n".join(merged)


def sysmsg_for(mode, rubric, order, gshots=None):
    if mode == 'plain':
        return PS.SYSTEMS[rubric](order)
    if mode == 'fs':
        # any rubric on L3's ten shared hand-written shots -- soft_fs was the best-calibrated arm
        # in the run, so this shape is worth having for the later rubrics too
        return u"\n\n".join(list(PS.CATFS_HEADS[rubric]) +
                             [P.ORNEK_BASLIK + "\n\n" + shots_text(order), P.out_block(order)])
    if mode.startswith('gfs') or mode.startswith('gbs'):
        assert gshots is not None
        return u"\n\n".join(list(PS.CATFS_HEADS[rubric]) +
                             [P.ORNEK_BASLIK + "\n\n" + gshots, P.out_block(order)])
    if mode.startswith('mfs'):
        # MERGED shots: L3's ten hand-written demonstrations AND the sampled block, in one pass.
        # Both are good draws on their own (wide3_fs L1 158, wide3_gfs2 L1 114) and they share no
        # rows, so if what a good draw supplies is coverage rather than luck, the union should
        # beat either. Still exactly one forward pass -- only the prompt is longer.
        assert gshots is not None
        return u"\n\n".join(list(PS.CATFS_HEADS[rubric]) +
                             [P.ORNEK_BASLIK + "\n\n" + shots_text(order)
                              + "\n\n---\n\n" + gshots, P.out_block(order)])
    raise ValueError(mode)


def encode_block(tok, sysmsg, recs, max_attr, max_model_len):
    """L9's sentinel trick: prove the shared rubric prefix tokenises identically for every row,
    so vLLM's prefix cache can actually share it, and so the forced read position is the LAST
    token for every prompt."""
    templated = tok.apply_chat_template(
        [{'role': 'user', 'content': sysmsg + "\n\n" + P.SIRADAKI + "\n\n" + SENTINEL}],
        tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX
    add_special = not (tok.bos_token and templated.startswith(tok.bos_token))
    pre_text, post_text = templated.split(SENTINEL)
    enc = lambda s: tok.encode(s, add_special_tokens=add_special)
    prefix_ids = enc(pre_text); Pn = len(prefix_ids)
    ids_list, lens = [], []
    # P3: tokenise the shared prefix ONCE and append the per-row tail (verified bit-identical
    # to the per-row full encode on all 90,000 rows by tools/verify_tok_90k.py).
    enc_tail = lambda s: tok.encode(s, add_special_tokens=False)
    for r in recs:
        ids = prefix_ids + enc_tail("SORGU: %s\n%s" % (_clean(r.get('query')),
                                                       render_item(r, max_attr)) + post_text)
        # NEVER truncate the tail: the label is read at the LAST position, so clipping silently
        # moves the read off the answer prefix and the 3-vector becomes noise.
        assert len(ids) <= max_model_len, ('prompt %d tokens > max_model_len %d'
                                           % (len(ids), max_model_len))
        ids_list.append(ids); lens.append(len(ids))
    return ids_list, Pn, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--specs', required=True,
                    help='comma list of MODE/RUBRIC[:TAG], e.g. plain/l7soft,catfs/wide')
    ap.add_argument('--split', default='public_test.parquet')
    ap.add_argument('--order', default='label')
    ap.add_argument('--outdir', default='/root/L10/out')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-attr', dest='max_attr', type=int, default=220)
    ap.add_argument('--logprobs', type=int, default=48)
    ap.add_argument('--max-model-len', dest='max_model_len', type=int, default=6144)
    ap.add_argument('--gpu-util', dest='gpu_util', type=float, default=0.95)
    ap.add_argument('--max-num-seqs', dest='max_num_seqs', type=int, default=256)
    ap.add_argument('--max-batched-tokens', dest='max_batched_tokens', type=int, default=16384)
    ap.add_argument('--level', type=int, default=2)
    ap.add_argument('--n-pair', dest='n_pair', type=int, default=3)
    ap.add_argument('--n-extra', dest='n_extra', type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    path = args.split if os.path.isabs(args.split) else os.path.join(DATA, args.split)
    df = pd.read_parquet(path)
    if args.limit:
        df = df.head(args.limit)
    df = df.reset_index(drop=True)
    df['_ord'] = np.arange(len(df))
    N = len(df)
    recs = df.to_dict('records')
    idcol = 'row_id' if 'row_id' in df.columns else df.columns[0]
    specs = [s for s in args.specs.split(',') if s]
    needs_cat = any(s.split(':')[0].split('/')[0] == 'catfs' for s in specs)
    # (kind, seed): 'gfs' draws 7 relevant + 7 semi + 3 authored irrelevant; 'gbs' draws 3/3/3.
    # champ_gfs1 (2044 irr) restrained LESS than champ_fs (2257) on an identical rubric, and the
    # only difference is the shot block -- so the shots' own LABEL BALANCE is a calibration dial,
    # and 'gbs' is the direct test of that. train.parquet has no gold `irrelevant` row, so the
    # irrelevant side is capped at L3's three authored shots either way; balancing therefore means
    # taking FEWER of the other two, not more of that one.
    gseeds = sorted({(s.split(':')[0].split('/')[0][:3], int(s.split(':')[0].split('/')[0][3:] or 0))
                     for s in specs
                     if s.split(':')[0].split('/')[0][:3] in ('gfs', 'gbs', 'mfs')})

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    t_load = time.time()
    # gemma-4 is multimodal and vLLM profiles the vision tower with 6 max-size VIDEO items,
    # which on a 46 GiB L40S reserves enough activation memory to leave only 5.7 GiB of KV cache
    # (measured: engine refused to start at max_model_len 10240). Every prompt here is pure text,
    # so the encoder is declared away rather than paid for.
    llm = LLM(model=args.model, tokenizer=args.model, dtype='auto',
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=True, max_logprobs=args.logprobs + 8,
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_batched_tokens,
              limit_mm_per_prompt={'image': 0, 'video': 0},
              seed=0, disable_log_stats=True)
    print('[L10] engine up in %.1fs' % (time.time() - t_load), flush=True)
    tok = llm.get_tokenizer()
    idx, cls = build_label_token_sets(tok)
    assert_label_tokens(tok, idx, cls)
    id2cls = dict(zip(idx.tolist(), cls.tolist()))
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=args.logprobs)

    cat_shots = cat_fallback = None
    gshot_by_seed = {}
    if needs_cat or gseeds:
        tr = pd.read_parquet(os.path.join(DATA, 'train.parquet'))
        for kind, sd in gseeds:
            npair, nextra = ((1, 2) if kind == 'gbs' else (4, 3))
            gshot_by_seed[(kind, sd)] = build_global_shots(tr, n_pair=npair, n_extra=nextra,
                                                           seed=sd, order=args.order)
            print('[L10] %s shot block seed %d: %d chars (%d rel + %d semi + 3 authored irr)'
                  % (kind, sd, len(gshot_by_seed[(kind, sd)]), npair + nextra, npair + nextra),
                  flush=True)
    if needs_cat:
        df['_ck'] = df.category.map(lambda c: catkey(c, args.level))
        t_s = time.time()
        cat_shots, cat_fallback = build_cat_shots(tr, set(df._ck), args.level, args.n_pair,
                                                  args.n_extra, order=args.order)
        print('[L10] %d category shot blocks in %.1fs' % (len(cat_shots), time.time() - t_s),
              flush=True)
    if needs_cat or gseeds:
        del tr

    for spec in specs:
        head, _, tagover = spec.partition(':')
        mode, _, rubric = head.partition('/')
        tag = tagover or ('%s_%s' % (rubric, mode) if mode != 'plain' else rubric)
        gsh = (gshot_by_seed.get((mode[:3], int(mode[3:] or 0)))
               if mode[:3] in ('gfs', 'gbs', 'mfs') else None)

        if mode == 'catfs':
            prompts, order_ix, plens = [], [], []
            blocks = PS.CATFS_HEADS[rubric]
            # category-contiguous emission so vLLM charges each shared block once
            for ck, d in df.groupby('_ck', sort=True):
                sysmsg = u"\n\n".join(list(blocks) +
                                      [P.ORNEK_BASLIK + "\n\n" + cat_shots.get(ck, cat_fallback),
                                       P.out_block(args.order)])
                sub = d.to_dict('records')
                ids_list, Pn, lens = encode_block(tok, sysmsg, sub, args.max_attr,
                                                  args.max_model_len)
                for ids, r in zip(ids_list, sub):
                    prompts.append(TokensPrompt(prompt_token_ids=ids))
                    order_ix.append(int(r['_ord']))
                plens += lens
            print('[L10] %-14s catfs prompts: mean %.0f max %d tok'
                  % (tag, float(np.mean(plens)), max(plens)), flush=True)
        else:
            sysmsg = sysmsg_for(mode, rubric, args.order, gsh)
            ids_list, Pn, plens = encode_block(tok, sysmsg, recs, args.max_attr,
                                               args.max_model_len)
            prompts = [TokensPrompt(prompt_token_ids=i) for i in ids_list]
            order_ix = list(range(N))
            print('[L10] %-14s shared prefix %d tok, mean %.0f max %d'
                  % (tag, Pn, float(np.mean(plens)), max(plens)), flush=True)

        t0 = time.time(); outs = llm.generate(prompts, sp); wall = time.time() - t0
        LGs, cov0 = read_out(outs, id2cls, len(prompts))
        LG = np.zeros((N, 3)); cov = np.zeros(N)
        for j, oi in enumerate(order_ix):
            LG[oi] = LGs[j]; cov[oi] = cov0[j]
        assert cov.mean() > 0.5, 'read position wrong (label mass %.3f)' % cov.mean()
        Pm = np.exp(LG - LG.max(1, keepdims=True))
        Pm = (Pm / Pm.sum(1, keepdims=True)).astype(np.float32)
        LAB = [LABELS[k] for k in Pm.argmax(1)]
        Qb, w = balance(Pm.astype(np.float64))
        LABb = [LABELS[k] for k in Qb.argmax(1)]
        raw = pd.Series(LAB).value_counts().to_dict()
        rec = {'tag': tag, 'spec': spec, 'mode': mode, 'rubric': rubric, 'rows': N,
               'wall_s': round(wall, 2), 'rows_per_s': round(N / wall, 2),
               'est_7500_s': round(7500.0 / (N / wall), 1),
               'prompt_tokens_mean': round(float(np.mean(plens)), 1),
               'label_mass_mean': round(float(cov.mean()), 4),
               'raw_counts': {k: int(raw.get(k, 0)) for k in LABELS},
               'balance_weights': [round(float(x), 4) for x in w]}
        if 'real_label' in df.columns:
            gt = df.real_label.tolist(); labs = sorted(set(gt))
            rec['macroF1_raw'], rec['perclass_raw'] = macro_f1(gt, LAB, labs)
            rec['macroF1_bal'], rec['perclass_bal'] = macro_f1(gt, LABb, labs)

        o = args.outdir
        np.save('%s/logits_L10_%s_log.npy' % (o, tag), LG.astype(np.float32))
        np.save('%s/logits_L10_%s.npy' % (o, tag), Pm)
        pd.DataFrame({'row_id': df[idcol], 'real_label': LAB}).to_csv(
            '%s/preds_public_test_L10_%s.csv' % (o, tag), index=False)
        pd.DataFrame({'row_id': df[idcol], 'real_label': LABb}).to_csv(
            '%s/preds_public_test_L10_%scal.csv' % (o, tag), index=False)
        json.dump(rec, open('%s/timing_L10_%s.json' % (o, tag), 'w'), indent=2)
        open('%s/%s.done' % (o, tag), 'w').write(json.dumps(rec))
        print('[L10] DONE %-14s %5d rows %6.1fs (%.1f r/s, 7500->%.1fs) raw irr/rel/semi '
              '%d/%d/%d  mass %.4f' % (tag, N, wall, N / wall, rec['est_7500_s'],
                                       rec['raw_counts']['irrelevant'],
                                       rec['raw_counts']['relevant'],
                                       rec['raw_counts']['semi-relevant'],
                                       rec['label_mass_mean']), flush=True)
    print('[L10] ALL DONE', flush=True)


if __name__ == '__main__':
    main()
