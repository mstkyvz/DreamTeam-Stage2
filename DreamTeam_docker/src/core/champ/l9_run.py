#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L9 -- multi-variant driver: load the 31B ONCE, sweep prompts x splits.

Engine start-up (weights + torch.compile + CUDA-graph capture) costs ~4 minutes on this box and
is completely independent of the prompt, so running one variant per process would spend more
wall-clock on start-up than on the actual judging. Everything shares one `LLM` object; the
per-variant numbers reported in `timing_L9_*.json` are the GENERATE call only, which is the
number the deliverable's "<3 minutes for 7,500 rows" claim is about.

Read-out and label-token discipline are identical to l9_judge.py -- see that file's header.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l9_judge import (LABELS, DATA, ANSWER_PREFIX, SENTINEL, _clean, render_item,
                      system_for, build_label_token_sets, assert_label_tokens, logsumexp)


def macro_f1(y_true, y_pred, labels=LABELS):
    f = []
    for l in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == l and p == l)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != l and p == l)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == l and p != l)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f)), [round(x, 4) for x in f]


def balance(p, target=(1/3, 1/3, 1/3), iters=200):
    """infra/calibrate.py's iterative proportional fitting, inlined so a run is self-contained."""
    t = np.array(target) * len(p); w = np.ones(p.shape[1])
    for _ in range(iters):
        cnt = np.bincount((p * w).argmax(1), minlength=p.shape[1]).astype(float)
        if np.abs(cnt - t).max() <= 2: break
        w *= (t / np.maximum(cnt, 1.0)) ** 0.25
    q = p * w
    return q / q.sum(1, keepdims=True), w


def build_prompts(tok, TokensPrompt, recs, variant, order, max_attr):
    import prompts_l3 as P
    sysmsg = system_for(variant, order)
    full_user = sysmsg + "\n\n" + P.SIRADAKI + "\n\n" + SENTINEL
    templated = tok.apply_chat_template([{'role': 'user', 'content': full_user}],
                                        tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX
    add_special = not (tok.bos_token and templated.startswith(tok.bos_token))
    pre_text, post_text = templated.split(SENTINEL)
    enc = lambda s: tok.encode(s, add_special_tokens=add_special)
    prefix_ids = enc(pre_text); Pn = len(prefix_ids)
    prompts, tails = [], []
    for r in recs:
        ids = enc(pre_text + "SORGU: %s\n%s" % (_clean(r.get('query')),
                                                render_item(r, max_attr)) + post_text)
        assert ids[:Pn] == prefix_ids, 'tokenizer merged across the sentinel boundary'
        tails.append(len(ids) - Pn)
        prompts.append(TokensPrompt(prompt_token_ids=ids))
    return prompts, Pn, float(np.mean(tails)), max(tails)


def read_out(outs, id2cls, N):
    LG = np.zeros((N, 3), dtype=np.float64); covered = np.zeros(N)
    for i, o in enumerate(outs):
        lp = o.outputs[0].logprobs[0]
        v = np.full(3, -np.inf)
        for tid, L in lp.items():
            k = id2cls.get(int(tid))
            if k is not None:
                v[k] = np.logaddexp(v[k], L.logprob)
        covered[i] = float(np.exp(logsumexp(v)))
        LG[i] = v - logsumexp(v)
    return np.nan_to_num(LG, neginf=-200.0), covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--variants', default='hardfs,fs,hard,sh,zs')
    ap.add_argument('--splits', default='public_test.parquet')
    ap.add_argument('--order', default='label')
    ap.add_argument('--prefix', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--outdir', default='/root/L9/out')
    ap.add_argument('--max-attr', dest='max_attr', type=int, default=220)
    ap.add_argument('--logprobs', type=int, default=48)
    ap.add_argument('--max-model-len', dest='max_model_len', type=int, default=3072)
    ap.add_argument('--gpu-util', dest='gpu_util', type=float, default=0.90)
    ap.add_argument('--max-num-seqs', dest='max_num_seqs', type=int, default=512)
    ap.add_argument('--max-batched-tokens', dest='max_batched_tokens', type=int, default=32768)
    ap.add_argument('--tp', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    t_boot = time.time()
    llm = LLM(model=args.model, tokenizer=args.model, dtype='auto',
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=True, max_logprobs=args.logprobs + 8,
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_batched_tokens,
              tensor_parallel_size=args.tp, seed=0, disable_log_stats=True)
    print('[L9] engine up in %.1fs' % (time.time() - t_boot), flush=True)
    tok = llm.get_tokenizer()
    idx, cls = build_label_token_sets(tok)
    assert_label_tokens(tok, idx, cls)
    id2cls = dict(zip(idx.tolist(), cls.tolist()))
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=args.logprobs)

    summary = []
    for split in [s for s in args.splits.split(',') if s]:
        path = split if os.path.isabs(split) else os.path.join(DATA, split)
        df = pd.read_parquet(path)
        if args.limit:
            df = df.head(args.limit).reset_index(drop=True)
        recs = df.to_dict('records'); N = len(df)
        has_gold = 'real_label' in df.columns
        sname = os.path.basename(split).replace('.parquet', '')
        for variant in [v for v in args.variants.split(',') if v]:
            tag = '%s%s' % (args.prefix, variant) if sname.startswith('public_test') \
                  else '%s%s_%s' % (args.prefix, variant, sname)
            prompts, Pn, tmean, tmax = build_prompts(tok, TokensPrompt, recs, variant,
                                                     args.order, args.max_attr)
            t0 = time.time(); outs = llm.generate(prompts, sp); wall = time.time() - t0
            t_r = time.time(); LG, covered = read_out(outs, id2cls, N)
            readout = time.time() - t_r
            assert covered.mean() > 0.5, 'read position wrong for %s' % tag
            Pm = np.exp(LG - LG.max(1, keepdims=True))
            Pm = (Pm / Pm.sum(1, keepdims=True)).astype(np.float32)
            LAB = [LABELS[k] for k in Pm.argmax(1)]
            Qb, _ = balance(Pm.astype(np.float64))
            LABb = [LABELS[k] for k in Qb.argmax(1)]
            rec = {'tag': tag, 'split': sname, 'variant': variant, 'rows': N,
                   'wall_s': round(wall, 2), 'rows_per_s': round(N / wall, 2),
                   'est_7500_s': round(7500.0 / (N / wall), 1),
                   # the generate call is what gets quoted; the Python logprob read-out is a
                   # separate, real cost and is recorded rather than hidden
                   'readout_s': round(readout, 2),
                   'est_7500_s_total': round((wall + readout) * 7500.0 / N, 1),
                   'prefix_tokens': Pn,
                   'tail_mean_tokens': round(tmean, 1), 'tail_max_tokens': tmax,
                   'label_mass_mean': round(float(covered.mean()), 4),
                   'raw_counts': pd.Series(LAB).value_counts().to_dict()}
            if has_gold:
                gt = df.real_label.tolist()
                # Score over the labels the split ACTUALLY contains. val_grp_L9 is two-class by
                # construction (no real `irrelevant` source exists for a train query), and the
                # grouped arm scores it the same way -- averaging in a third class that has zero
                # true rows would make the pointwise-vs-grouped comparison meaningless.
                labs = sorted(set(gt))
                rec['classes'] = labs
                rec['macroF1_raw'], rec['perclass_raw'] = macro_f1(gt, LAB, labs)
                rec['macroF1_bal'], rec['perclass_bal'] = macro_f1(gt, LABb, labs)
                if 'src' in df.columns:
                    rec['acc_by_src'] = {s: round(float(np.mean(
                        [p == t for p, t, ss in zip(LABb, gt, df.src) if ss == s])), 4)
                        for s in sorted(df.src.unique())}
            np.save('%s/logits_L9_%s.npy' % (args.outdir, tag), Pm)
            np.save('%s/logits_L9_%s_log.npy' % (args.outdir, tag), LG.astype(np.float32))
            idcol = 'row_id' if 'row_id' in df.columns else df.columns[0]
            pd.DataFrame({'row_id': df[idcol], 'real_label': LAB}).to_csv(
                '%s/preds_public_test_L9_%s.csv' % (args.outdir, tag), index=False)
            pd.DataFrame({'row_id': df[idcol], 'real_label': LABb}).to_csv(
                '%s/preds_public_test_L9_%scal.csv' % (args.outdir, tag), index=False)
            json.dump(rec, open('%s/timing_L9_%s.json' % (args.outdir, tag), 'w'), indent=2)
            summary.append(rec)
            print('[L9] %-28s %5d rows  %6.1fs (+%.1fs read)  %6.1f r/s  7500->%6.1fs  %s'
                  % (tag, N, wall, readout, N / wall, rec['est_7500_s_total'],
                     ('macroF1 raw %.4f bal %.4f' % (rec['macroF1_raw'], rec['macroF1_bal']))
                     if has_gold else str(rec['raw_counts'])), flush=True)
    json.dump(summary, open('%s/summary_L9%s.json' % (args.outdir,
              ('_' + args.prefix.strip('_')) if args.prefix else ''), 'w'), indent=2)
    print('[L9] DONE', flush=True)


if __name__ == '__main__':
    main()
