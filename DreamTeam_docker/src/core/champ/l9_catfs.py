#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L9 -- PER-CATEGORY few-shot: the shots are drawn from the target row's own department.

WHY.  The shared few-shot block every arm has used so far is ten hand-picked rows covering the
whole catalogue -- three of them authored by hand because train.parquet has no gold `irrelevant`.
It has to teach "what does semi-relevant mean" using a helmet, a detergent and a cutlery set,
and then that block is shown to a row about phone cases. But train.parquet holds 250,000 gold
`relevant` and 250,000 gold `semi-relevant` rows, and 99.9% of test rows sit in a level-2
category that has at least 40 gold train rows of its own. The demonstrations can simply BE from
the department being judged.

The strongest shots available are CONTRASTIVE: the same query appearing once as `relevant` and
once as `semi-relevant`. 7,949 train queries carry both labels, so those pairs are picked first
and show the boundary rather than the classes.

WHY IT IS STILL FAST.  Per-row shots would destroy prefix caching -- 7,500 distinct 3,000-token
prefixes is 22M tokens of prefill. Per-CATEGORY shots give only 109 distinct prefixes for the
whole test set, and prompts are emitted category-contiguously so vLLM charges each block once:
about 109 x 3,000 = 0.3M tokens of prefix work in total, against 7,500 x ~120 tokens of row
tails either way. The speed budget is untouched.

NO LEAK.  Test queries overlap train queries 0 / 5,853 (measured), so a shot can never be the
row being judged, nor share its query.
"""
import os, sys, json, time, argparse, hashlib
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts_l3 as P
import prompts_l4_extra as PX
from l9_judge import (LABELS, DATA, ANSWER_PREFIX, SENTINEL, _clean, render_item, user_block,
                      build_label_token_sets, assert_label_tokens)
from l9_run import macro_f1, balance, read_out

GEREKCE_R = u"Ürün sorgunun istediği şeyin ta kendisi ve sorgunun yazdığı şartların hepsi tutuyor."
GEREKCE_S = u"Ürün aynı ihtiyaca kısmen hizmet ediyor ama sorgunun istediği varyantın kendisi değil."


def catkey(c, level=2):
    return '/'.join(str(c).split('/')[:level])


def shot_json(r, lab, order='label'):
    ger = GEREKCE_R if lab == 'relevant' else GEREKCE_S
    ans = ({"etiket": lab, "gerekce": ger} if order == 'label'
           else {"gerekce": ger, "etiket": lab})
    return user_block(r) + "\nCEVAP: " + json.dumps(ans, ensure_ascii=False)


def build_cat_shots(tr, keys, level=2, n_pair=3, n_extra=2, seed=0, order='label'):
    """For each category key: prefer contrastive same-query (relevant, semi) pairs; top up with
    independent rows of whichever class is short. The 3 authored `irrelevant` shots from L3 are
    appended unchanged -- train.parquet has no gold row of that class to draw a per-category one
    from, and inventing one per category would be manufacturing evidence, not demonstrating it."""
    tr = tr.copy()
    tr['_ck'] = tr.category.map(lambda c: catkey(c, level))
    tr = tr[tr._ck.isin(set(keys))]
    out = {}
    authored = []
    for r in P.SHOTS_IRRELEVANT:
        ans = ({"etiket": r['etiket'], "gerekce": r['gerekce']} if order == 'label'
               else {"gerekce": r['gerekce'], "etiket": r['etiket']})
        authored.append(user_block(r) + "\nCEVAP: " + json.dumps(ans, ensure_ascii=False))
    for ck, d in tr.groupby('_ck'):
        # WAS `abs(hash(ck))`. Python randomises str hashing per process, so the contrastive
        # pairs L9 picked are NOT the pairs a re-run picks: the `catfs_hard` control here came
        # back 2846/2550/2104 against L9's 2956/2525/2019 -- 110 rows of drift with a
        # byte-identical rubric, against 10 rows for the plain `hard` control. That drift is
        # shot selection, not the model. md5 makes the block a function of the category name.
        rng = np.random.default_rng(
            int(hashlib.md5(ck.encode('utf-8')).hexdigest()[:8], 16) + seed)
        blocks = []
        nq = d.groupby('query').real_label.nunique()
        both = list(nq[nq == 2].index)
        rng.shuffle(both)
        for q in both[:n_pair]:
            dq = d[d['query'] == q]
            r1 = dq[dq.real_label == 'relevant'].iloc[0]
            r2 = dq[dq.real_label == 'semi-relevant'].iloc[0]
            blocks.append(shot_json(r1, 'relevant', order))
            blocks.append(shot_json(r2, 'semi-relevant', order))
        for lab in ('relevant', 'semi-relevant'):
            dl = d[(d.real_label == lab) & (~d['query'].isin(set(both[:n_pair])))]
            if len(dl):
                for _, r in dl.sample(min(n_extra, len(dl)), random_state=seed).iterrows():
                    blocks.append(shot_json(r, lab, order))
        # interleave the authored irrelevant shots so no class sits in a run
        merged = []
        for i, b in enumerate(blocks):
            merged.append(b)
            if i % 3 == 2 and authored:
                merged.append(authored[(i // 3) % len(authored)])
        for a in authored:
            if a not in merged:
                merged.append(a)
        out[ck] = "\n\n---\n\n".join(merged)
    return out, "\n\n---\n\n".join(authored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--splits', default='public_test.parquet')
    ap.add_argument('--rubric', default='hard', choices=['hard', 'zs'])
    ap.add_argument('--order', default='label')
    ap.add_argument('--tag', default='catfs')
    ap.add_argument('--level', type=int, default=2)
    ap.add_argument('--n-pair', dest='n_pair', type=int, default=3)
    ap.add_argument('--n-extra', dest='n_extra', type=int, default=2)
    ap.add_argument('--outdir', default='/root/L9/out')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-attr', dest='max_attr', type=int, default=220)
    ap.add_argument('--shot-attr', dest='shot_attr', type=int, default=140)
    ap.add_argument('--logprobs', type=int, default=48)
    ap.add_argument('--max-model-len', dest='max_model_len', type=int, default=10240)
    ap.add_argument('--gpu-util', dest='gpu_util', type=float, default=0.90)
    ap.add_argument('--max-num-seqs', dest='max_num_seqs', type=int, default=512)
    ap.add_argument('--max-batched-tokens', dest='max_batched_tokens', type=int, default=32768)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=args.model, tokenizer=args.model, dtype='auto',
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=True, max_logprobs=args.logprobs + 8,
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_batched_tokens,
              seed=0, disable_log_stats=True)
    tok = llm.get_tokenizer()
    idx, cls = build_label_token_sets(tok)
    assert_label_tokens(tok, idx, cls)
    id2cls = dict(zip(idx.tolist(), cls.tolist()))
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=args.logprobs)

    tr = pd.read_parquet(os.path.join(DATA, 'train.parquet'))
    head = ([P.ROL, P.ETIKETLER, PX.KARAR_HARD, PX.KALIBRASYON_HARD] if args.rubric == 'hard'
            else [P.ROL, P.ETIKETLER, P.KARAR, P.KALIBRASYON])

    for split in [s for s in args.splits.split(',') if s]:
        path = split if os.path.isabs(split) else os.path.join(DATA, split)
        df = pd.read_parquet(path)
        if args.limit:
            df = df.head(args.limit).reset_index(drop=True)
        df = df.reset_index(drop=True); df['_ord'] = np.arange(len(df))
        df['_ck'] = df.category.map(lambda c: catkey(c, args.level))
        sname = os.path.basename(split).replace('.parquet', '')
        tag = args.tag if sname.startswith('public_test') else '%s_%s' % (args.tag, sname)

        t_s = time.time()
        shots, fallback = build_cat_shots(tr, set(df._ck), args.level, args.n_pair,
                                          args.n_extra, order=args.order)
        miss = sum(1 for k in df._ck.unique() if k not in shots)
        print('[L9C] %s: %d categories, shot blocks built in %.1fs (%d without train rows -> '
              'global block)' % (sname, df._ck.nunique(), time.time() - t_s, miss), flush=True)

        prompts, order_ix, plens = [], [], []
        for ck, d in df.groupby('_ck', sort=True):
            sysmsg = u"\n\n".join(head + [P.ORNEK_BASLIK + "\n\n" + shots.get(ck, fallback),
                                          P.out_block(args.order)])
            templated = tok.apply_chat_template(
                [{'role': 'user', 'content': sysmsg + "\n\n" + P.SIRADAKI + "\n\n" + SENTINEL}],
                tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX
            add_special = not (tok.bos_token and templated.startswith(tok.bos_token))
            pre_text, post_text = templated.split(SENTINEL)
            enc = lambda s: tok.encode(s, add_special_tokens=add_special)
            prefix_ids = enc(pre_text); Pn = len(prefix_ids)
            for r in d.to_dict('records'):
                ids = enc(pre_text + "SORGU: %s\n%s" % (_clean(r.get('query')),
                                                        render_item(r, args.max_attr)) + post_text)
                assert ids[:Pn] == prefix_ids, 'tokenizer merged across the sentinel boundary'
                # NEVER truncate from the end: the label is read at the LAST position, so
                # clipping the tail silently moves the read off the answer prefix and the
                # 3-vector becomes noise. Fail loudly instead -- max_model_len is a knob.
                assert len(ids) <= args.max_model_len, (
                    'prompt %d tokens exceeds max_model_len %d for category %r; raise '
                    '--max-model-len or lower --n-pair/--shot-attr'
                    % (len(ids), args.max_model_len, ck))
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                order_ix.append(int(r['_ord'])); plens.append(len(ids))
        print('[L9C] prompt tokens mean %.0f max %d' % (float(np.mean(plens)), max(plens)),
              flush=True)

        t0 = time.time(); outs = llm.generate(prompts, sp); wall = time.time() - t0
        N = len(df)
        LGs, cov0 = read_out(outs, id2cls, len(prompts))
        LG = np.zeros((N, 3)); cov = np.zeros(N)
        for j, oi in enumerate(order_ix):
            LG[oi] = LGs[j]; cov[oi] = cov0[j]
        assert cov.mean() > 0.5, 'read position wrong'
        Pm = np.exp(LG - LG.max(1, keepdims=True))
        Pm = (Pm / Pm.sum(1, keepdims=True)).astype(np.float32)
        LAB = [LABELS[k] for k in Pm.argmax(1)]
        Qb, _ = balance(Pm.astype(np.float64))
        LABb = [LABELS[k] for k in Qb.argmax(1)]
        rec = {'tag': tag, 'split': sname, 'rows': N, 'level': args.level,
               'n_pair': args.n_pair, 'n_extra': args.n_extra, 'rubric': args.rubric,
               'categories': int(df._ck.nunique()), 'cats_without_train_rows': miss,
               'wall_s': round(wall, 2), 'rows_per_s': round(N / wall, 2),
               'est_7500_s': round(7500.0 / (N / wall), 1),
               'prompt_tokens_mean': round(float(np.mean(plens)), 1),
               'label_mass_mean': round(float(cov.mean()), 4),
               'raw_counts': pd.Series(LAB).value_counts().to_dict()}
        if 'real_label' in df.columns:
            gt = df.real_label.tolist(); labs = sorted(set(gt))
            rec['macroF1_raw'], rec['perclass_raw'] = macro_f1(gt, LAB, labs)
            rec['macroF1_bal'], rec['perclass_bal'] = macro_f1(gt, LABb, labs)
            rec['classes'] = labs
        np.save('%s/logits_L9_%s.npy' % (args.outdir, tag), Pm)
        np.save('%s/logits_L9_%s_log.npy' % (args.outdir, tag), LG.astype(np.float32))
        idcol = 'row_id' if 'row_id' in df.columns else df.columns[0]
        pd.DataFrame({'row_id': df[idcol], 'real_label': LAB}).to_csv(
            '%s/preds_public_test_L9_%s.csv' % (args.outdir, tag), index=False)
        pd.DataFrame({'row_id': df[idcol], 'real_label': LABb}).to_csv(
            '%s/preds_public_test_L9_%scal.csv' % (args.outdir, tag), index=False)
        json.dump(rec, open('%s/timing_L9_%s.json' % (args.outdir, tag), 'w'), indent=2)
        print('[L9C] %-22s %5d rows %6.1fs %6.1f r/s 7500->%6.1fs %s'
              % (tag, N, wall, N / wall, rec['est_7500_s'],
                 ('macroF1 raw %.4f bal %.4f' % (rec['macroF1_raw'], rec['macroF1_bal']))
                 if 'macroF1_raw' in rec else str(rec['raw_counts'])), flush=True)
    print('[L9C] DONE', flush=True)


if __name__ == '__main__':
    main()
