#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L9 -- 31B-class 3-class judge on vLLM with a ZERO-GENERATION logprob read-out.

WHY THIS EXISTS.  Every LLM leg in this run so far was a 12B (or smaller) checkpoint.  Stage-1
measured the size ladder on the *same* catalogue: encoder 0.920 -> gemma-4-12b 0.937 ->
12b LoRA 0.949 -> gemma-4-31B 0.975.  Size was a stronger lever there than fine-tuning, and
F122/F123 say the only lever left under a <=2-model budget is the LLM leg's own strength.
So: same measured prompts (L3/L4's, reused byte-identically so the diff is the weights),
bigger weights.

READ-OUT.  The assistant turn is PREFILLED with `{"etiket": "` so the label's first token sits at
one known position.  vLLM decodes exactly ONE token with `logprobs=K` and the 3-vector is the
marginal over every returned vocabulary token that is a unique prefix of exactly one label.
`relevant` is a token shared by all three labels ('ir'+'relevant', 'semi'+'-'+'relevant'), so
keying on it reads `relevant` for every row -- the unique-prefix rule is what makes the marginal
correct, and `assert_label_tokens` refuses to run if the three first tokens ever collide.

Everything is kept in LOG space and `_log.npy` written alongside: a confident row puts the losing
labels near -150 nats and exp(-150) is exactly 0.0 in float32, which silently destroys the
ranking information a blend needs.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts_l3 as P
import prompts_l4_extra as PX

LABELS = ['irrelevant', 'relevant', 'semi-relevant']
DATA = os.environ.get('S2_DATA', '/root/s2/data')
ANSWER_PREFIX = '{"etiket": "'
SENTINEL = 'ITEMSENTINEL'


def _clean(v):
    s = '' if v is None else str(v)
    return '' if s.lower() in ('', 'nan', 'none', 'unknown') else s.strip()


def render_item(r, max_attr=220):
    parts = [u"BAŞLIK: %s" % _clean(r.get('title'))]
    for key, lbl in (('category', u'KATEGORİ'), ('brand', u'MARKA')):
        v = _clean(r.get(key))
        if v:
            parts.append("%s: %s" % (lbl, v))
    g, a = _clean(r.get('gender')), _clean(r.get('age_group'))
    if g or a:
        parts.append(u"CİNSİYET/YAŞ: %s / %s" % (g or '-', a or '-'))
    at = _clean(r.get('attributes'))
    if at:
        parts.append(u"ÖZELLİKLER: %s" % at[:max_attr])
    return "\n".join(parts)


def user_block(r):
    return "SORGU: %s\n%s" % (_clean(r.get('query')), render_item(r))


def shots_text(order='label'):
    out = []
    for r in P.shot_blocks():
        ans = ({"etiket": r['etiket'], "gerekce": r['gerekce']} if order == 'label'
               else {"gerekce": r['gerekce'], "etiket": r['etiket']})
        out.append(user_block(r) + "\nCEVAP: " + json.dumps(ans, ensure_ascii=False))
    return "\n\n---\n\n".join(out)


def system_for(variant, order='label'):
    return {'zs':     lambda: P.system_zs(order),
            'fs':     lambda: P.system_fs(shots_text(order), order),
            'sh':     lambda: P.system_sh(order),
            'hard':   lambda: PX.system_hard(order),
            'hardfs': lambda: PX.system_hard_fs(shots_text(order), order)}[variant]()


# ------------------------------------------------------------------ label token sets (vLLM tok)
def build_label_token_sets(tok):
    """Token ids whose surface form, after stripping the JSON punctuation that can precede a
    label, is a prefix of exactly ONE label."""
    # One BATCH call, not 262,144 individual ones: convert_tokens_to_string per id crossed the
    # Rust boundary once per token and cost ~5 minutes of pure start-up on this vocabulary.
    strip_chars = ' \t\n\r"\'{,:'
    idx, cls = [], []
    V = len(tok)
    pieces = tok.convert_ids_to_tokens(list(range(V)))
    for i, piece in enumerate(pieces):
        if not piece:
            continue
        s = piece.replace(u'▁', ' ').lstrip(strip_chars).lower()
        if not s:
            continue
        hit = [k for k, lab in enumerate(LABELS) if lab.startswith(s)]
        if len(hit) == 1:
            idx.append(i); cls.append(hit[0])
    return np.array(idx, dtype=np.int64), np.array(cls, dtype=np.int64)


def assert_label_tokens(tok, idx, cls):
    pre = tok.encode(ANSWER_PREFIX, add_special_tokens=False)
    id2cls = dict(zip(idx.tolist(), cls.tolist()))
    firsts = {}
    for k, lab in enumerate(LABELS):
        ids = tok.encode(ANSWER_PREFIX + lab, add_special_tokens=False)
        assert ids[:len(pre)] == pre, 'tokenizer merged across the prefix boundary: %s' % lab
        t = ids[len(pre)]
        assert t in id2cls, 'first token of %r (id %d, %r) is not in any label set' % (
            lab, t, tok.convert_ids_to_tokens([t]))
        assert id2cls[t] == k, 'first token of %r classified as %r' % (lab, LABELS[id2cls[t]])
        firsts[lab] = t
    assert len(set(firsts.values())) == 3, 'label first tokens collide: %r' % firsts
    print('[L9] label first tokens OK: %r' % {l: (t, tok.convert_ids_to_tokens([t])[0])
                                              for l, t in firsts.items()}, flush=True)


def logsumexp(a, axis=None):
    a = np.asarray(a, dtype=np.float64)
    m = a.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def temper_log(LG, T):
    z = LG / float(T)
    z = z - logsumexp(z, axis=1)[:, None]
    p = np.exp(z)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--variant', default='hardfs', choices=['zs', 'fs', 'sh', 'hard', 'hardfs'])
    ap.add_argument('--order', default='label', choices=['label', 'reason'])
    ap.add_argument('--tag', default=None)
    ap.add_argument('--split', default='public_test.parquet')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--outdir', default='/root/L9/out')
    ap.add_argument('--max-attr', dest='max_attr', type=int, default=220)
    ap.add_argument('--logprobs', type=int, default=48)
    ap.add_argument('--max-model-len', dest='max_model_len', type=int, default=3072)
    ap.add_argument('--gpu-util', dest='gpu_util', type=float, default=0.90)
    ap.add_argument('--max-num-seqs', dest='max_num_seqs', type=int, default=256)
    ap.add_argument('--max-batched-tokens', dest='max_batched_tokens', type=int, default=16384)
    ap.add_argument('--tp', type=int, default=1)
    args = ap.parse_args()
    tag = args.tag or args.variant
    os.makedirs(args.outdir, exist_ok=True)

    path = args.split if os.path.isabs(args.split) else os.path.join(DATA, args.split)
    df = pd.read_parquet(path)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    recs = df.to_dict('records')
    N = len(df)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=args.model, tokenizer=args.model, dtype='auto',
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=True, max_logprobs=args.logprobs + 8,
              max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_batched_tokens,
              tensor_parallel_size=args.tp, seed=0, disable_log_stats=True)
    tok = llm.get_tokenizer()

    idx, cls = build_label_token_sets(tok)
    assert_label_tokens(tok, idx, cls)
    id2cls = dict(zip(idx.tolist(), cls.tolist()))
    print('[L9] label-token set sizes: %r' % np.bincount(cls, minlength=3).tolist(), flush=True)

    # ---- build the templated prompt.  The chat template comes from the checkpoint itself so we
    # cannot get the turn markers wrong for a model family we have not run before.  The sentinel
    # proves the shared rubric prefix tokenises identically for every row (i.e. that vLLM's
    # automatic prefix cache can actually share it).
    sysmsg = system_for(args.variant, args.order)
    full_user = sysmsg + "\n\n" + P.SIRADAKI + "\n\n" + SENTINEL
    templated = tok.apply_chat_template([{'role': 'user', 'content': full_user}],
                                        tokenize=False, add_generation_prompt=True)
    templated = templated + ANSWER_PREFIX
    # apply_chat_template already emits <bos> for gemma; adding specials again would double it.
    add_special = not (tok.bos_token and templated.startswith(tok.bos_token))
    pre_text, post_text = templated.split(SENTINEL)
    enc = lambda s: tok.encode(s, add_special_tokens=add_special)
    prefix_ids = enc(pre_text)
    Pn = len(prefix_ids)

    prompts, tails = [], []
    for r in recs:
        ids = enc(pre_text + "SORGU: %s\n%s" % (_clean(r.get('query')),
                                                render_item(r, args.max_attr)) + post_text)
        assert ids[:Pn] == prefix_ids, 'tokenizer merged across the sentinel boundary'
        tails.append(len(ids) - Pn)
        prompts.append(TokensPrompt(prompt_token_ids=ids))
    print('[L9] rows=%d  shared prefix=%d tok  tail mean %.0f max %d  variant=%s'
          % (N, Pn, float(np.mean(tails)), max(tails), args.variant), flush=True)

    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=args.logprobs)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    wall = time.time() - t0

    LG = np.zeros((N, 3), dtype=np.float64)
    covered = np.zeros(N)
    for i, o in enumerate(outs):
        lp = o.outputs[0].logprobs[0]           # {token_id: Logprob}
        v = np.full(3, -np.inf)
        tot = []
        for tid, L in lp.items():
            tot.append(L.logprob)
            k = id2cls.get(int(tid))
            if k is not None:
                v[k] = np.logaddexp(v[k], L.logprob)
        covered[i] = float(np.exp(logsumexp(v) - 0.0))     # sum p over label tokens
        LG[i] = v - logsumexp(v)
    LG = np.nan_to_num(LG, neginf=-200.0)
    print('[L9] label-token mass at the forced position: mean %.4f  min %.4f  (<0.5 on %d rows)'
          % (covered.mean(), covered.min(), int((covered < 0.5).sum())), flush=True)
    assert covered.mean() > 0.5, 'read position is wrong -- the label tokens hold no mass'

    Pm = np.exp(LG - LG.max(1, keepdims=True)); Pm = (Pm / Pm.sum(1, keepdims=True)).astype(np.float32)
    LAB = [LABELS[k] for k in Pm.argmax(1)]
    print('[L9] === %d rows in %.1f s = %.2f rows/s  (7500 rows would be %.1f s) ==='
          % (N, wall, N / wall, 7500.0 / (N / wall)), flush=True)
    print('[L9] one-hot %.2f%%  mean max-prob %.4f' % ((Pm.max(1) >= 0.999999).mean() * 100,
                                                       Pm.max(1).mean()), flush=True)
    print('[L9] RAW predicted counts:', pd.Series(LAB).value_counts().to_dict(), flush=True)

    np.save('%s/logits_L9_%s.npy' % (args.outdir, tag), Pm)
    np.save('%s/logits_L9_%s_log.npy' % (args.outdir, tag), LG.astype(np.float32))
    for T in (2, 3):
        np.save('%s/logits_L9_%s_T%d.npy' % (args.outdir, tag, T), temper_log(LG, T))
    idcol = 'row_id' if 'row_id' in df.columns else df.columns[0]
    pd.DataFrame({'row_id': df[idcol], 'real_label': LAB}).to_csv(
        '%s/preds_public_test_L9_%s.csv' % (args.outdir, tag), index=False)
    json.dump({'rows': N, 'wall_s': round(wall, 2), 'rows_per_s': round(N / wall, 2),
               'est_7500_s': round(7500.0 / (N / wall), 1),
               'prefix_tokens': Pn, 'tail_mean_tokens': float(np.mean(tails)),
               'variant': args.variant, 'order': args.order, 'max_attr': args.max_attr,
               'label_mass_mean': float(covered.mean()),
               'raw_counts': pd.Series(LAB).value_counts().to_dict(),
               'model': args.model, 'split': args.split},
              open('%s/timing_L9_%s.json' % (args.outdir, tag), 'w'), indent=2)
    print('[L9] wrote %s/logits_L9_%s.npy' % (args.outdir, tag), flush=True)


if __name__ == '__main__':
    main()
