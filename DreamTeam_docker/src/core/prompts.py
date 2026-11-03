# -*- coding: utf-8 -*-
u"""Prompt construction for the two stages, and the reason there are two shapes.

CHAMPION (`gfs4/champ3`) -- `lib.champ.l10_run.encode_block`, untouched.  1,669-token rubric +
a global few-shot block drawn from train.parquet with seed 4, then the full product body.
~5,097 tokens per prompt, of which a 4,951-token shared prefix.

GATE, two of them, and they are architecturally different animals:

  `c3p_t`   the SAME checkpoint as the champion, wearing the champ3 rubric with NO shot block and
            a title-only body.  1,796-token prefix, 46.7-token body, agreement with the champion
            0.8637.  Because it is the same weights, gate and champion share ONE engine and ONE
            boot -- the single largest thing the recorded tables never priced.
  `moe26`   a DIFFERENT checkpoint (26B-A4B MoE), wearing the champion's own prompt byte for
            byte.  Agreement 0.8587.  24.7 GiB of weights next to the champion's 30.6 GiB does
            not fit a 40 GiB card, so this path costs a teardown and a second boot.

`render()` and the field order are `agents/L14/l14_gate.py`'s verbatim, so a gate body that shows
every field is byte-identical to the champion's body; the only thing a field subset changes is
which lines are absent.
"""
import numpy as np

FIELDS = [('title', u'BAŞLIK'), ('category', u'KATEGORİ'), ('brand', u'MARKA'),
          ('gender_age', u'CİNSİYET/YAŞ'), ('attributes', u'ÖZELLİKLER')]


def render(r, keep, max_attr, _clean):
    parts = []
    for key, lbl in FIELDS:
        if key not in keep:
            continue
        if key == 'gender_age':
            g, a = _clean(r.get('gender')), _clean(r.get('age_group'))
            if g or a:
                parts.append(u"%s: %s / %s" % (lbl, g or '-', a or '-'))
            continue
        v = _clean(r.get(key))
        if v:
            parts.append(u"%s: %s" % (lbl, v[:max_attr] if key == 'attributes' else v))
    return "\n".join(parts) if parts else u"(bu ürün hakkında başka bilgi verilmedi)"


def champion_sysmsg(shot_block, order='label'):
    import l10_run as L10
    return L10.sysmsg_for('gfs4', 'champ3', order, shot_block)


def gate_sysmsg(kind, shot_block=None, order='label'):
    import prompts_l3 as P
    import prompts_l10 as PS
    if kind == 'c3plain':
        return PS.SYSTEMS['champ3'](order)
    if kind == 'c3gfs4':
        return champion_sysmsg(shot_block, order)
    raise ValueError(kind)


def encode_champion(tok, sysmsg, recs, max_attr, max_model_len):
    import l10_run as L10
    return L10.encode_block(tok, sysmsg, recs, max_attr, max_model_len)


def encode_gate(tok, sysmsg, recs, keep, max_attr, max_model_len):
    u"""`l14_gate.encode_split` + its loop, verbatim in behaviour.  The answer prefix is appended
    AFTER the templated tail, so the read position is the last token of every prompt."""
    from l9_judge import ANSWER_PREFIX, SENTINEL, _clean
    import prompts_l3 as P
    templated = tok.apply_chat_template(
        [{'role': 'user', 'content': sysmsg + "\n\n" + P.SIRADAKI + "\n\n" + SENTINEL}],
        tokenize=False, add_generation_prompt=True)
    add_special = not (tok.bos_token and templated.startswith(tok.bos_token))
    pre_text, post_text = templated.split(SENTINEL)
    enc = lambda s: tok.encode(s, add_special_tokens=add_special)
    prefix_ids = enc(pre_text); Pn = len(prefix_ids)
    suffix = tok.encode(ANSWER_PREFIX, add_special_tokens=False)
    ids_list, plens = [], []
    enc_tail = lambda s: tok.encode(s, add_special_tokens=False)   # P3: prefix-append
    for r in recs:
        body = u"SORGU: %s\n%s" % (_clean(r.get('query')), render(r, keep, max_attr, _clean))
        ids = prefix_ids + enc_tail(body + post_text) + suffix
        # NEVER truncate: the label is read at the LAST position, so clipping moves the read off
        # the answer prefix and the 3-vector silently becomes noise.
        assert len(ids) <= max_model_len, ('gate prompt %d tokens > max_model_len %d'
                                           % (len(ids), max_model_len))
        ids_list.append(ids); plens.append(len(ids))
    return ids_list, Pn, plens


# ------------------------------------------------------------------ tokenisation, done ONCE
def tokenizer_fingerprint(tok):
    u"""Enough of a tokenizer's identity to decide whether two engines can share token ids.

    L21's `verify_readout.py` recorded that gemma-4-31B, the 26B-A4B MoE, the 12B, E4B and Asure
    all carry the SAME tokenizer -- vocab 262,144, read positions `ir`=583, `relevant`=85531,
    `semi`=69807.  When that holds, the gate and the champion can be tokenised once instead of
    twice, which on a 7,500-row run is ~16 s of the ~32 s L33 measured going into tokenisation.
    It is a property to CHECK, not to assume: a checkpoint with a different tokenizer would
    silently receive another model's token ids, and the read position would move.
    """
    probe = u'{"etiket": "irrelevant"} SORGU: kırmızı elbise\nBAŞLIK: test — ürün'
    return (len(tok), tuple(tok.encode(probe, add_special_tokens=False)),
            tuple(tok.encode('{"etiket": "', add_special_tokens=False)))


class PromptCache(object):
    u"""Token ids for one (prompt shape, system message, tokenizer) triple, indexed by row.

    FRAGILITY 4 -- the run pays for the same tokenisation twice.  L33 measured 31.7 s of prompt
    tokenisation in a 177 s sequential cascade: 18% of the wall, for work that is identical
    between the two stages whenever the gate wears the champion's prompt (the MoE gate does,
    byte for byte) and the two checkpoints share a tokenizer (they do).  The cache key contains
    the tokenizer fingerprint, so a checkpoint that does NOT share it simply misses the cache and
    re-tokenises, rather than being handed the wrong ids.
    """

    def __init__(self):
        self._store = {}
        self.aligns = {}          # (prompt shape, sysmsg length, body head) -> the pad it needs
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(kind, fields, sysmsg, fp):
        import hashlib
        return (kind, tuple(fields or ()), hashlib.sha1(sysmsg.encode('utf-8')).hexdigest(), fp)

    def get(self, key, row_ids):
        e = self._store.get(key)
        if e is None:
            self.misses += 1
            return None
        ids_by_row, Pn = e
        if not all(r in ids_by_row for r in row_ids):
            self.misses += 1
            return None
        self.hits += 1
        out = [ids_by_row[r] for r in row_ids]
        return out, Pn, [len(i) for i in out]

    def put(self, key, row_ids, ids_list, Pn):
        self._store[key] = (dict(zip(row_ids, ids_list)), Pn)


# ---------------------------------------------------- cache-block alignment of the shared prefix
# vLLM hashes its prefix cache in fixed-size blocks **from position 0**.  The champion's shared
# prefix is 4,951 tokens = 309x16 + 7, so those last 7 prefix tokens land in a block that also
# holds 9 body tokens.  That block is row-specific, so it can never be cached, and all 7,500 rows
# re-prefill it: 4.6% of every computed token, carrying zero information.
#
# The optimum is NOT remainder 0.  The body's constant head (`SORGU:`) is 3 tokens, so the first
# body block should start where the body's VARIABLE part starts.  Target `prefix + 3 ≡ 0 (mod G)`,
# i.e. 4,957 (+6 tokens) at G=16.  Conveniently 4,957 is also -3 mod 32, so the same +6 is optimal
# whether the binding granularity is 16 or 32 -- which matters, because the engine reports
# `block 32` for the FullAttentionSpec group and it is not established which one binds the hash.
# Padding to `≡ -B (mod 32)` satisfies both.
#
# ⚠️ It is a knife edge: one token off (4,956) is +4.8 s at a tight cache cap, i.e. it turns a
# 6.2% win into a loss.  So the alignment is MEASURED and ASSERTED, never assumed, and the pad is
# searched for against the real tokenizer rather than hard-coded as "+6".
ALIGN_GRANULARITY = 32          # satisfies 16 as well
PAD_UNIT = u"\n"                # whitespace between two blocks already separated by "\n\n"


def constant_body_head_tokens(tok, sysmsg, recs, encode_fn, max_attr, max_model_len):
    u"""How many tokens every row's body begins with, measured rather than assumed.

    F190 measured `SORGU:` at 3 tokens.  Measuring it here means the alignment stays correct if
    the body template, the tokenizer or the language ever changes -- and it is cheap: two rows.
    """
    if len(recs) < 2:
        return 0
    a, b = encode_fn(tok, sysmsg, [recs[0]], max_attr, max_model_len), \
           encode_fn(tok, sysmsg, [recs[-1]], max_attr, max_model_len)
    ia, Pn, _ = a
    ib, _, _ = b
    ia, ib = ia[0][Pn:], ib[0][Pn:]
    n = 0
    while n < min(len(ia), len(ib)) and ia[n] == ib[n]:
        n += 1
    return n


def find_prefix_pad(measure_prefix_len, body_head, granularity=ALIGN_GRANULARITY, max_units=256):
    u"""Smallest inert pad whose prefix length lands on `-body_head (mod granularity)`.

    `measure_prefix_len(pad_text) -> int` must build the prefix exactly the way the real prompt
    builder does.  The pad is searched, not computed, because repeated whitespace does not
    tokenise one-token-per-character on a BPE vocabulary -- k newlines is not k tokens.
    Returns `(pad_text, Pn_before, Pn_after)`, or raises if no pad in range lands exactly.
    """
    target = (-int(body_head)) % int(granularity)
    base = measure_prefix_len(u"")
    if base % granularity == target:
        return u"", base, base
    seen = {0: base}
    # One probe first: if the pad unit is worth exactly one token, the answer is arithmetic and
    # no search is needed.  It usually is not -- a BPE vocabulary merges runs of whitespace -- so
    # the linear scan below is the real path and the probe just skips it when it can.
    one = measure_prefix_len(PAD_UNIT)
    seen[1] = one
    per = one - base
    if per == 1:
        k = (target - base) % granularity
        n = measure_prefix_len(PAD_UNIT * k)
        if n % granularity == target:
            return PAD_UNIT * k, base, n
    if one % granularity == target:
        return PAD_UNIT, base, one
    for k in range(2, max_units + 1):
        n = measure_prefix_len(PAD_UNIT * k)
        if n % granularity == target:
            return PAD_UNIT * k, base, n
    raise RuntimeError(
        'no inert pad of up to %d units lands the shared prefix on %d (mod %d); prefix is %d '
        '(%d mod %d). Run with --prefix-align off rather than shipping a misaligned prefix that '
        'was believed to be aligned.' % (max_units, target, granularity, base,
                                         base % granularity, granularity))
