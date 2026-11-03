# -*- coding: utf-8 -*-
u"""Engine construction, boot timing, teardown -- and a replay mock so the whole pipeline runs
end to end without a GPU.

THREE THINGS THIS MODULE EXISTS TO STOP HAPPENING AGAIN.

1. **`VLLM_ATTENTION_BACKEND` is a silent no-op in vLLM 0.28** -- it was removed from `envs.py`.
   Every earlier attempt on that axis set the variable, read `TRITON_ATTN` in the log and
   concluded vLLM was *forcing* Triton; it was ignoring the request (F169).  So this module
   (a) never sets the variable, (b) passes `attention_backend` as an ENGINE ARGUMENT, and
   (c) after boot, probes the live model for the attention implementation class actually in use
   and refuses to continue if it is not the one asked for.  A silent fallback here costs 1.42x.

2. **Nobody ever recorded an engine boot.**  L28's shipped logs contain `28.3 / 28.5 / 28.7 /
   30.8 / 31.5 s` warm and `118.5 / 129.0 / 131.4 s` cold for the 31B INT8 on an A100 40GB, and
   L34's 80GB box shows `142.2 s`; no table contains any of them.  L33 then measured the pair
   directly, and the cold/warm split turns out to be a CACHE effect, not a compile effect:

       eager (the shipped recipe)   first-on-box   warm        compiled  cold    warm
       gate (FP8 MoE)                    63.4 s    19.5 s                82.3 s  35.3 s
       champion (31B INT8)               23.6 s    22.2 s                82.4 s  34.0 s

   The gate's 63.4 -> 19.5 s collapse is FP8-MoE JIT/autotune caches in `~/.cache/vllm`: 44 s
   that evaporate after ONE warm-up run on the box.  `setup.sh` therefore ends with a warm-up.
   Every boot here is timed into its own manifest line, and so is every teardown.

3. **A 40 GB card cannot hold the gate and the champion at once, and it is not close.**
   Measured, not argued: gate weights **24.75 GiB** + champion weights **30.63 GiB** = **55.38
   GiB** before a single KV block.  Running it anyway -- gate at its own minimum, champion handed
   the 10.28 GiB left of the A100 budget -- produces `Available KV cache memory: -21.61 GiB`.
   A 48/49 GB card does not fix it either.  So the two-checkpoint path is SEQUENTIAL by
   construction (load-run-free-load-run), not as a fallback, and `shutdown()` has to really free:
   vLLM keeps the model resident until the engine core is collected, and a second `LLM(...)` on
   top of a live one OOMs during memory PROFILING rather than during weight load, which makes the
   traceback point at the wrong model.
"""
import os, gc, json, time
import numpy as np

# ---------------------------------------------------------------- engine profiles (all measured)
PROFILES = {
    # the deliverable's champion: INT8 W8A8 + FlashInfer attention.  269.0 s / 7500 rows on an
    # A100-SXM4-40GB (L28, full-run measurement, not extrapolated).  The util/mml values are not
    # free choices: at 0.97 FlashInfer's workspace leaves its sampler no room and the engine OOMs
    # during warm-up; at <=0.92 with mml 6144 the KV reservation no longer fits.  The longest
    # prompt in public_test is 5,178 tokens, so 5,250 is the smallest safe max_model_len.
    'champ_int8_fi': dict(
        model='lokeshe09/gemma-4-31B-it-INT8',
        max_model_len=5250, gpu_memory_utilization=0.94, enforce_eager=True,
        max_num_batched_tokens=4096, max_num_seqs=256, attention_backend='FLASHINFER',
        expect_impl=('FlashInfer',), measured_7500_s=269.0, weights_gib=30.63),
    # the zero-risk twin: logits BIT-IDENTICAL to L20's INT8 baseline, 382.8 s steady state.
    'champ_int8_triton': dict(
        model='lokeshe09/gemma-4-31B-it-INT8',
        max_model_len=6144, gpu_memory_utilization=0.97, enforce_eager=True,
        max_num_batched_tokens=4096, max_num_seqs=256, attention_backend=None,
        expect_impl=('Triton',), measured_7500_s=382.8, weights_gib=30.63),
    # the accuracy reference: 0.9133, 565.4 s.  FP8 checkpoint, but SM80 has no FP8 tensor cores,
    # so this runs as W8A16 through Marlin and is the slow one.
    'champ_fp8': dict(
        model='RedHatAI/gemma-4-31B-it-FP8-block',
        max_model_len=6144, gpu_memory_utilization=0.95, enforce_eager=True,
        max_num_batched_tokens=4096, max_num_seqs=256, attention_backend=None,
        expect_impl=('Triton',), measured_7500_s=565.4, weights_gib=31.07),
    # the gate.  ⚠ NEVER MEASURED ON AN A100.  205.9 s on an A40 (SM86), where the FP8 checkpoint
    # also falls back to W8A16 Marlin -- the same regime an A100 will use -- so the A40 ratio is
    # the right one to carry, and carrying it against the FP8 champion's A100 wall (565.4 s)
    # gives ~86 s, not the 58.5 s the frontier tables use (they scaled it against the INT8
    # champion's 384.9 s, i.e. they gave the gate an INT8 speed-up it does not get).  The run's
    # own audit puts it at ~104 s.  `measured_7500_s` below is therefore marked DERIVED and the
    # manifest prints the real wall, so the first real run settles it.
    # the OTHER gate: the SAME 31B INT8 checkpoint wearing the champ3 rubric with no shot block
    # and a title-only body.  It shares the champion's engine, so this path pays ONE boot, not two
    # -- an advantage worth 28-31 s warm / 118-142 s cold that no recorded table contains.
    # Priced on the reference box class (L24's `g1796`, 121.15 s) it is 102.5 s in the A100-INT8
    # frame, i.e. the SAME price as the MoE (104.2 s), and at 20% routing it scores 0.8987 against
    # the MoE's 0.8767.  Which gate actually wins is being measured on one A100 in one session;
    # until that lands, both are supported and neither is hard-coded.
    'gate_c3p_t': dict(
        model=None,                      # inherits the champion's checkpoint
        same_checkpoint_as_champion=True,
        prompt=dict(kind='c3plain', fields=('title',)),
        measured_7500_s=None, audited_a100_int8_s=102.5, weights_gib=30.63),
    'gate_moe': dict(
        model='RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic',
        max_model_len=5250, gpu_memory_utilization=0.90, enforce_eager=True,
        max_num_batched_tokens=4096, max_num_seqs=256, attention_backend=None,
        expect_impl=('Triton',), measured_7500_s=None, audited_a100_int8_s=104.2,
        prompt=dict(kind='c3gfs4', fields=None), same_checkpoint_as_champion=False,
        weights_gib=24.75),
}
for _p in PROFILES.values():
    _p.setdefault('prompt', dict(kind='c3gfs4', fields=None))
    _p.setdefault('same_checkpoint_as_champion', False)


def _probe_kv_pool(llm):
    u"""How many cache blocks the engine actually got, and how big a block is.

    Read from the live engine rather than from the log, for the same reason the attention backend
    is: a log line is a claim about what vLLM intended, and this decides whether the shared prefix
    survives in cache or is evicted every batch.
    """
    out = {}
    for path in (('llm_engine', 'cache_config'), ('llm_engine', 'vllm_config', 'cache_config')):
        o = llm
        try:
            for a in path:
                o = getattr(o, a)
            for k, src in (('block_size', 'block_size'), ('num_gpu_blocks', 'num_gpu_blocks'),
                           ('num_gpu_blocks_override', 'num_gpu_blocks_override')):
                v = getattr(o, src, None)
                if v:
                    out[k] = int(v)
            if out.get('num_gpu_blocks'):
                return out
        except Exception:
            continue
    return out


def check_prefix_cache_headroom(llm, prefix_tokens, label='engine'):
    u"""FRAGILITY -- the shared prefix does not fit the cache with room to spare.

    The whole cascade rests on a ~4,950-token prefix staying resident.  With the hybrid KV manager
    ON that is comfortable: 2,194 blocks against a 309-block prefix, 7.1x clear.  With it OFF --
    which measured 3.1% faster, so somebody will try it -- the pool is 398 blocks against a
    323-block floor: **1.23x clear**, and a slightly longer prefix leaves 56 blocks for every row
    in flight.  At that ratio the prefix is evicted between batches and the alignment win inverts.
    Checked at boot, where it is still cheap to change the flags.
    """
    info = _probe_kv_pool(llm)
    bs = info.get('block_size') or 0
    pool = info.get('num_gpu_blocks') or 0
    rec = dict(info)
    if not (bs and pool and prefix_tokens):
        rec['headroom'] = None
        print('[L38] %-18s KV pool not readable from the engine; prefix-cache headroom UNCHECKED'
              % label, flush=True)
        return rec
    need = -(-int(prefix_tokens) // bs)                     # ceil
    rec.update(prefix_tokens=int(prefix_tokens), prefix_blocks=need,
               headroom=round(pool / float(need), 2))
    msg = ('[L38] %-18s KV pool %d blocks x %d tok; shared prefix %d tok = %d blocks; headroom %.2fx'
           % (label, pool, bs, prefix_tokens, need, rec['headroom']))
    if rec['headroom'] < 2.0:
        msg += ('\n[L38] WARNING: under 2x. The shared prefix will be evicted between batches and '
                'the block-alignment win inverts. This is the signature of the hybrid KV manager '
                'being OFF (measured: 398 blocks against a 323-block floor = 1.23x). Turn it back '
                'on, or raise gpu_memory_utilization / lower max_model_len.')
    print(msg, flush=True)
    return rec


def _probe_attention_impl(llm):
    u"""Ask the LIVE model which attention implementation it built, rather than trusting the log
    line.  Returns a sorted list of class names, or [] if the probe is not available."""
    def _fn(model):
        names = set()
        for _, m in model.named_modules():
            impl = getattr(m, 'impl', None)
            _n = type(m).__name__.lower()
            if impl is not None and ('attn' in _n or 'attention' in _n):
                names.add(type(impl).__name__)
        return sorted(names)
    try:
        out = llm.apply_model(_fn)
        return sorted(set(out[0] if isinstance(out, list) and out else out or []))
    except Exception as e:                                   # diagnostic only, never fatal
        return ['<probe failed: %r>' % (e,)]


class VllmJudge(object):
    u"""One vLLM engine, booted from a named profile, timed, probed and freed."""

    def __init__(self, profile, model=None, overrides=None, logprobs=48, seed=0):
        self.profile_name = profile
        self.cfg = dict(PROFILES[profile])
        if model:
            self.cfg['model'] = model
        self.cfg.update(overrides or {})
        self.logprobs = logprobs
        self.seed = seed
        self.llm = None
        self.boot_s = None
        self.attn_impl = None
        if os.environ.pop('VLLM_ATTENTION_BACKEND', None) is not None:
            print('[L38] WARNING: VLLM_ATTENTION_BACKEND was set in the environment and has been '
                  'removed -- it is a silent no-op in vLLM 0.28 and its presence means somebody '
                  'believes it works (F169).', flush=True)

    # ---------------------------------------------------------------------------- boot / teardown
    def boot(self):
        from vllm import LLM
        c = self.cfg
        kw = dict(model=c['model'], tokenizer=c['model'], dtype='auto',
                  max_model_len=c['max_model_len'],
                  gpu_memory_utilization=c['gpu_memory_utilization'],
                  enable_prefix_caching=True, max_logprobs=self.logprobs + 8,
                  max_num_seqs=c['max_num_seqs'],
                  max_num_batched_tokens=c['max_num_batched_tokens'],
                  # gemma-4 is multimodal; vLLM otherwise profiles the vision tower with six
                  # max-size videos and the reservation eats the KV cache the text run needs.
                  limit_mm_per_prompt={'image': 0, 'video': 0},
                  enforce_eager=c['enforce_eager'], seed=self.seed, disable_log_stats=True)
        t0 = time.time()
        backend = c.get('attention_backend')
        if backend:
            try:
                self.llm = LLM(attention_backend=backend, **kw)
            except TypeError:
                # vLLM moved the knob between point releases; the other spelling is the config
                # object.  If BOTH are rejected the run must stop -- silently booting on Triton
                # would report a champion wall 1.42x too slow and nobody would know why.
                self.llm = LLM(attention_config={'backend': backend}, **kw)
        else:
            self.llm = LLM(**kw)
        self.boot_s = time.time() - t0
        self.attn_impl = _probe_attention_impl(self.llm)
        self.kv_pool = _probe_kv_pool(self.llm)
        print('[L38] %-18s engine up in %.1fs   attention impl=%s'
              % (self.profile_name, self.boot_s, ','.join(self.attn_impl) or '?'), flush=True)
        if self.boot_s > 45 and c.get('enforce_eager'):
            print('[L38] NOTE: %.0f s is a COLD boot -- the JIT/autotune caches in ~/.cache/vllm '
                  'are empty. A second run on this box boots in ~20 s. If this is the delivery '
                  'run, you are paying %.0f s that setup.sh\'s warm-up would have removed.'
                  % (self.boot_s, self.boot_s - 20), flush=True)
        exp = c.get('expect_impl') or ()
        if exp and self.attn_impl and not any(e.lower() in i.lower()
                                              for i in self.attn_impl for e in exp):
            raise RuntimeError(
                'attention backend not honoured: asked for %r, the live model built %r. '
                'This is exactly the 1.42x that a silent fallback costs; refusing to continue. '
                'Pass --allow-attention-fallback to override.' % (backend or exp, self.attn_impl))
        return self.boot_s

    def shutdown(self):
        u"""Actually give the VRAM back.  On a 40 GB card the next boot fails during memory
        profiling if this is a no-op, and the traceback names the wrong model."""
        if self.llm is None:
            return 0.0
        t0 = time.time()
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel, \
                destroy_distributed_environment
            destroy_model_parallel(); destroy_distributed_environment()
        except Exception:
            pass
        try:
            del self.llm.llm_engine
        except Exception:
            pass
        self.llm = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            free, tot = torch.cuda.mem_get_info()
            print('[L38] %-18s freed; %.2f GiB of %.2f GiB now free'
                  % (self.profile_name, free / 2 ** 30, tot / 2 ** 30), flush=True)
        except Exception:
            pass
        return time.time() - t0

    # -------------------------------------------------------------------------------- inference
    def check_headroom(self, prefix_tokens):
        return check_prefix_cache_headroom(self.llm, prefix_tokens, self.profile_name)

    def get_tokenizer(self):
        return self.llm.get_tokenizer()

    def generate(self, prompts, sp):
        return self.llm.generate(prompts, sp)


# ------------------------------------------------------------------------------------ the mock
class _LP(object):
    __slots__ = ('logprob',)

    def __init__(self, v):
        self.logprob = float(v)


class _Out(object):
    def __init__(self, lp):
        self.logprobs = [lp]


class _Req(object):
    def __init__(self, lp):
        self.outputs = [_Out(lp)]


class MockTokenizer(object):
    u"""A deterministic greedy word/char tokenizer with the REAL read-position ids.

    ids 583 / 85531 / 69807 carry `ir` / `relevant` / `semi`, the same ids the gemma-4 tokenizer
    uses (L21's `verify_readout.py` recorded them for every model in the ladder), so
    `build_label_token_sets` and `assert_label_tokens` run their real code paths against a real
    label geometry.  Multi-character pieces are matched only at a word start and never contain
    whitespace, so `encode(a + b) == encode(a) + encode(b)` whenever `a` ends in whitespace --
    which is what makes the sentinel prefix-stability assert in `encode_block` meaningful here
    rather than vacuous.
    """
    V = 262144
    bos_token = '<bos>'

    def __init__(self, corpus_words=()):
        self.pieces = [''] * self.V
        self.piece2id = {}
        specials = {583: 'ir', 85531: 'relevant', 69807: 'semi'}
        nxt = 1000
        for i, p in specials.items():
            self.pieces[i] = p; self.piece2id[p] = i
        for ch in sorted(set('abcdefghijklmnopqrstuvwxyz0123456789'
                             u'çğıöşüâîû'
                             ' \n\t.,:;/-–—&()[]{}"\'%+*_=<>?!#|\\@$~^`')):
            if ch not in self.piece2id:
                while self.pieces[nxt]:
                    nxt += 1
                self.pieces[nxt] = ch; self.piece2id[ch] = nxt; nxt += 1
        for w in corpus_words:
            if w and w not in self.piece2id and not any(c.isspace() for c in w):
                while self.pieces[nxt]:
                    nxt += 1
                self.pieces[nxt] = w; self.piece2id[w] = nxt; nxt += 1
        self._unk = self.piece2id['?']
        self._maxlen = max((len(p) for p in self.piece2id), default=1)
        self._pcache = {}
        self._pcache_keys = []

    def __len__(self):
        return self.V

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.pieces[ids]
        return [self.pieces[i] for i in ids]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        body = '\n'.join(m['content'] for m in msgs)
        return '%s<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n' % (
            self.bos_token, body)

    def encode(self, s, add_special_tokens=False):
        # Prefix memo.  `encode_block` re-encodes the whole ~30 KB shared prefix once per row;
        # without this the CPU end-to-end run is ~225 M characters of Python loop and never
        # finishes.  A cached prefix is only reused when it ends on a NON-alphanumeric character,
        # which is precisely the condition under which this tokenizer cannot merge across the
        # boundary -- so the memo cannot paper over the prefix-stability property that
        # `encode_block`'s assert exists to check.
        for key in self._pcache_keys:
            if len(key) < len(s) and s.startswith(key):
                return self._pcache[key] + self._encode_raw(s[len(key):])
        ids = self._encode_raw(s)
        if s and not s[-1].isalnum() and len(s) > 256 and len(self._pcache) < 8:
            self._pcache[s] = ids
            self._pcache_keys = sorted(self._pcache, key=len, reverse=True)
        return ids

    def _encode_raw(self, s):
        out = []
        i, n = 0, len(s)
        while i < n:
            word_start = (i == 0) or (not s[i - 1].isalnum())
            hit = None
            if word_start:
                for L in range(min(self._maxlen, n - i), 1, -1):
                    cand = s[i:i + L].lower()
                    if cand in self.piece2id and not any(c.isspace() for c in cand):
                        hit = (cand, L); break
            if hit is None:
                cand = s[i].lower()
                hit = (cand, 1) if cand in self.piece2id else ('?', 1)
            out.append(self.piece2id[hit[0]]); i += hit[1]
        return out


class MockJudge(object):
    u"""Replays a committed (N,3) log-probability matrix, keyed by `row_id`.

    It is deliberately NOT a random number generator: replaying real measured matrices means the
    CPU end-to-end run reproduces the real routed set, the real class counts and the real
    submission file, so the mock tests the pipeline's arithmetic against a known answer instead
    of only its syntax.
    """
    def __init__(self, profile, replay_by_row_id, tokenizer, boot_s=None, logprobs=48):
        self.profile_name = profile
        self.cfg = dict(PROFILES.get(profile, {}))
        self.replay = replay_by_row_id
        self._tok = tokenizer
        self.logprobs = logprobs
        self.boot_s = None
        self._fake_boot = boot_s if boot_s is not None else (
            self.cfg.get('measured_7500_s') and 28.5 or 28.5)
        self.attn_impl = ['<mock>']
        self._pending = None

    def boot(self):
        t0 = time.time()
        self.boot_s = time.time() - t0
        print('[L38] %-18s MOCK engine up in %.2fs (real measured boot for this profile: '
              '%s s warm / 118-142 s cold)' % (self.profile_name, self.boot_s,
                                               '28-31' if '31' in str(self.cfg.get('model'))
                                               else 'unmeasured'), flush=True)
        return self.boot_s

    def shutdown(self):
        return 0.0

    def check_headroom(self, prefix_tokens):
        return {'headroom': None, 'note': 'mock engine: no KV pool'}

    def get_tokenizer(self):
        return self._tok

    def set_rows(self, row_ids):
        self._pending = list(row_ids)

    def generate(self, prompts, sp):
        assert self._pending is not None and len(self._pending) == len(prompts), (
            'MockJudge.set_rows must be called with the row_ids of this batch, in order '
            '(%r rows staged, %d prompts)' % (
                None if self._pending is None else len(self._pending), len(prompts)))
        out = []
        for rid in self._pending:
            v = self.replay[rid]
            lp = {583: _LP(v[0]), 85531: _LP(v[1]), 69807: _LP(v[2])}
            out.append(_Req(lp))
        self._pending = None
        return out
