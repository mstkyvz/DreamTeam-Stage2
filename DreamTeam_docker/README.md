# Stage 2 — three-class query/product relevance

One Gemma-4-31B INT8 judge, read out through a single forced token, on one A100-SXM4-40GB.

**Measured on 90,000 rows: macro-F1 0.9131 in 46.9 minutes** (limit: 150 minutes).

---

## 1. What the image does

For each `(query, product)` row the model is shown the rubric and a fixed few-shot block, the
assistant turn is prefilled with `{"etiket": "`, and **exactly one token is decoded** with
logprobs. The 3-vector is the marginal over vocabulary tokens that uniquely prefix exactly one
label. `max_tokens=1`: nothing is ever generated, there is no JSON to parse, and there is no
free-text answer that can be malformed.

That vector is the model's decision variable. Everything else in this image is arithmetic on it:
the label is `argmax` after a global IPF pass, the explainability view plots it directly, and the
field attributions are differences of it.

### Why one model and not the two-stage cascade

The development pipeline also supports a cheap MoE gate that scores every row, escalating only
the least-certain fraction to the 31B champion. It is a **wall-clock** optimisation and it costs
accuracy. All four points measured on the same 90,000 rows, same box, same read-out:

| configuration | macro-F1 | wall clock |
| --- | ---: | ---: |
| cascade @ 5% routed | 0.8741 | 24.8 min |
| cascade @ 10% routed | 0.8798 | 28.6 min |
| cascade @ 15% routed | 0.8903 | 29.4 min |
| cascade @ 20% routed | ≤ 0.8935 | 32.8 min |
| **champion-only (shipped)** | **0.9131** | **46.9 min** |

The cascade is faster and worse at every fraction tried, and only draws level around 30% routing.
Since the organizer allows 150 minutes and this finishes in 46.9, there is nothing to buy with
the time it saves. **So the gate is not in this image**: not downloaded, not shipped, one
checkpoint instead of two.

Why the fraction is the wrong lever anyway: the gate scores *every* row at every fraction, so it
is ~62% of the cascade's wall clock no matter how little it escalates. Each 5 points of routing
costs ~2.5 min and buys a shrinking fraction of a point — the flip rate on newly-routed rows
decays ~0.79× per step (375.6 → 346.4 → 274.0 flips per 1,000 newly-routed rows).

The cascade code still ships and is still tested — it is the same orchestrator with a gate stage
wired in. `DT_FRAC=0.15` selects it, and `wiring.build_engine_factories` then refuses with an
explicit error because the gate checkpoint is not present, rather than attempting a download on a
network-disabled run.

---

## 2. Architecture

Ports and adapters. The contract forces a single image, so this is not microservices in the
process sense — but every capability is an independently testable service behind a declared
interface, and the composition root is the only thing that knows both sides.

```
src/
  main.py                  contract entry: main.py INPUT.csv OUTPUT.csv
  app/
    contracts.py           Protocols (InferencePort, RouterPort, ReadoutPort) + crossing data
    config.py              every knob, each with the measurement it came from
    errors.py              typed failures; exit_code is part of the type
    wiring.py              composition root — the only module that imports adapters
  services/                business logic, importable with no vLLM present
    cascade_service.py     orchestration: gate -> route -> champion -> compose -> read out
    routing_service.py     margin router
    readout_service.py     IPF against a soft class target
    io_service.py          CSV contract in and out, atomic write
    explain_service.py     occlusion attribution over the predictor
  adapters/
    vllm_engine.py         InferencePort over vLLM (engine lifecycle, boot timeout, probes)
    mock_engine.py         InferencePort on CPU, deterministic — what the tests run against
  api/
    server.py, ui.py       explainability HTTP service and its self-contained page
  core/                    VENDORED, UNMODIFIED: the measured numerical path
```

`services/` never imports `adapters/`. That is what lets the whole orchestration run on a laptop
with no GPU, which is what `smoke_test.sh` and the 15 unit tests do.

`core/` is vendored deliberately and is not refactored. It is the code every reported number was
measured through — the prompt construction, the forced-token read-out, the IPF, the engine
profiles. Rewriting it behind a nicer interface would put the measured numbers and the shipped
numbers on two different code paths, which is the one thing a submission must not do.

### Tests

```bash
python3 -m pytest tests/ -q     # 15 tests, ~0.5 s, no GPU
./smoke_test.sh                 # main.py end to end against the mock engine
```

They cover what silently breaks a submission rather than crashing it: champion answers landing on
the wrong rows, engines leaking on a failed stage (a leaked engine starves the next boot),
`row_id` `007` renumbered to `7`, quoted multiline fields, truncated output, non-deterministic
routing.

---

## 3. Build and run

```bash
./fetch_models.sh                                   # once; needs HF_TOKEN and network (~32 GB)
CONTAINER_NAME="DreamTeam_solution" ./build_docker.sh

CONTAINER_NAME="DreamTeam_solution" \
  ./run_solution.sh /abs/path/private_input.csv /abs/path/submission.csv

NGROK_AUTHTOKEN="..." CONTAINER_NAME="DreamTeam_solution" \
  ./run_explainability.sh /abs/path/public_test.csv
```

`run_solution.sh` and `run_explainability.sh` are the organizer's files, byte for byte unmodified.

The download is a separate step on purpose: build-time internet is permitted, but a build that
pulls 32 GB can fail at minute 55 of a 60-minute limit and take the submission with it. With the
weights already in the context the build is a local `COPY` — deterministic, offline, repeatable.

`build_docker.sh` deliberately does **not** prune the Docker daemon. The reference implementation
does, and the rules state that global Docker cleanup is grounds for rejection *even when it
appears in a reference implementation*.

---

## 4. Reproducibility record

| item | value |
| --- | --- |
| Image name | `DreamTeam_solution` |
| Base image | `nvidia/cuda:13.0.0-devel-ubuntu24.04` (`devel`, not `runtime` — Triton's runtime JIT needs ptxas/nvcc) |
| Serving stack | `vllm==0.28.0` (pulls `torch 2.13.0+cu130`), `flashinfer-python==0.6.16.post3` |
| Model | `lokeshe09/gemma-4-31B-it-INT8` — the only checkpoint in the image |
| Precision | INT8 W8A8 — native on SM80 tensor cores. FP8 is **not**: SM80 has no FP8 units, so an FP8 checkpoint is demoted to Marlin W8A16 and runs slower |
| Size on disk | 30.63 GiB of weights, ~32 GB directory |
| Attention backend | FlashInfer, verified by live probe (`FlashInferImpl`) and by the engine log line |
| `max_model_len` | 5250 — longest observed public_test prompt is 5,178 tokens |
| `gpu_memory_utilization` | 0.94. Not free: at 0.97 FlashInfer's workspace leaves the sampler no room and the engine OOMs during warm-up |
| Batching | `max_num_seqs=256`, `max_num_batched_tokens=4096`, prefix caching on, `enforce_eager=True` |
| Decoding | greedy, `temperature=0`, `max_tokens=1`, `logprobs=48`, `seed=0` |
| Read-out | IPF against a soft class target, λ=0.5 |
| Resume behaviour | none. The run is 47 minutes against a 150-minute limit, and a checkpoint file on a read-only rootfs is a liability rather than insurance |

### Runtime and resources, measured on 90,000 rows

| stage | seconds |
| --- | ---: |
| engine boot | 45.1 |
| tokenisation (90,000 prompts) | 27.0 |
| generate | 2,732.3 |
| teardown | 8.4 |
| **total** | **2,812.8 (46.9 min)** |

33.0 rows/s. VRAM: 0.94 of 40 GiB, KV pool 2,194 blocks × 16 tokens = 35,104 tokens, 7.08×
headroom over the longest prompt. Host RAM stays far below the 70 GB cap; peak is during
safetensors load and is dominated by page cache, which is reclaimable.

> Note on the vLLM log: its `GPU KV cache size` line counts one cache group and **reads ~5.5×
> low** on this model's heterogeneous attention layers (50 sliding-window + 10 full). Four
> independent runs reproduced the ratio. Every KV figure above is from a live probe of the
> running engine, not from that line.

### Tokenisation

Naive per-row encoding of 90,000 prompts costs 588 s single-threaded. Two changes take it to
27.0 s: the shared prefix is tokenised once and the per-row body spliced onto it, and
`TOKENIZERS_PARALLELISM` is re-enabled (transformers pins it to `false` on import, idling every
core but one). **Verified bit-identical on 90,000/90,000 rows, 0 mismatches**, before any run was
scored. `encode_spliced` re-encodes a sample the original way and falls back to the plain loop on
any mismatch, so a BPE boundary that does not splice safely degrades to correct-and-slow.

### External models and data used in development

| asset | source | licence | purpose |
| --- | --- | --- | --- |
| `lokeshe09/gemma-4-31B-it-INT8` | Hugging Face | Gemma Terms of Use | the shipped judge |
| `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` | Hugging Face | Gemma Terms of Use | cascade gate — measured during development, **not shipped** |
| Stage 1 + Stage 2 organizer files | competition | competition terms | training, few-shot draw, evaluation |

The few-shot block is a deterministic draw (seed 4) from `train.parquet`, shipped rendered as
`src/assets/shots_gfs4.txt` with its SHA so the image needs no 500,000-row training file at
inference. A silently different shot block is a silently different model, so a missing asset is
fatal rather than defaulted.

---

## 5. Explainability

`run_explainability.sh` starts a local FastAPI UI and an ngrok tunnel in the same container. The
UI shows, for any `row_id` and any query the reviewer types:

1. **The posterior** — the predictor's actual 3-vector, the one argmax is taken over. Not a
   confidence score computed afterwards.
2. **Field attribution by occlusion** — each field is blanked in turn and the row is re-scored
   through the identical path; the change in log-posterior is that field's contribution. A
   Shapley-style attribution with single-feature coalitions, exact for the model as deployed
   rather than estimated from a surrogate. Cost is one `max_tokens=1` prefill per field, and the
   variants share the entire system prompt so the prefix cache serves all of them.
3. **The routing margin** — the gap between the top two log-probabilities, which is the number
   the cascade's router sorts on. It answers "would the big model have been called for this row".
4. **What is not reproduced per row** — IPF is a global operation over all 90,000 rows. The UI
   shows the raw posterior and names the read-out instead of inventing a per-row adjusted number.

This is the predictor explaining itself, not a second model narrating it. There is no zero-shot
explainer anywhere in the image.

The UI boots the **champion**, which is the model that writes every label in the shipped
configuration. Startup polls `/health` for up to 8 minutes, inside the organizer's 10-minute
limit; `/health` fails closed until the engine is actually up, so the tunnel can never open in
front of a dead model. The page is one self-contained document — no CDN, no external font — so it
renders behind an outbound firewall.

---

## 6. Running under the organizer's constraints

`run_solution.sh` runs the image `--read-only`, `--network none`, `--cap-drop ALL`, as an
arbitrary host uid with an 8 GB `/tmp` tmpfs. Each of these breaks something naive:

- **Read-only rootfs.** vLLM, Triton, torch inductor and HF all write caches under `$HOME` by
  default. Every one is redirected into `/tmp` by `ENV` in the Dockerfile.
- **Arbitrary uid with no account.** `os.path.expanduser()` runs during import in several
  libraries and fails on a missing home, so `entrypoint.sh` creates `$HOME` before exec'ing
  Python. Model files are `--chmod=755` at COPY time, world-readable, in one layer.
- **No network.** `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`,
  `VLLM_NO_USAGE_STATS`, `DO_NOT_TRACK` are all set. Weights are baked; nothing is fetched.
- **No swap.** `--memory-swap` equals `--memory`, so the 70 GB cap is hard.
- **Artifacts.** The run manifest goes to `/tmp`, never next to the submission — the judge's
  output directory receives exactly one file.

No build secret is used. The checkpoint is vendored into the build context before `docker build`
starts, so there is no token to persist in a layer, an environment variable, or the image history.

---

## 7. Known limitations

- **The learned router is not in this image.** It scores AUC 0.8892 against the shipped margin
  router's 0.8184, but its features read two encoder checkpoints (`tyonly_cal`, `s1_ty`) that
  were never saved — only their `public_test` outputs survive, so it cannot be evaluated on
  private rows at all. Every cascade number above uses margin routing and is therefore a lower
  bound. The shipped champion-only configuration does not route, so this does not affect it.
- **The gate is batch-nondeterministic.** Two identical cascade runs reproduce only 79.2% of each
  other's top-10% hardest set, worth up to ±0.01 macro-F1 of replicate slack. Champion-only has
  no routing step and does not carry this.
- **Public macro-F1 from the built image is not yet recorded here.** The 0.9131 above is from the
  same code and configuration on 90,000 held-out rows, measured on an A100-SXM4-40GB, but the
  build-and-run-on-`public_test.csv` step has not been executed at the time of writing. It must
  be, and the number and output checksum recorded, before submission:

  ```bash
  CONTAINER_NAME="DreamTeam_solution" ./run_solution.sh $PWD/public_test.csv $PWD/public_pred.csv
  python evaluate.py public_pred.csv --solution public_solution.csv
  md5sum public_pred.csv
  ```

- **`enforce_eager=True` is deliberate.** CUDA graphs help decode-heavy workloads; this one is
  pure prefill with `max_tokens=1`, where eager measured 0.4% faster and boots sooner.
- **Forcing FlashInfer on the MoE gate is harmful** — 1.56× slower than Triton (1,895 s vs
  1,218 s). It helps the dense INT8 champion only. This matters only if the cascade is re-enabled.
