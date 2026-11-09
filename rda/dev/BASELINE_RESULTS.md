# nanochat FineWeb-EDU Baseline — Results

**Status: WORKING**

A known-good, reproducible single-GPU baseline on FineWeb-EDU, for later comparison against
architectural changes (affective state, memory, MoE, new objectives, ...).

**No architectural changes were made.** The Transformer, attention, MLP, normalization,
embeddings, positional encoding, output head, optimizer algorithm, and KV-cache are untouched.

---

## 1. Provenance

| Item | Value |
|---|---|
| Git commit | `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` |
| Commit subject | `clean up fragile code` |
| Commit date | 2026-07-03 22:54:57 +0000 |
| Python | 3.10.21 (uv venv at `.venv`; **not** 3.11 as assumed) |
| PyTorch | 2.9.1+cu128 |
| GPU | NVIDIA GeForce RTX 5060 Ti (8 GB), CUDA SM 12.0 (Blackwell) |
| Compute dtype | `torch.bfloat16` (auto-detected, bf16 supported) |
| Attention kernel | PyTorch SDPA (Flash Attention 3 unavailable on Blackwell) |
| `torch.compile` | **disabled** — Triton has no Windows wheel (see §6) |
| OS | Windows 11, PowerShell 5.1 |

Reproducibility: nanochat seeds globally in `compute_init` (`torch.manual_seed(42)`,
`torch.cuda.manual_seed(42)`), and every RNG used for sampling/gens uses explicit seeded
generators. Seeds were left at their repository defaults. Note `torch.use_deterministic_algorithms`
is commented out upstream, so cuDNN/SDPA kernel nondeterminism remains (same as stock nanochat).

---

## 2. Environment / dataset

| Item | Value |
|---|---|
| Dataset selected | **FineWeb-EDU-100B (shuffled)** |
| HF identifier | `karpathy/fineweb-edu-100b-shuffle` |
| Base URL | `https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main` |
| Local path | `%USERPROFILE%\.cache\nanochat\base_data_fineweb_edu` |
| Shards downloaded | **11 total = 10 train + 1 validation** |
| Train shards | `shard_00000.parquet` … `shard_00009.parquet` |
| Validation shard | `shard_00997.parquet` (last shard, always pinned as val) |
| Shard size | ~93–95 MB each (~1.0 GB total) |
| Total available | ~2.76B chars ≈ ~0.69B tokens (all 11 shards) |
| Parquet schema | `text: string` (single column; 1,024 rows/row-group) |

### Dataset identity verification (not ClimbMix)

The current nanochat mainline default is ClimbMix-400B, so the downloaded data was verified
cryptographically rather than assumed. Every shard's SHA-256 was compared against the HF
LFS OIDs of **both** repos:

```
shard_00000.parquet  sha256-match fineweb-edu=True  climbmix=False
shard_00001.parquet  sha256-match fineweb-edu=True  climbmix=False
...
shard_00997.parquet  sha256-match fineweb-edu=True  climbmix=False

ALL SHARDS AUTHENTIC FINEWEB-EDU: True
```

All 11 shards are byte-for-byte FineWeb-EDU. Spot-checked documents are edu-curated web text
(india.gov.in legal text, Tolkien fan fiction, etc.), consistent with FineWeb-EDU.

### Correct way to request FineWeb-EDU in the current code

The current `nanochat/dataset.py` hardcoded ClimbMix and no longer exposed FineWeb-EDU. The
minimal compatibility change added a dataset registry selected by env var / flag:

```powershell
$env:NANOCHAT_DATASET = "fineweb-edu"          # or: python -m nanochat.dataset --dataset fineweb-edu
```

`climbmix` remains the default, so stock behaviour is unchanged.

> **Note on shard count:** the historical nanochat FineWeb-EDU config used `MAX_SHARD = 1822`,
> but the `fineweb-edu-100b-shuffle` repo now ships **998** shards (last = `shard_00997`).
> The registry uses 997 as the validation-shard pin, matching today's repo, verified via the
> HF API.

---

## 3. Tokenizer

Used nanochat's own pipeline (`scripts/tok_train.py` → `RustBPETokenizer`). No HF `tokenizers`,
no tiktoken, no GPT-2, no SentencePiece.

| Item | Value |
|---|---|
| Command | `python -m scripts.tok_train --max-chars=500000000 --doc-cap=10000 --vocab-size=32768` |
| Vocab size | 32,768 (2^15) — unchanged from nanochat default |
| Max chars scanned | 500,000,000 (reduced from the 2B default, since the dataset is tiny) |
| Doc cap | 10,000 chars (default) |
| Merges learned | 32,503 |
| Training time | 14.13 s |
| Output | `%USERPROFILE%\.cache\nanochat\tokenizer\{tokenizer.pkl, token_bytes.pt}` |
| Round-trip test | **passed** (ASCII + CJK + emoji) |
| Special tokens | 9, incl. `<|bos|>` — standard nanochat set |

`token_bytes.pt`: shape `(32768,)`, dtype `int32`, sum 216,008 — required for BPB evaluation.

---

## 4. Model

Small dense baseline, sized for 8 GB VRAM, using nanochat's own scaling controls
(`--depth`, `--aspect-ratio`, `--head-dim`, `--window-pattern`). No MoE, no extra state.

| Item | Value |
|---|---|
| **Total parameters** | **207,618,378** (~208 M) |
| Depth (`n_layer`) | 8 |
| Model dim (`n_embd`) | 768 |
| Heads (`n_head` / `n_kv_head`) | 6 / 6 (full GQA, no GQA reduction) |
| Head dim | 128 |
| Context length | 1024 |
| `window_pattern` | `L` (full context — SDPA cannot accelerate sliding windows) |
| Vocab | 32,768 |
| MoE | none |

Parameter breakdown reported by nanochat:

```
wte                  :  25,165,824
value_embeds         : 100,663,296
lm_head              :  25,165,824
transformer_matrices :  56,623,392
scalars              :             42
total                : 207,618,378
```

Note `value_embeds` (ResFormer value embeddings) is 100.7M of the 207.6M — nearly half the
model is embeddings rather than matmul weights. This is nanochat's current architecture,
unchanged.

Effective FLOPs/token (nanochat estimate): `5.662e+08`.

---

## 5. Training

### Exact command

```powershell
cd C:\AI\nanochat
$env:NANOCHAT_DATASET  = "fineweb-edu"
$env:PYTHONIOENCODING  = "utf-8"      # required: banner is UTF-8, console is cp1252
$env:PYTHONUTF8        = "1"
.\.venv\Scripts\python.exe -m scripts.base_train `
  --model-tag=baseline-fwedu-d8 `
  --depth=8 --aspect-ratio=96 --head-dim=128 `
  --max-seq-len=1024 --window-pattern=L `
  --device-batch-size=4 --total-batch-size=32768 `
  --num-iterations=600 `
  --eval-every=50 --eval-tokens=524288 `
  --core-metric-every=-1 --sample-every=-1
```

### Configuration

| Item | Value |
|---|---|
| Objective | standard next-token prediction / causal LM (stock nanochat, unmodified) |
| Iterations | 600 |
| Device batch size | 4 |
| Micro-batch | 4 × 1024 = 4,096 tokens/rank |
| Gradient accumulation | 8 steps (computed by nanochat) |
| **Effective total batch** | **32,768 tokens** |
| Total training tokens | 19,660,800 (~19.7M) |
| Total training FLOPs | ~1.11e16 |
| LR warmup / warmdown | 40 steps / 65% |
| Matrix LR (Muon) | 0.02 × 0.1768 batch-scale = 0.003537 |
| Weight decay | 0.28 → 0.1385 (depth-8 scaled) |
| Wall clock | 10.60 min (600 steps @ ~1.07 s/step) |
| Throughput | ~30,000–31,000 tok/s |
| **Peak VRAM (`torch.cuda.max_memory_allocated`)** | **5,206.66 MiB (~5.1 GiB)** |
| `nvidia-smi` observed | 6,966 MiB, 97% util |

VRAM headroom is comfortable: 5.1 GiB peak against 8 GB, with no OOM at any point.

### Training loss

| Step | Smoothed train loss |
|---|---|
| 0 | 10.397894 |
| 200 | ~5.55 |
| 300 | ~5.14 |
| 450 | ~4.55 |
| 599 | 4.373696 |

Loss fell monotonically from **10.398 → 4.374**. (Initial 10.398 ≈ ln(32768) = 10.397,
i.e. the model started at chance for a uniform 32,768-way vocabulary — correct behaviour.)

### Validation BPB curve

Eval every 50 steps over 524,288 validation tokens (128 eval batches):

| Step | Val BPB |
|---|---|
| 0 | 3.144283 |
| 50 | 1.944599 |
| 100 | 1.809330 |
| 150 | 1.706102 |
| 200 | 1.609461 |
| 250 | 1.536990 |
| 300 | 1.480823 |
| 350 | 1.432675 |
| 400 | 1.390146 |
| 450 | 1.354029 |
| 500 | 1.328420 |
| 550 | 1.310699 |
| **600** | **1.300717** |

**Monotonic, no divergence, no overfitting signal** (only ~3.6% of one epoch consumed, so
val and train are effectively in-distribution). Final/min val BPB = **1.300717**.

---

## 6. Validation checklist

| # | Requirement | Result |
|---|---|---|
| 1 | FineWeb-EDU shards downloaded | **PASS** — 11/11 |
| 2 | Files are actually FineWeb-EDU, not ClimbMix | **PASS** — SHA-256 matches FineWeb-EDU repo, zero ClimbMix matches |
| 3 | Tokenizer training completes | **PASS** — 32,503 merges, round-trip OK |
| 4 | Small GPT instantiates | **PASS** — 207,618,378 params |
| 5 | CUDA detected | **PASS** — RTX 5060 Ti, SM 12.0, bf16 |
| 6 | Training runs on RTX 5060 Ti | **PASS** — 97% GPU util, ~30.6k tok/s |
| 7 | Loss decreases | **PASS** — 10.398 → 4.374 (train), 3.144 → 1.301 (val BPB) |
| 8 | Val loss/BPB evaluable | **PASS** — 13-point BPB curve, min 1.300717 |
| 9 | Checkpoint written and loaded | **PASS** — reloaded step 600, reproduced BPB 1.300717 exactly |
| 10 | No unrelated global Python packages modified | **PASS** — `uv.lock`/`pyproject.toml` untouched; no installs performed |

**Checkpoint round-trip detail:** after training, the model was reloaded from disk via
`load_model_from_dir(...)` and re-evaluated on the validation split. The reloaded model
returned `1.300717`, matching the training-time value to 6 decimal places.

**Generative smoke test** (greedy, 12 tokens) — the model produces well-formed English
structure; quality is low as expected after only 19.7M tokens (~0.03B model-token ratio):

```
'The capital of France is' -> '<|bos|>The capital of France is a city of the city of the city of the city of'
'Water boils at'           -> '<|bos|>Water boils at the bottom of the water. The water is a water source'
```

---

## 7. Files changed

Exactly three files, all minimal. No architecture, no optimizer math, no data-pipeline redesign.

### `nanochat/dataset.py` — enable FineWeb-EDU selection

- Replaced the hardcoded ClimbMix `BASE_URL`/`MAX_SHARD`/`DATA_DIR` constants with a small
  `DATASETS` registry keyed by name (`climbmix`, `fineweb-edu`), each with its own URL, last-shard
  index, and local directory.
- Added `NANOCHAT_DATASET` env var (source of truth) and a `--dataset` CLI flag alias.
- Set FineWeb-EDU's `max_shard` to **997** (current repo has 998 shards, not the historical 1822).
- Gave FineWeb-EDU its own directory `base_data_fineweb_edu` and pointed the legacy fallback at
  the original `base_data` name, so the two datasets coexist without clobbering each other.
- Replaced the stale "you must upgrade to ClimbMix" warning with a neutral "dataset directory
  not found" message, and added a clear `FileNotFoundError` if no data is present.
- The `__main__` block now prints the active dataset and source URL, and says how many are
  train vs. val shards.

*Why:* the current code has no way to request FineWeb-EDU; this is the smallest change that
restores the legacy path while keeping ClimbMix as the untouched default.

### `nanochat/optim.py` — Windows Triton fallback

- Added a `_triton_available()` probe and a `_maybe_compile` decorator, and replaced the two
  bare `@torch.compile(dynamic=False, fullgraph=True)` decorators on `adamw_step_fused` and
  `muon_step_fused` with it.

*Why:* `torch.compile`'s inductor backend requires Triton, which has **no official Windows
wheel**. These decorated functions are invoked every optimizer step, so on Windows they would
raise `TritonMissing` during training. The fallback returns the function undecorated, so the
exact same eager Python executes. **The optimizer algorithm, math, and dtype policy are
unchanged** — this is purely a performance/compat guard, and is a no-op on platforms with
Triton. Can be forced off with `NANOCHAT_COMPILE=0`.

### `scripts/base_train.py` — Windows Triton fallback for the model

- Reused `nanochat.optim.COMPILE_STEPS` to guard the top-level `torch.compile(model, ...)` call,
  falling back to the uncompiled model on Triton-less platforms.

*Why:* same root cause. Without this, `base_train.py` crashes at the first forward pass with
`TritonMissing`. `orig_model` is untouched, so checkpoint state-dict keys are unaffected
(they no longer carry the `_orig_mod.` prefix, and `load_model` already strips that prefix).

### Not modified (explicitly, per the baseline requirements)

`nanochat/gpt.py` (attention, MLP, block structure, RMSNorm, embeddings, rotary embeddings,
output head), the optimizer math in `nanochat/optim.py`, `nanochat/dataloader.py`,
`nanochat/loss_eval.py`, `nanochat/checkpoint_manager.py`, `nanochat/flash_attention.py`,
`nanochat/tokenizer.py`, `nanochat/engine.py`, `scripts/tok_train.py`, `uv.lock`,
`pyproject.toml`, and the virtualenv.

### Known Windows quirks (worked around, not code changes)

| Issue | Workaround |
|---|---|
| Console is cp1252; nanochat's banner is UTF-8 block art → `UnicodeEncodeError` | set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` |
| No Triton → `torch.compile` raises `TritonMissing` | eager fallback above (or `NANOCHAT_COMPILE=0`) |
| `pip` is not installed in the uv-managed venv | no packages were installed; nothing needed it |
| PowerShell 5.1 has no `&&` | used `;` separators |

---

## 8. Reproducing

```powershell
cd C:\AI\nanochat
$env:NANOCHAT_DATASET  = "fineweb-edu"
$env:PYTHONIOENCODING  = "utf-8"
$env:PYTHONUTF8        = "1"

# 1) data: 10 train shards + 1 val shard
.\.venv\Scripts\python.exe -m nanochat.dataset -n 10 -w 5

# 2) tokenizer
.\.venv\Scripts\python.exe -m scripts.tok_train --max-chars=500000000 --doc-cap=10000 --vocab-size=32768

# 3) train
.\.venv\Scripts\python.exe -m scripts.base_train `
  --model-tag=baseline-fwedu-d8 --depth=8 --aspect-ratio=96 --head-dim=128 `
  --max-seq-len=1024 --window-pattern=L --device-batch-size=4 --total-batch-size=32768 `
  --num-iterations=600 --eval-every=50 --eval-tokens=524288 `
  --core-metric-every=-1 --sample-every=-1
```

Artifacts: tokenizer in `%USERPROFILE%\.cache\nanochat\tokenizer\`, data in
`%USERPROFILE%\.cache\nanochat\base_data_fineweb_edu\`, checkpoint
(`model_000600.pt`, `optim_000600_rank0.pt`, `meta_000600.json`) in
`%USERPROFILE%\.cache\nanochat\base_checkpoints\baseline-fwedu-d8\`.
Full training log: `logs/baseline_train.log`.

---

## 9. Baseline conclusion

### **WORKING**

Data, tokenizer, model instantiation, CUDA execution, training, loss decrease, validation BPB,
and checkpoint save/load/re-evaluate all work on this Windows + RTX 5060 Ti setup against
genuine FineWeb-EDU data.

**Caveats to carry forward (none blocking):**

1. **`torch.compile` is inactive on Windows** (no Triton), so throughput is ~30.6k tok/s rather
   than what FA3+inductor would give. Baselines and variants must all be compared under the
   same eager regime, or the comparison is invalid.
2. **Flash Attention 3 is unavailable** on Blackwell; SDPA is used, which is why
   `--window-pattern L` is required (SDPA has no sliding-window fast path).
3. **MFU is reported as 0%** because nanochat has no peak-FLOPS table entry for this GPU name;
   it is a logging gap, not a training problem.
4. **This run is deliberately tiny** — 19.7M tokens on a 208M model (~0.09 tokens/param,
   ~3.6% of one epoch). Absolute numbers (val BPB 1.301) reflect a barely-trained model and
   are *only* meaningful as a relative comparison point against later variants run with the
   identical command.
5. Python is **3.10.21**, not the 3.11 stated in the brief.

The baseline is established. **Do not begin architectural work until the first variant is
scored against these exact numbers using this exact command with only the intended
architecture changes applied.**
