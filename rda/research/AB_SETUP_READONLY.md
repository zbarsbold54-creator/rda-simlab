# Phase 2 — B' A/B setup (READ ONLY, nothing launched)

Date: 2026-10-01. No command below was executed.

## Canonical commands (from logs/a_s42.log, logs/b_stab_s42.log COMMAND: line)

Common flags (`run_sweep.ps1` `$BASE_FLAGS`, identical in every A/B run):

    --depth=8 --aspect-ratio=96 --head-dim=128 --max-seq-len=1024
    --window-pattern=L --device-batch-size=4 --total-batch-size=32768
    --num-iterations=600 --eval-every=50 --eval-tokens=524288
    --core-metric-every=-1 --sample-every=-1

So d_model = 8 * 96 = 768, T = 1024. Toy scale, matching the probe.

Baseline arm A (no --rda):

    .\.venv\Scripts\python.exe -m scripts.base_train --model-tag a-s42 --seed 42 <BASE_FLAGS>
    ... --model-tag a-s43 --seed 43 <BASE_FLAGS>
    ... --model-tag a-s44 --seed 44 <BASE_FLAGS>

Treatment arm B' (identical, plus --rda):

    .\.venv\Scripts\python.exe -m scripts.base_train --model-tag b-s42 --seed 42 <BASE_FLAGS> --rda
    ... --model-tag b-s43 --seed 43 <BASE_FLAGS> --rda
    ... --model-tag b-s44 --seed 44 <BASE_FLAGS> --rda

Run order per `run_sweep.ps1` header: A 42,43,44 then B' 42,43,44.
Each run is preceded by `diagnostics/write_manifest.py --label <tag> --seed <n> [--rda] --num-iterations 600`.

## Step count and wall time

- `--num-iterations=600` for every arm; `--eval-every=50` -> 12 eval points.
- Measured: b-stab-s42 (B', seed 42) = **33.85 m**, peak 7445.49 MiB, "Total training time: 33.85m".
- Baseline A is the same architecture and step count; A and B' differ only by `--rda`,
  so ~34 min/run is the expectation for both arms. No per-arm A wall time is
  recoverable (see Ambiguities).

## Metric

- **Validation bits-per-byte (BPB)**, the quantity compared.
- Written by `base_train.py` to stdout as `Minimum validation bpb: <x>` at end of run
  (observed in logs/b_stab_s42.log).
- Independently reproducible via `scripts/base_eval.py --eval bpb --model-tag <tag> [--step N]`,
  which prints `{train,val} bpb: <x>` using
  `tokenizing_distributed_data_loader_bos_bestfit(..., split_name)` over splits
  ["train","val"]; val is the LAST parquet shard, and base_eval BPB uses split "val".

## Is the leak fix (969e067) in both arms?

**NO — and this is the blocking finding.** No B' A/B run has ever been executed on
post-`969e067` code.

Evidence:

- `b-stab-s42` ran at TREE: CLEAN, commit **`6ae93ce`**, which is the commit
  IMMEDIATELY BEFORE the fix. `969e067` ("Fix a one-token label leak in the RDA
  drive state") is dated 2026-09-29 20:24:33; `6ae93ce` is 2026-09-29 19:10:43.
- The 1.226305 BPB that the A/B would compare is from that pre-fix B' run. It is
  the already-WITHDRAWN number (E14); it measured the leak.
- The A baselines (a-s42/43/44) ran at commit `030a384` (2026-09-28), also pre-fix.
  This is harmless: those runs never passed `--rda`, and E14 established the
  baseline is unaffected by the leak. But they are at a DIFFERENT commit than any
  post-fix B' would be, so a re-run must decide whether that matters (it should
  not -- `969e067` touches only `nanochat/rda.py` -- but it is unverified here).

`run_sweep.ps1` now runs `diagnostics/test_no_label_leak.py` as a hard pre-flight
and refuses to start on failure, so any NEW sweep is protected.

## Baseline arm record at the same step count and split

- Step count: yes, 600 for both arms, same flags.
- Split: A used `--eval-tokens=524288`, and B' used the identical value.
- Seeds: baseline has 42/43/44; B' has only seed 42 (b-stab-s42, pre-fix).
  So there is NO post-fix baseline-vs-B' pair at any seed, and no post-fix B' at all.

## Ambiguities I cannot resolve from the repo

**RESOLVED 2026-10-01 — items 2 and 3 below were misdiagnoses. Both are now
known, and neither cause was the pipe.**

1. ~~**Which checkpoint the A/B evaluates.**~~ STILL OPEN as a definition, but
   `ab_v2` fixes it operationally: validation BPB at the FINAL step (600),
   parsed from `Step 00600 | Validation bpb:` (`base_train.py:740`), NOT
   `Minimum validation bpb` (line 1065, best checkpoint).
2. ~~**Baseline wall time unrecoverable / logs truncated.**~~ **WRONG. The
   baseline logs are NOT truncated and did NOT fail.** `a_s42.log`,
   `a_s43.log`, `a_s44.log` each completed600 steps and end with
   `Minimum validation bpb:` 1.301217 / 1.287544 / 1.295109, `Total training
   time:` 9.98 / 9.94 / 9.93 m, and a saved `optim_000600_rank0.pt`. They ran at
   commit `030a384`, which PREDATES `ab9b560` and therefore predates the
   NameError described below. Baseline wall time is ~10 m, not ~34 m: the B'
   arm's FiLM path is what makes it slow (b-stab-s42 = 33.85 m).
   **They appeared empty because the files are MIXED-ENCODING** — a UTF-8
   header written by `Set-Content -Encoding utf8`, then a UTF-16LE body
   appended by `Tee-Object`. Reading them as UTF-8 yields the header and
   nothing else. This is a THIRD defect, in `run_sweep.ps1`'s logging, not in
   training. Correct read: `[System.Text.Encoding]::Unicode.GetString(bytes)`.
3. ~~**"B' tag naming — three names for one run."~~ PARTLY WRONG.
   `b_stab_s42.log` and the `B'-s42` checkpoint are the SAME run (commit
   `6ae93ce`). `b_causal_s42.log` is a DIFFERENT, earlier attempt (commit
   `030a384`, dirty tree) that never produced a BPB. So there are two runs, not
   three names for one.
4. **Whether a-s42 should be re-run.** `manifests/a-s42-post.json` exists,
   written at commit `6ee7cf6` with a DIRTY tree that included
   `M scripts/base_train.py`. Its purpose is still undocumented.
5. **Eval batch size for the recorded BPB.** STILL OPEN. `--eval-every=50` uses
   `--device-batch-size=4`, whereas `base_eval.py` defaults to
   `--device-batch-size=32`. Two different code paths that need not agree. The
   A/B number's provenance is still not pinned.

## A LATENT BUG THAT MADE THE BASELINE ARM UNRUNNABLE (found 2026-10-01)

`scripts/base_train.py` referenced `_rda_diag_lr_scale` at line 941 — inside the
per-step LR rescale that runs for EVERY run — but defined it at line 680,
nested inside `if rda_active:` (line 536). Any run WITHOUT `--rda` therefore
died at the first optimizer step with
`NameError: name '_rda_diag_lr_scale' is not defined`.

Both halves came from ONE commit, `ab9b560` ("Stabilize the RDA state"), which
added the `RDA_STATE_LR` diagnostic hook and indented its default under the RDA
block while leaving the use unconditional.

It is invisible to the arm that was being run: `b-stab-s42` passed `--rda`, so
it executed fine. Every BASELINE run since `ab9b560` would have failed. The
`a-s42/43/44` logs predate `ab9b560`, which is why they are intact. Fixed by
initialising the value above the `if rda_active:` block; default stays 1.0, so
`--rda` behaviour is unchanged.
---

## Related Concepts

- [[BPB]] — the metric this setup verifies (final step, not best checkpoint)
- [[Label Leakage]] — the fix commit `969e067` that both arms must contain
- [[Control Arm]] — what the phase-2 B' arm is later compared against
- [[RDA and Persistent State]] — the B' configuration being set up
- [[Optimizer State]] — the latent bug that made the baseline arm unrunnable

The latent bug recorded here is a `NameError` from a definition nested inside a
conditional while its use was unconditional — invisible to any arm that took the
conditional branch, fatal to every arm that did not.
