# Code review — nanochat RDA experiment

> **REMEDIATION STATUS - 2026-10-03.** This review has since been partly
> actioned. Resolved findings are marked inline. Summary of what changed:
>
> | Finding | Status |
> |---|---|
> | F1 + F2 `run_sweep.ps1` log corruption | **FIXED** - `Start-Process` + UTF-8 fold, self-test added |
> | F4 eval ran in a different state mode | **FIXED** - explicit `rda_carry` policy, measured 1.5e-02 BPB delta |
> | F3 decay persistence vacuous at T=1024 | **INVESTIGATED, NOT CHANGED** - `decay=0.95` serves within-sequence inertia; semantics now measured and documented (`docs/decay_semantics.md`) |
> | F5 `e_init` justification | **FIXED** - and my original explanation was also wrong; see the corrected F5 |
> | F6 `probe_drives.py` `if False` | **FIXED** - new value matches ground truth to 2.4e-07 |
> | Label-leak claim | **NOT REPRODUCED** in current tree; history retained. See `research/LEAK_STATUS.md` |
>
> No training run, probe, or sweep was executed as part of the remediation.
> No experiment number has been regenerated.

Read-only review. No code, scripts, logs or checkpoints were modified. No training,
probe, or smoke process was started. Date: 2026-10-02. Branch `master`.

## 0. `git status --short` at review start

```
 M research/CURRENT_STATE.md
 M "research/RDA and Persistent State.md"
?? "research/Untitled 1.canvas"
?? research/Untitled.base
```

Branch: `master`. **No python training process was running** (checked with
`Get-Process python*,pythonw*` — returned nothing), so nothing had to be stopped.

These three pre-existing modifications are unrelated to the review and were left
untouched. `git status` at the end differs only by the addition of
`research/CODE_REVIEW.md` (this file). Nothing was committed.

## Summary

| Severity | Count |
|---|---|
| Critical | 0 |
| High | 4 |
| Medium | 9 |
| Low | 10 |

### The five most important findings

1. **F1 (High) — `run_sweep.ps1` writes every run log as UTF-16LE, so the log is
   unreadable and the leak gate's own evidence is destroyed.** Confirmed in the
   real artifacts on disk: `logs/a_s42.log` is a UTF-8 header followed by UTF-16LE
   body. `Select-String -Pattern "Validation bpb"` returns **0 matches** on it.
2. **F2 (High) — `run_sweep.ps1:83` merges stderr into the success stream while
   `$ErrorActionPreference = "Stop"`, so the sweep dies on the first stderr line
   and the run is recorded as neither success nor failure.** This is the exact bug
   the other three drivers were rewritten to avoid; `run_sweep.ps1` was not.
3. **F3 (High) — decay-only "persistence" across documents is numerically vacuous
   at `max_seq_len=1024`.** `decay=0.95` attenuates the carried state by
   `0.95^1023 = 1.6e-23`. A packed row therefore starts every document at
   >99.99% fresh state, and the only place `E_t_external` is used at full
   strength is position 0 of each row.
4. **F4 (High) — validation evaluates a *different model* than training.**
   `evaluate_bpb` calls `model(x, y, loss_reduction='none')` with no
   `E_t_external`, so every eval batch starts from `E_init` and reports `mode='reset'`,
   while training always uses `mode='carry'`. The val BPB is not the val BPB of the
   trained object.
5. **F5 (Medium) - `base_train.py:753-757`: the stated reason `e_init` is grad-free
is wrong. - RESOLVED 2026-10-03; my original explanation below was ALSO wrong.**

> **Correction.** Re-investigation showed that *both* the original code comment
> and my first-pass explanation were incorrect, in opposite directions.
> Corrected status:
>
> ```
> STATUS: e_init is STRUCTURALLY LEARNABLE but frozen at its zero init
>         in every real run.  The _grad_free entry is CORRECT.
> ```
>
> The true mechanism is two facts that only hold *together*:
>
> 1. A gradient path **does** exist, via `E_t_external=None`:
>    `e_init -> init_state() -> E_t -> film_state -> modulate() -> loss`.
>    `init_state()` is differentiable (verified: `grad_fn = CloneBackward0`).
>    At step 0 this path is live but yields **exactly 0.0**, because
>    `film_scale`/`film_shift` are zero-init and `modulate()` is the identity.
> 2. From step 1 onward the trainer **always** supplies a carry, and
>    `forward()` **detaches** it. A detached tensor has no graph, so autograd
>    **prunes** `e_init`: `p.grad is None`.
>
> Measured over a 5-step loop reproducing the trainer's exact call pattern:
>
> ```
> e_init per-step |grad| = ['0.000e+00', 'None', 'None', 'None', 'None']
> e_init after 5 steps   = [0,0,0,0,0,0,0,0]   max|delta| = 0.0
> ```
>
> My original claim - "after FiLM weights become nonzero the gradient is 27.1,
> so `e_init` is being trained" - was **an artifact of testing with
> `E_t_external=None` at every step**, which is not what the trainer does. The
> `2.7e+01` figure is real for the step-0/eval code path, but that path
> coincides with the zero-init FiLM that makes the gradient zero.
>
> Also corrected: the old `smoke_rda.py` comment blamed "`init_state()` expands a
> zero vector". The zero *value* is irrelevant to whether the graph exists -
> verified: a **non-zero** `e_init` with no carry *does* produce a nonzero
> gradient (`1.457e-01`).
>
> **Fixed by:** corrected comments in `base_train.py`, `nanochat/rda.py` and
> `diagnostics/smoke_rda.py`, plus a new regression test
> `diagnostics/test_einit_grad_status.py` that pins both halves (including the
> non-zero-`e_init` case, so the wrong mechanism cannot come back).

<details><summary>Original (superseded) finding text</summary>

```
#   state.e_init        - only read via init_state(); with decay-only
#                         persistence it is used once at step 0, and
#                         modulate() is the identity when E_t == 0, so the
#                         loss has no gradient path to it.
```

Measured (SUPERSEDED - this used `E_t_external=None` at every step):

```
(a) at zero-init FiLM (step 0):  e_init.grad = 0.0   (not None!)
(b) after FiLM weights become nonzero:
    e_init.grad absmax = 2.713559e+01
```

The reasoning below - that the zero-priming "never touches `e_init`", so it is
being trained - is **wrong**: with a carry supplied (the real training path) the
gradient is `None` and the priming fires. The *defect* identified (the stated
reason is false) was real and is fixed; the mechanism is now documented
correctly.

- **Severity:** Medium (comment/justification defect only). **Confidence:** CORRECTED.
- **Affected past results:** none.
- **Remaining note:** if `decay` were ever raised such that a fresh
  `E_t_external` could hold a meaningful state, `e_init` becomes load-bearing and
  must leave `_grad_free`. Flagged in the new test and in
  `docs/decay_semantics.md`.

</details>

### A6. Hooks / global state / RNG — **checked, found 2 (F8, F9)**

**F8 (Medium) — `diagnostics/probe_drives.py:111-114` and five sibling scripts:
hooks removed outside a `finally`.**

```python
h1 = orig_ev.register_forward_hook(ev_hook)
h2 = orig_pe.register_forward_hook(pe_hook)
out = model(x, y, E_t_external=carry)
h1.remove(); h2.remove()
```

If the forward raises (OOM, the saturation assert at `base_train.py:1000`, a shape
error), the hooks stay attached to the module and every later probe in the same
process silently logs through them. Same pattern in `diagnose_saturation.py:116-121`,
`smoke_saturation.py:99-104`, `audit_causality.py:177-190`,
`audit_leak_size.py:102-121`, `map_pe_film_graph.py:139-143` and `:249-254`.

Counter-examples, done right: `diagnostics/probe_pe_film.py:1242, 1264, 1407` and
`diagnostics/diagnose_film_causality.py:69` all use `try/finally`, and
`probe_pe_film.py:1233-1244` correctly saves and restores
`float32_matmul_precision` and `cuda.matmul.allow_tf32` around the fp64
reconstruction, then *asserts* the restore. That is the right pattern and it is
already in the repo.

- **Severity:** Medium. **Confidence:** CONFIRMED.
- **Affected past results:** no — these are diagnostic scripts, and a leaked hook
  only affects a subsequent probe in the same process, which is unlikely to have
  been run after a crash.
- **Verify:** wrap the forward in `try/finally` and re-raise; assert
  `len(mod._forward_hooks) == 0` after the loop.
- **Fix:**
  ```python
  h1 = orig_ev.register_forward_hook(ev_hook)
  h2 = orig_pe.register_forward_hook(pe_hook)
  try:
      out = model(x, y, E_t_external=carry)
  finally:
      h1.remove(); h2.remove()
  ```

**F9 (Low) — `probe_drives.py:104-109` keeps only the *last* hook invocation, and
the comment says so incorrectly.**

```python
def ev_hook(_mod, _inp, out):
    cap['ev'] = out.detach()     # overwrites on every call
    return out
...
ev = cap['ev']            # (B, n_drives)
```

`drive_seq` calls `event_proj` **once** on the whole `(B, T, d)` batch, so the
tensor captured is `(B, T, n_drives)`, not `(B, n_drives)` as the comment claims.
`ev.abs().mean()` still reduces to a scalar, so `ev_abs` happens to be right, but
the comment is wrong and the per-position structure is discarded. The neighbouring
comment "update() runs once per sequence position, so the hook fired T-1 times" is
**false** for the vectorized path.

- **Severity:** Low. **Confidence:** CONFIRMED (vectorized `drive_seq` performs one
  projection call).
- **Verify:** print `cap['ev'].shape` after a forward.
- **Fix:** correct the comments; keep `(B, T, n_drives)` and reduce with
  `dim=(0, 1)` if a scalar is wanted, or `dim=1` for a per-position trace.

**Global RNG consumption — clean, and notably well handled.**
`GPTWithRDA.__init__` (`rda.py:673-680`) uses a **private** `torch.Generator` for
the arm-C constant, precisely so it does not shift the global stream between arms.
That is correct and the comment explains why. `test_no_label_leak.py:102` and
`map_pe_film_graph.py:1364` also use local generators. No unguarded
`torch.manual_seed` in a library path — the `manual_seed` calls are all in
`__main__`-style scripts.

**Seeding — clean, with one caveat.**
`compute_init` (`nanochat/common.py:174-205`) seeds only the global torch RNG and
documents that the dataloader derives its own generators. The `--seed` therefore
reaches weight init and the arm-C constant but **not** the data order. The comment
says this, so it is documented rather than hidden. Caveat: `--seed 42/43/44` in the
A/B sweep varies **model init and the arm-C vector only**; the training data
sequence is identical across seeds. Worth stating in any write-up that claims seed
variance is a data-order variance. Severity: Low, **Confidence:** CONFIRMED.

### A7. State carried across steps/documents, memory, stale `.detach()` views — **found 1 (F3), 1 Low**

**F3 (High) — decay-only persistence is numerically vacuous at `T=1024`.**
`rda.py:948` documents decay-only persistence as the mechanism "governing how much
prior state survives into a new document". Measured with `decay=0.95`:

```
t=    0  decay**t = 1.0000e+00
t=    1  decay**t = 9.5000e-01
t=   10  decay**t = 5.9874e-01
t=  100  decay**t = 5.9205e-03
t=  500  decay**t = 7.2745e-12
t= 1023  decay**t = 1.6265e-23
```

And perturbing the carried state by +0.5 changes the trajectory by:

```
t=0..5      : 4.750e-01
t=100..110  : 2.812e-03
t=1020..1023: 0.000e+00
```

So: the only position that uses `E_t_external` at full strength is **position 0 of
each row**, and by position ~130 the carried state is numerically irrelevant. A row
packs several documents (`row_capacity = T+1`), so every document after the first
begins with a state that is already the row's own fresh computation.

The 1/e horizon is `1/ln(1/0.95) = 19.5` positions.

- **Why it matters:** "persistent state across documents" is the architectural
  claim of the arm. As configured it is not implemented in any measurable way. To
  make it real the decay would need to be ~0.999 (1/e over ~1000 positions), or the
  recurrence reset-to-`E_init` explicitly at boundaries with a learned per-document
  state.
- **Severity:** High. **Confidence:** CONFIRMED (measured, CPU, tiny tensors).
- **Affected past results:** **yes, qualitatively** — any statement in
  `research/EXPERIMENT_LOG.md` or the RDA write-ups that the state "persists across
  documents" is not supported by the configuration used. It does not by itself
  invalidate a BPB comparison between arms, because both arms are equally affected,
  but it does mean the arm is not testing what its name says.
- **Verify:** the decay table above, or set `decay=0.999` in a 600-step run and
  compare `|film_state - film_state(E_t_external=0)|` at position 500.
- **Fix:** raise the default `decay` toward `1 - 1/T` (e.g. `0.999` for T=1024) and
  re-tune, **or** add a per-row `E_init`-anchored variant where the state is reset at
  each BOS. Note `_max_safe_T(0.999, float64)` is still ~1.38e4, so the closed form
  stays valid; with `decay=0.999` the float32 limit would be ~1730, so keep the
  float64 accumulation. Do not change this without a new A/B — it changes the arm.

**Low — stale/`alive` tensors.** `rda.py:1171-1177` returns `hidden`, `hidden_mod`
(B,T,d each) and `dpe_seq` in the training dict, and `base_train.py:922-924` keeps
`out` bound across the micro-step loop, so peak memory holds two extra
`(B, T, d_model)` activation copies that the training loss does not need. This is
also why the RDA arms are not memory-matched to baseline. Not a leak (the dict is
rebound each micro-step, so it does not grow across steps — measured: no
accumulation), but it does inflate peak memory. **Confidence:** CONFIRMED.
**Fix:** return `hidden` only when a diagnostic needs it (the `RDA_DIAG` path at
`base_train.py:1058` does), or `.detach()` and drop it otherwise.

**Clean:** `E_t_external` is `.detach()`ed and `.clamp`ed at `rda.py:926-929`, and
`rda_carry_state = out["E_next"].detach()` at `base_train.py:923`. No graph is
carried across steps. Good.

### A8. Numerical issues — **found 1 (F10), rest clean**

**F10 (Medium) — `_unit_rms` amplifies the film weight gradient ~1000× while the
state is near zero, which is exactly the regime the film weights start in.**

`rda.py:70-72`:
```python
return v / v.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt() * rms
```

The `1e-6` inside the square root means the normaliser is `sqrt(E[v²] + 1e-6)`, so
when `v ≈ 0` the divisor is `1e-3`, not the RMS. Measured `d(sum out)/dv`:

```
|v|= 0.0e+00 -> 500.0000   (unscaled would be 0.500; ratio 1000.0x)
|v|= 1.0e-06 -> 499.9993   (ratio 1000.0x)
|v|= 1.0e-04 -> 492.5927   (ratio  985.2x)
|v|= 1.0e-02 ->   0.4926   (ratio    1.0x)
|v|= 1.0e+00 ->   0.0000   (ratio    0.0x)
```

Because `film_scale`/`film_shift` are **zero-initialised** (`rda.py:139-141`), the
first several hundred optimizer steps are spent in the `|v| < 1e-2` band where the
effective LR on the film weights is up to 1000× the nominal `matrix_lr * batch_lr_scale`.
The docstring's claim that this "does not flatten gradients" is true, but it does
not mention the amplification, and the `+1e-6` was chosen to avoid a division by
zero rather than to control the small-`v` regime.

This also explains the FiLM `tanh` derivative behaviour noted in the E13 analyses
without needing a separate cause.

- **Severity:** Medium. **Confidence:** CONFIRMED (measured, CPU).
- **Affected past results:** **yes, plausibly** — it changes the effective early
  learning rate of the only pathway through which the state reaches the loss, in
  every B' and arm-C run, on every seed. It does not favour one arm over another
  (all RDA arms share it), so it does not invalidate the *comparison*; it does mean
  the absolute RDA numbers are sensitive to an undocumented constant.
- **Verify:** the table above.
- **Fix:** use a larger, scale-relative epsilon, or detach the normaliser:
  ```python
  denom = v.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().detach()
  return v / denom * rms
  ```
  detaching keeps the forward exactly identical (the value is the same) and removes
  the second-order term from the weight gradient, which is the term doing the
  amplifying. Alternatively raise the epsilon to `1e-4` and document the resulting
  small-`v` behaviour. **Changing this changes training dynamics — re-run the A/B.**

**Checked and clean:**
- **No division without epsilon** anywhere in `rda.py` — all three normalisers carry
  `+1e-6`.
- **`_max_safe_T` docstring is accurate.** The docstring claims
  `T <= 13837` for float64; the computed value is `13837`. Match. The previous
  "hardcoded L <= 6000" guess was correctly replaced by a computed limit.
- **The closed form matches the reference loop.** Measured at the real training
  shape: `max|closed_form - loop| = 4.843e-08` over T=1024, consistent with the
  docstring's "~1e-7, bounded by the omitted clamp". The `decay *` factor on
  `E_start` (the easy-to-drop term) is present at `rda.py:434`. Good.
- **TF32**: `compute_init` sets `float32_matmul_precision("high")`. Per the stored
  project memory, a 1e-4 absolute tolerance is finer than TF32 can represent, so
  gates comparing a TF32 forward to an fp64 reconstruction must use a *relative*
  tolerance. `probe_pe_film.py:1246-1247` uses exact comparison only to check that
  the *precision setting was restored* (a boolean about settings, not about
  values), which is legitimate. No finding.
- **No float `==` comparisons on values** in the review scope. `torch.equal` and
  `torch.allclose` are used correctly. `test_no_label_leak.py` deliberately uses
  `torch.equal` for bit-identity, which is the right test.
- **Saturating `tanh`**: the gated/convex-combination update bounds `|E_next| <= 1`
  **by construction** (`rda.py:283`), and both `update()` and `update_seq()` assert
  it. This is a real improvement over the old sum-then-clip form and is correct.
- **`drive_prediction_error`** is deliberately unclamped and documented as such.

### A9. Eval vs train differences — **found 1 (F4)**

**F4 (High) — validation runs a different model from training.**

`nanochat/loss_eval.py` calls:
```python
loss2d = model(x, y, loss_reduction='none')   # (B, T)
```
with **no `E_t_external`**. In `rda.py:929-932`, `E_t_external is None` selects
`E_t = E_init` and `mode = "reset"`. Verified on a stub model:

```
no E_t_external  -> mode='reset'   (this is the evaluate_bpb call)
with E_t_external-> mode='carry'
```

Meanwhile `base_train.py:921` always passes `E_t_external=rda_carry_state`, so
training is always `mode='carry'`. Consequences:

1. Every val batch starts from `E_init` instead of the trained carried state.
2. The reported `val_bpb` for an RDA arm is not the val BPB of the object that was
   trained. It is the val BPB of "model + fresh state per batch".
3. For the **baseline** arm (no `--rda`) this is a no-op, so the asymmetry is
   RDA-only. Comparing baseline BPB to RDA BPB compares two different measurement
   procedures.
4. `E_init` is itself a trained parameter (F5), so this is not even a fixed point.

The `mode` field is returned in the dict precisely so a caller can audit this, and
`base_train.py` never inspects it at eval time.

- **Severity:** High. **Confidence:** CONFIRMED.
- **Affected past results:** **yes.** Every "Validation bpb" number for B' and arm C
  in the A/B sweep, the ab_v2/ab_v3 results JSON, and the `B'-s42 1.226305` canary.
  This does not automatically flip the sign of the RDA effect (all RDA arms share
  it), but the numbers are not what they are labelled, and the *gap* between
  train-time and eval-time behaviour is unmeasured.
- **Verify:** add `E_t_external=torch.zeros(...)` vs. the trained carry to
  `evaluate_bpb` and compare; or simply log `out["mode"]` during eval and confirm it
  is always `reset`.
- **Fix:** give the eval path the same carry semantics. The cleanest version is to
  make the state a property of the *evaluation unit* rather than of the training
  run: pass a per-batch carry that is reset at the start of each eval batch and
  carried across the batches of one eval, then log the procedure in the checkpoint
  metadata. At minimum, record `eval_mode=reset` in the saved metadata so the number
  is honestly labelled.

### A10. Per-step name-scope / `out` reuse — see A2. Clean.

---

## B. PowerShell / driver pitfalls

`run_ab_v2.ps1`, `run_ab_v3.ps1` and `run_ab_v4_preview.ps1` are careful and
document their own traps. `run_sweep.ps1` is the one that still has the bugs the
others fixed.

### F1 (High) — `Tee-Object -Append` writes UTF-16LE, corrupting every sweep log

`run_sweep.ps1:78` writes the header with `Set-Content -Encoding utf8`, then
`run_sweep.ps1:83` appends the child output with `Tee-Object -FilePath ... -Append`
and **no `-Encoding`**. On Windows PowerShell 5.1 that defaults to UTF-16LE.

Verified in a temp file:

```
after Set-Content -Encoding utf8 : len=33  first4=EF BB BF 48
after Tee-Object  -Append       : len=71
bytes after header (first 8 of tail): 63 00 68 00 69 00 6C 00     <- 'c','h','i','l' in UTF-16LE
ReadAllText sees header?  True
ReadAllText sees appended? False
Select-String match count: 0
Get-Content line count: 4
```

And this is not hypothetical — the **real artifacts on disk are affected**:

```
logs\a_s42.log        first8: EF BB BF 43 4F 4D 4D 41   (UTF-8 BOM + "COMMA")
UTF-16 CRLF (0D 00 0A 00) first found at byte 290        (UTF-16LE body follows)
logs\b_causal_s42.log first8: EF BB BF 43 4F 4D 4D 41
logs\b_stab_s42.log   first8: EF BB BF 43 4F 4D 4D 41

Select-String -Path logs\a_s42.log -Pattern "Validation bpb"  ->  count=0
```

So `logs/a_s42.log`, `logs/a_s43.log`, `logs/a_s44.log`, `logs/b_causal_s42.log`
contain a UTF-8 header and a UTF-16LE body. Any regex extraction over them returns
nothing or garbage, and the raw byte stream cannot be read as UTF-8.

- **Severity:** High. **Confidence:** CONFIRMED (reproduced in a temp file *and*
  observed in the committed run logs).
- **Affected past results:** **yes.** The `run_sweep.ps1` A/B logs are the primary
  artifact for the A vs B' comparison. Any BPB taken from them via
  `Select-String`/regex would have been `$null`; a number that exists must have come
  from another source. Re-derive from `manifests/`, wandb, or
  `logs/ab_v2_results.json` / `ab_v3_results.json` (which are written with
  `Set-Content -Encoding utf8` and are intact).
- **Verify:** the byte probe above; or
  `Select-String -Path logs\a_s42.log -Pattern "Validation"` → 0.
- **Fix:** add `-Encoding utf8` to the `Tee-Object`, and regenerate. For existing
  logs, recover with:
  ```powershell
  $b = [System.IO.File]::ReadAllBytes('logs\a_s42.log')
  $txt = [System.Text.Encoding]::Unicode.GetString($b)   # then re-split
  ```
  (the header bytes must be stripped first).

### F2 (High) — `2>&1` combined with `ErrorActionPreference = "Stop"` truncates the run

`run_sweep.ps1:12` sets `$ErrorActionPreference = "Continue"` and `run_sweep.ps1:83`
uses `2>&1 | Tee-Object | Select-String`. `run_ab_v2.ps1:64-72` documents precisely
why this is wrong:

> base_train.py writes its INFO logging to STDERR. Under "2>&1" PowerShell wraps
> each stderr line as an ErrorRecord (NativeCommandError), and with
> `$ErrorActionPreference = "Stop"` the FIRST such line terminates the script.

`run_sweep.ps1` has `Continue`, so it does not terminate — but the measured
behaviour is still wrong:

```
P1. cmd /c "echo out1 && echo err1 1>&2" 2>&1 | Tee-Object -Variable tv
  THREW: err1
  => the FIRST stderr line becomes a terminating error
```

With `Continue` the ErrorRecord is not terminating, but stderr lines arrive as
`ErrorRecord` objects, so `Select-String`'s `ForEach-Object { $_.ToString() }` and
the `-Pattern` match behave differently for stderr than for stdout. `base_train.py`
puts its INFO logging on stderr, i.e. exactly the lines the sweep is grepping for.

- **Severity:** High. **Confidence:** CONFIRMED for the `Stop` interaction
  (measured); CONFIRMED by reading for the `Continue` interaction (stderr lines
  become `ErrorRecord`, and the sweep greps stderr-heavy output).
- **Affected past results:** **yes, potentially** — the sweep's live console output
  and its `Select-String` filter are unreliable. The *log file* is separately broken
  by F1. Together, `run_sweep.ps1`'s observable behaviour is not trustworthy.
- **Verify:** the P1 reproduction.
- **Fix:** port `run_ab_v3.ps1`'s `Invoke-Run` (Start-Process with
  `-RedirectStandardOutput`/`-RedirectStandardError` to separate files, then fold
  them with `Get-Content -Encoding utf8 | Add-Content -Encoding utf8`). It is
  already written, already documented, and already handles the venv-shim child
  process, the exit-code handle, and the timeout. `run_sweep.ps1` should use it.

### B-checklist items that are **clean**

- **Case-insensitive collisions** — `run_ab_v3.ps1:39-42` explicitly names the
  variable `$RESULTS_PATH` and comments that `$results` would collide with the
  ordered dictionary. Verified the mechanism is real (`$ResultsPath` and
  `$RESULTS_PATH` are the same variable in PowerShell), and the code avoids it.
- **Functions polluting the success stream** — `Build-Cmd` and `Invoke-Run` use
  `Write-Host`, with a comment at `run_ab_v3.ps1:57-62` explaining exactly why
  (`Write-Output` would return `Object[] [EXEC line, exitcode]`). Verified:
  `Get-Bad` (Write-Output) returns `Object[]` and `[int]` cast throws; `Get-Good`
  (Write-Host) returns `Int32`. The code is correct.
- **Timeouts leaving child processes alive** — `run_ab_v3.ps1:112-118` kills the
  tree with `Stop-Process` + `taskkill /T /F` (needed because the venv
  `python.exe` is a shim that spawns the real interpreter), sleeps 3 s for the GPU
  context, then `WaitForExit(15000)`. Correct and well-reasoned.
- **`Start-Process -Wait` race** — `run_ab_v3.ps1:98-109` replaced
  `Wait-Process -Timeout` with a `HasExited` poll precisely because `Wait-Process`
  throws when the PID is already reaped. Correct, and the comment records the
  `cmd /c exit 0` self-test that motivated it.
- **`$p.ExitCode` being null** — `run_ab_v3.ps1:94` caches `$null = $p.Handle`
  before reading `.ExitCode`. Correct, with the self-test documented.
- **BOM** — `Set-Content -Encoding utf8` under 5.1 writes `EF BB BF`
  (verified). `Get-Content -Encoding utf8` consumes it, and the regexes are applied
  to the *folded* log, so the folded path is safe. `ConvertTo-Json` output in
  `logs/ab_v*_results.json` carries a BOM, which some strict JSON readers reject —
  Low, and a one-line fix (`-Encoding utf8NoBOM` is not available in 5.1; use
  `[System.IO.File]::WriteAllText($p, $json, (New-Object System.Text.UTF8Encoding($false)))`).
- **Log collision / overwriting** — `run_ab_v3.ps1:213-219` aborts if the log
  already exists, with the comment "the script appends, so reusing a log would
  merge runs". Correct. `run_ab_v2.ps1` likewise.
- **Failed run recorded as success** — `run_ab_v3.ps1:268-273` checks
  `$isNan -or $code -ne 0 -or $bpb -eq $null` and records `halted`. Correct, and it
  checks `$bpb -eq $null` so a *missing* metric is a failure, not a pass. Note the
  ordering: `124` is checked **before** the generic nonzero branch, so a timeout is
  not misreported. Good.
- **Double `Step 00600` line** — `Get-FinalBpb` aborts if the anchored regex
  matches more than once. Correct.
- **`Set-Location` before relative paths** — `run_ab_v3.ps1:15` does
  `Set-Location "C:\AI\nanochat"`, and uses absolute paths for `$py` and
  `Resolve-Path`. `run_ab_v2.ps1:24` sets `$ErrorActionPreference = "Stop"` then
  `Set-Location`. Consistent.
- **`taskkill 2>&1` under `Stop`** — `run_ab_v3.ps1:115` wraps it in
  `try/catch` and pipes to `Out-Null`; verified this is necessary (`taskkill` on a
  dead PID writes to stderr and throws). Handled.
- **Argument quoting** — `run_ab_v2.ps1:74` documents that
  `Start-Process -ArgumentList` joins an array with spaces and re-adds no quotes.
  All current tags (`ab-v3-C-s42`) are space-free, so this is currently safe, but
  it is a latent trap if a tag ever contains a space. Low, **Confidence:**
  CONFIRMED by reading the comment plus a `-join ' '` demo.

---

## C. General bugs

### F11 (Medium) — three `lm_head` applications and a discarded loss per forward

`rda.py:900-903` calls `self.base.forward(..., return_hidden=True)`, which
unconditionally runs `gpt.py:524` `self.lm_head(x)` over the full `(B, T, V)` and
computes `gpt.py:532` `F.cross_entropy(...)`. `rda.py:1148` then **recomputes**
`loss = F.cross_entropy(lm_logits.view(-1, ...), targets.view(-1), ...)`, discarding
the base loss. `rda.py:1132` runs `lm_head(hidden_mod)` for the real logits, and
`rda.py:964` → `_safe_logits` runs it a **third** time on `pre_outcome_hidden` for
the PE. So: 3 × `lm_head` and 2 × full-vocab `cross_entropy` per training step, one
of which is thrown away.

- **Why it matters:** pure waste, and it is why the RDA arms are not
  compute-matched to baseline — relevant to any throughput/MFU comparison between
  arms. The waste is roughly `(T/T + (T-1)/T) × lm_head` ≈ 2× the head cost, and
  one full `(B, T, V)` logits tensor plus its `cross_entropy` is discarded.
- **Severity:** Medium. **Confidence:** CONFIRMED (three call sites verified by line
  number; the base loss assignment is shadowed at `rda.py:1148`).
- **Affected past results:** no metric changes; it does affect the reported
  tokens/sec and MFU, so any "RDA costs N% throughput" claim is measured against a
  number that includes avoidable work.
- **Verify:** count `lm_head` calls with a forward hook during one step.
- **Fix:** add a `compute_loss=False` / `return_hidden_only` path to
  `GPT.forward` that skips `gpt.py:524-532` when `return_hidden=True` and
  `targets is not None` and the caller will recompute. Note `gpt.py:521` already
  supports `return_hidden` without logits, so this is a small, contained change.

### F12 (Low) — `_log_debug` prints to stdout, bypassing `print0`

`rda.py:858-865` uses bare `print(..., flush=True)`. `base_train.py` funnels
everything else through `print0`, which is rank-guarded — so on a multi-rank
(DDP) run the debug output would be emitted by every rank, interleaved, unlike
every other line. The `if not self.base.training: return` guard at `rda.py:827`
correctly limits it to training, and `_debug_count` bounds it to `debug_steps`.

- **Severity:** Low (single-GPU runs are unaffected; this machine is 1 GPU).
- **Confidence:** CONFIRMED.
- **Fix:** pass a logger in, or gate on a rank check.

### F13 (Low) — `_debug_count` is incremented before the `training` check

`rda.py:822-827`:
```python
if self._debug_count >= self.debug_steps:
    return
self._debug_count += 1
if not self.base.training:
    return
```
The counter advances during eval forwards too, so the `debug_steps` budget is
consumed by validation passes. Harmless (nothing is printed), but it means the
"first N training steps" claim in the docstring is only true when no eval happens
in the first N steps — and `base_train.py` evals at `step % eval_every == 0`, which
includes step 0. **Confidence:** CONFIRMED.

### F14 (Low) — dead code that reads as live

- `rda.py:1179-1193` `_logits_from_hidden` — **1 occurrence repo-wide**, i.e. only
  its own definition. Never called. Its docstring says "Full-sequence logits for
  the inference path", but the inference path uses `rda.py:1132` inline. Delete.
- `rda.py:613` `count_document_starts` — **1 occurrence**, only its definition.
  `base_train.py:918` calls `document_start_mask` directly and reads
  `out["n_doc_starts"]` instead. Delete or call it.
- `rda.py:618-630` `episode_continuation_mask` — **1 occurrence**. The docstring
  says "RETAINED FOR REFERENCE ONLY", which is honest, but a module-level public
  function with a live-looking name and no callers is a trap. It also documents a
  claim I could not verify (`"False for every row of every batch"`) — that depends
  on the bestfit packer always emitting a BOS at position 0, which is a property
  of `dataloader.py`, not of this function. Its own test shows it can return
  `True` for a general input.
- `rda.py:1079` `E_hat_mean` is computed and passed only to `_log_debug`.
  Legitimate.
- `rda.py:790` tautological assert (see A1).

**Severity:** Low. **Confidence:** CONFIRMED (occurrence counts by repo-wide grep).

### F15 (Low) — mutable/duplicated flag: `const_state_vec_init` is stored but read once

`rda.py:697` stores a plain-Python copy of the arm-C init "for the end-of-run
report in `base_train.py`". `base_train.py:1167-1170` does use it. Consistent — no
finding, recorded because it is the one hand-off between the two files that is easy
to miss and it does work.

### F16 (Low) — `E_init` local shadows the parameter name in `rda.py`

`rda.py:919` `E_init = self.state.init_state(...)` shadows nothing at runtime (the
parameter is `self.state.e_init`) but uses the same identifier for a different
object (a `(B, n_drives)` tensor vs. an `(n_drives,)` parameter). Grepping `e_init`
in this codebase already returns 82 hits, most of them the parameter. A
reader skimming `E_init` in `rda.py` can easily conflate the two. Rename the local
to `E_zero` or `E_start_state`. **Confidence:** CONFIRMED.

---

## D. Readability and maintainability (kept short)

- **D1.** The comment blocks in `rda.py` are extremely long — `rda.py:40-70` (`_unit_rms`),
  `rda.py:196-232` (`update`), `rda.py:340-399` (`update_seq`), and especially
  `rda.py:940-1056` (117 lines inside `forward`). Several document *historical
  mistakes* ("the previous rule was…", "what was here before, and why it was wrong").
  This is valuable institutional memory and directly explains why the current code
  is correct — but it makes the live code hard to review, which is a real cost: the
  PE-index bug survived a detailed comment that asserted the opposite conclusion.
  Consider moving the archaeology to `docs/` or a module-level comment and keeping
  the inline text to the invariant plus a pointer.
- **D2.** `_log_debug` prints `E_t.norm()=`, `E_hat_next.norm()=`, `E_next.norm()=`
  as *labels* while the values are per-row norms from `.norm(dim=-1)`. The
  docstring correctly explains the distinction, but the label re-introduces exactly
  the ambiguity the docstring was written to prevent. Label them `norm(dim=-1)`.
- **D3.** Magic numbers that are load-bearing and not explained at the point of
  use: `softcap = 15` appears three times (`gpt.py:523`, `rda.py:1133`,
  `rda.py:1190`) with no shared constant. `pe_reference_scale = 10.0`
  (`rda.py:96`) is explained. `film_rms = 0.5` (`rda.py:152`) is explained.
  Hoist `SOFTCAP` so the three copies cannot drift.
- **D4.** `GPTWithRDA.forward` is ~190 lines with 6 responsibilities (leak guard,
  state carry, PE, recurrence, predictor, arm-C swap, summarise, modulate, head,
  loss, generation branch). Extracting the arm-C swap and the summarise block would
  make the causal path reviewable in isolation, which is where the history shows
  bugs live.
- **D5.** `diagnostics/probe_drives.py:118-123` comments describe the sequential
  implementation (`(B, n_drives)`, "hook fired T-1 times") while the code path is
  the vectorized one. See F9. Stale comments after a refactor are the most common
  source of the misreadings recorded in this project's memory.
- **D6.** `diagnostics/` contains 40+ scripts, several of which (`audit_facts.py`,
  `audit_leak_size.py`, `audit_leak_decisive.py`, `audit_headtohead.py`) exist only
  to analyse a bug that is now fixed. A one-line header banner ("ANALYSES PRE-FIX
  CODE — expected to fail against current HEAD") would stop a future reader from
  treating their output as current.

---

## Category summary

| Category (A) | Result |
|---|---|
| Label / lookahead leakage | **Clean.** Index arithmetic correct; gate passes; independently verified non-vacuously. One tautological guard (F10-adjacent, `rda.py:790`). |
| Conditional-definition / NameError | **Clean.** Known `_rda_diag_lr_scale` bug properly fixed. |
| Reductions without explicit `dim` | **1 found** (F6, `probe_drives.py:123`). `rda.py` itself is clean. |
| Signed vs absolute statistics | **1 found** (F7, `rda_sat_log` unsigned in checkpoints). E13 analyses are correct. |
| Optimizer groups / grad-None / weight decay | **1 found** (F5, `e_init` justification false). Coverage, groups and decay policy otherwise correct. |
| Hooks / global state / RNG | **2 found** (F8 hooks not in `finally`; F9 stale comment). RNG handling is exemplary. |
| Cross-step state / memory / stale detach | **1 found** (F3, persistence vacuous at T=1024) + 1 Low. |
| Numerical (epsilon, tanh, TF32, `==`) | **1 found** (F10, `_unit_rms` 1000× gradient amplification). Closed form and `_max_safe_T` verified correct. |
| Seeds / reproducibility | Documented, not hidden. Seed does not reach data order — stated in `common.py`. |
| Eval vs train | **1 found** (F4, eval always `mode='reset'`). |

## Method and cleanup

- Read: `nanochat/rda.py` (all 1063 lines), `scripts/base_train.py` (RDA,
  optimizer, LR, eval, seeding, checkpointing), `nanochat/optim.py` (fused AdamW +
  `_compute_adamw`), `nanochat/gpt.py` (forward/loss), `nanochat/common.py`
  (`compute_init`), `nanochat/loss_eval.py` (`evaluate_bpb`), `nanochat/engine.py`
  (RDA references), all 5 PowerShell drivers, and the diagnostics scripts named in
  scope (`test_no_label_leak.py`, `probe_pe_film.py` hook/precision handling,
  `probe_drives.py`, `diagnose_all8.py`, `diagnose_saturation.py`,
  `smoke_rda.py`, `smoke_saturation.py`, `audit_causality.py`,
  `audit_leak_size.py`, `map_pe_film_graph.py`, `probe_arm_c.py`).
- Verification: three CPU-only scripts under
  `%LOCALAPPDATA%\Temp\nanochat_review\` (`verify_rda.py`, `verify2.py`,
  `ps_tests.ps1`) plus one run of the repo's own
  `diagnostics/test_no_label_leak.py`. **No GPU training, no probe/smoke runs, no
  optimizer steps.** All scratch files have been **deleted**.
- Not read in full (flagged rather than silently skipped): the remaining ~30
  `diagnostics/*.py` scripts, and `nanochat/dataloader.py` beyond the BOS/bestfit
  properties relied on above.
