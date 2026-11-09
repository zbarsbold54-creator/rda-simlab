# Python for ML

Python concepts needed to read and modify this project. This is **not** a
general Python tutorial — it covers only what appears in `nanochat/rda.py`,
`scripts/base_train.py`, and the diagnostics.

---

## Tensors and shapes

The single most important habit: **know the shape of every tensor, and write it
in a comment.**

```python
pre_outcome_hidden = hidden[:, :-1, :]   # (B, T-1, d_model)
```

This project has been damaged twice by getting a shape's *meaning* wrong while
its *size* looked fine. See [[Prediction vs Actual State]] for the 2-D vs 3-D
`std()` axis error.

### The indexing operators

| syntax | meaning |
|---|---|
| `x[i]` | one element, drops the dimension |
| `x[i, j]` | two elements, drops both |
| `x[:, j]` | **all** of dimension 0, pick `j` from dimension 1 |
| `x[:, :-1]` | all of dim 0, all but the last of dim 1 |
| `x[..., :-1]` | all leading dims, all but last |
| `x.unsqueeze(1)` | insert a size-1 dimension at position 1 |
| `x.expand_as(y)` | broadcast to y's shape — a **view**, allocates nothing |

`unsqueeze` and `expand_as` appear constantly in `rda.py`. Both are used to let
one code path accept two different shapes — for example `predict_seq` accepts
`(B, T, 8)` and `(B, 1, 8)`.

---

## Classes and inheritance

RDA wraps a base model rather than modifying it:

```python
class GPTWithRDA(nn.Module):
    def __init__(self, base_model, ...):
        super().__init__()              # REQUIRED before assigning submodules
        self.base_model = base_model
        self.state = RDAStateModule(...)
```

`super().__init__()` is not optional. PyTorch's `nn.Module` uses it to register
submodules and parameters; skip it and assigning `self.state = ...` fails or,
worse, silently produces a module with no parameters.

`self.base_model = base_model` is how a wrapper reuses an existing model rather
than copying its weights.

---

## The training loop idiom

```python
for step in range(num_iterations):
    for micro in range(grad_accum_steps):        # accumulate
        x, y = next(train_loader)
        loss = model(x, y)["loss"] / grad_accum_steps
        loss.backward()
    optimizer.step()                              # then update
    optimizer.zero_grad(set_to_none=True)
    model.train()
```

Two ordering rules this project depends on:

**Zero grads *after* the step.** `zero_grad(set_to_none=True)` sets grads to
`None` rather than zero. That is why the grad-free priming in `base_train.py`
must run on **every** backward — prime once and step 1 crashes.

**`.item()` forces a GPU sync.** `int(doc_starts.sum().item())` stalls the
pipeline until the GPU catches up. Used sparingly here for logging.

---

## Type hints and casts

```python
E_t_external: torch.Tensor = None
if E_t_external is not None:
    E_t = E_t_external.detach().to(device=device, dtype=torch.float32)
```

`.detach()` returns a tensor sharing storage but **detached from the graph** — no
gradients flow through it. Required when carrying state across steps; without it
the graph would grow across the whole run.

`.to(device=..., dtype=...)` moves and/or casts. The base model runs in bf16
while RDA keeps fp32 master weights, so RDA casts incoming activations:

```python
hidden_state = hidden_state.to(self.event_proj.weight.dtype)
```

Omitting this is a common source of dtype errors.

---

## Assertions

```python
assert E_next.abs().max() <= 1.0 + 1e-6, (
    f"gated update escaped [-1, 1]: max|E_next|={E_next.abs().max()}"
)
```

In this project assertions are **design tools**, not debug leftovers. The one
above means a future change that reintroduces saturation stops training instead
of silently producing thousands of meaningless steps.

There is also a coverage assertion that raises if any RDA parameter is in no
optimizer group — otherwise it would silently never update.

---

## Views versus copies

The most dangerous subtlety in PyTorch:

```python
film_state = self.const_state_vec.to(film_state.dtype).expand_as(film_state)
```

`expand_as` returns a **view**. Writing to it writes to the original. The
comment in `rda.py` notes that `.to(dtype)` is a no-op when dtypes already match,
so the whole expression allocates nothing — which is why arm C adds no memory
cost.

`.contiguous()` forces a copy. Needed before in-place ops on a view.

---

## Control flow and reduction

```python
E = E_t_external
if E is None:                 # not `if E:` -- ambiguous for tensors
    E = init_state(B)

for p in mod.parameters():    # iterate a generator
    ...
```

`.parameters()` returns a **generator**, so `len(mod.parameters())` raises
`TypeError`. Use `len(list(...))`. This has bitten the project twice.

`if tensor:` raises for tensors with more than one element — always compare
explicitly with `is None`.

---

## NumPy in a PyTorch codebase

Used sparingly, for dtype limits:

```python
finfo = np.finfo(torch.float64) if ... else np.finfo(torch.float32)
limit = math.log(finfo.max) / math.log(1 / decay)
```

`torch.finfo` would avoid the numpy dependency, but numpy was already imported
here. Either is fine — just be consistent.

---

## Numerical precision — the TF32 trap

This project runs fp32 matmuls with **TF32 enabled**:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')     # NOT 'highest'
```

TF32 carries **10 mantissa bits**, so relative resolution is about
`4.9e-4`. Reconstructing a forward pass from captured inputs in pure fp64
therefore differs by ~`3.9e-4` — **not a bug**.

Diagnostic tell: fp64 and `'highest'`-without-TF32 give the *identical* number.
Two different arithmetics agreeing means "different arithmetic", not "more
accurate view".

> Any gate comparing a TF32 forward against an fp64 reconstruction must use a
> **relative** tolerance of about `1e-3 × max|out|`. A `1e-4` absolute
> tolerance is finer than TF32 can represent and will reject correct code.

This gate once rejected a bit-exact decomposition.

---

## Buffering and logging

Python block-buffers stdout when it is redirected to a file — in ~8 KB chunks.

**Symptom:** a log shows step 200 while the trainer is really at step 123, and a
45-second poll sees zero new bytes.

**Fix:** set `PYTHONUNBUFFERED=1`, or use `print(..., flush=True)`.

This produced a 1.8× pace measurement error and a phantom stall in this
project — it was the reason a run was wrongly judged to be over a 90-minute
cap.

---

## PowerShell pitfalls on Windows

The driver scripts are PowerShell, and several of its sharp edges cost real
time here.

### Variable names are case-insensitive

`$results` and `$RESULTS_PATH` are different names — but `$results` and
`$RESULTS` are **the same variable**. This silently overwrote a results path
with an ordered dictionary.

### `2>&1` plus `Stop` is fatal

```powershell
$ErrorActionPreference = "Stop"
some_command 2>&1 | ...
```

A benign stderr line from a child process becomes a terminating error. Use
`Start-Process` with separated `-RedirectStandardOutput` / `-RedirectStandardError`
to real files instead.

### `Start-Process -PassThru` hides the exit code

```powershell
$p = Start-Process ... -PassThru
$null = $p.Handle          # REQUIRED before reading ExitCode
$p.WaitForExit()
$p.ExitCode                # $null without the cached handle
```

Verified: `cmd /c exit 5` gives `ExitCode=[]` uncached and `[5]` cached.
Unfixed, **every** run looks like it had no exit code.

### `Wait-Process -Timeout` misfires

It raises when the PID cannot be found — which also happens when a
short-lived process is reaped before the wait starts. A `cmd /c exit 0` dummy
was reported as a *timeout*. Poll `HasExited` against a deadline instead.

### Output-file writes lose buffered data

`Out-File` in PowerShell 5.1 writes UTF-16; `Tee-Object` piped output can be
mixed-encoding. Use `[System.IO.File]::WriteAllLines` with an explicit
`UTF8Encoding($false)` for anything a script will parse later.

---

## Testing patterns in this project

**A reference implementation kept for comparison.** `_update_loop_reference()`
is the sequential recurrence, preserved verbatim, purely so the vectorised
closed form can be diffed against it. It caught a missing factor of `decay`
that made the two disagree by `2.9e-2` in fp32.

**Independence checks.** The A/B verifies that 78 shared parameters are
bit-identical between arms — a strong statement that requires `torch.equal`,
not an approximate comparison.

**Wrapping methods for instrumentation.** `modulate` is a method, so
`register_forward_hook` does not catch it. Wrap the bound method on the instance
and return the original output unchanged.

---

## Related concepts

- [[RDA and Persistent State]] — the code these concepts serve
- [[Label Leakage]] — a bug that lived in an indexing expression
- [[Optimizer State]] — where `grad_free` is and is not read
- [[Control Arm]] — the experiment these tools enable
- [[BPB]] — the metric the results are read in
