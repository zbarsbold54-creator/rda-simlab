# Label-leak audit — provenance

Read-only audit that found, proved and quantified the one-token label leak in
`nanochat/rda.py`, fixed in `969e067`. No training was run for any of this.
The scripts here are kept as the evidence trail for the correction recorded
in `research/EXPERIMENT_LOG.md` (E14, E13-SUPERSEDED, E13-fixed),
`paper/LIMITATIONS.md` §4a–4b, `paper/METHODS.md` §1.2, and `README.md`.

## What the leak was

The dataloader makes `y[p] == x[p+1]` exactly (verified, fraction 1.000000)
and `gpt.py:532` scores `cross_entropy(logits[p], targets[p])` with **no
shift**. So `y[p]` is the label attached to position `p`'s logits.

`rda.py` built the per-position prediction error from `targets[:, 1:]`:

| step | location | consequence |
|---|---|---|
| 1 | `rda.py:901` `pe_seq[j] = PE(logits_j, targets[:,1:][j])` | `= PE(·, y[j+1]) = PE(·, x[j+2])` |
| 2 | `rda.py:920` `E_next_seq[j]` | depends on `x[j+2]` |
| 3 | `rda.py:972` `film_state[p] = E_next_seq[p-1]` | depends on `x[p+1]` |
| 4 | `rda.py:1036` `logits[p] = lm_head(modulate(hidden[p], film_state[p]))` | depends on `x[p+1]` |

`logits[p]` therefore depended on the token it is graded against.
`film_state` is the **only** route from the drive state to the logits, so
this was the entire leak surface. The fix indexes `pe_seq` as
`targets[:, :-1]`.

## Scripts

| script | what it establishes |
|---|---|
| `audit_facts.py` | The two load-bearing indexing facts, read off the data: `y[p] == x[p+1]` = 1.000000, and the no-shift loss alignment. Also spells out the derivation step by step. |
| `audit_leak_decisive.py` | Locality: perturbing `y[:,k]` leaves hidden/logits at positions `< k` **exactly** unchanged and moves every position `>= k` (cumulative recurrence). The violation is the diagonal. |
| `audit_leak_size.py` | Sizes the leak, and shows PE→E is the only route (`max|ΔE|` = 0.9287). Also contains two **retracted** methods — see below. |
| `audit_headtohead.py` | Baseline vs B′ on identical batches from both checkpoints. |
| `audit_causality.py` | The three requested dynamic tests plus eval-only ablations. Contains the exact-zero-delta bug that revealed the wrong checkpoint load. |
| `audit_followup.py` | Establishes that the carried state sits at a **fixed point** (`max|E|` identical to 8 dp every batch), which is why batch order cannot matter. |
| `reconcile_gain_metric.py` | Establishes what the console `film s` column actually is and that the harness was unchanged between E13 and E13-fixed. |
| `recompute_gates.py` | Recomputes G1–G3 on the **correct** (applied-scale) series. |

## Output files

`audit_*.txt` are the raw captured output of each script, unmodified.
`leaktest_prefix.txt` is the regression test **failing with 12 violations**
on `6ae93ce`; `leaktest_postfix.txt` is the same test passing with 0 after
`969e067`. `gates_recomputed.txt` is the corrected gate evaluation.

## Retracted methods — do not reuse

These were tried during the audit, found invalid, and are recorded here
because a reader who finds the same scripts should not repeat them.

1. **"Shuffle `y` and re-score" as a label-blindness test.** The *baseline*
   also moves (+1.748) under it, because scoring against wrong answers is
   simply a harder task. It confounds cheating with difficulty. Not
   evidence in either direction.
2. **"PE OFF via `targets=None`" (bpb 4.29).** Invalid counterfactual: it
   puts the *trained* FiLM into a state distribution it never saw, so it
   measures a broken model rather than an ablation.
3. **`max|ΔE| < k > 0` as a leak verdict.** Wrong bound. `E_next_seq[p]`
   legitimately *records* token `p+1`; the causal bound belongs on
   hidden/logits, not on the state.

## One process note

The first load of `b-stab-s42/model_000600.pt` targeted `m.base` instead of
the wrapper, leaving FiLM at zero init. The signature was unmistakable:
every ablation delta exactly `+0.000000` at `bpb 10.397310` = `ln(32768)`,
i.e. a randomly-initialised model. The checkpoint is in fact a **full
`GPTWithRDA` state dict** (78 keys = 63 `base.*` + 11 `state.*` + 4
`predictor.*`) and must be loaded with `m.load_state_dict(sd, strict=True)`.

Generalisation: an implausible *exact* equality between a changed and an
unchanged condition means the experiment is broken, not that the change had
no effect. This heuristic caught the wrong load, a monkey-patched diagnostic
that was shadowing the real fix, and a "frozen base" probe that was not
actually in any optimizer group.
