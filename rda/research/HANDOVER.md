# HANDOVER

## Established
- **E1** baseline 3 seeds 1.301217 / 1.287544 / 1.295109, spread **0.0137**.
- **E3–E5** fixed a 1022-token FiLM and a one-token PE lookahead; per-position `film_state`. Vectorized recurrence 3.33e-16, 222.8x.
- **E9–E11** the runaway is **FiLM self-excitation** (frozen-base control: PE flat at 10.50, FiLM gain 1.00→9.57). Fix holds |E|≤0.94 with **loss unchanged** (5.808 vs 5.805).
- **E14** `pe_seq` read `targets[:,1:]` so logits[p] saw x[p+1]. Fixed 969e067; the **0.0749 BPB gain is withdrawn**.
- **E13-fixed-3000** threat: `pe_proj.bias` = **-1.711 = 80%** of arg (p50 -2.11, residual 4.6e-07); p50 crosses -2 at 1131, never -3. Drive **unimodal at -0.972** (mode 58-64%, f>0.9 = 0.0). State on fixed point E*=d* (diff 3e-05). Applied gain 1.0→3.59, still climbing. |E_mean|/E_std = **436 threat**, **140 novelty**. Reconstruction residual is **TF32** (`precision='high'`): replay bit-exact 0.000e+00.

## Retractions
E2 (single seed + leaky); E13 1000-step (carried the leak); G2/G3 (wrong FiLM gain column, twice); "positive tail" (twice — absmax is |min|); "arousal/lp/homeostatic mirror threat" (they sit near centre); "threat 6th/least offset" (`.std()` with no `dim` over (768,B,T)); reconstruction gate FAIL 3.2e-04 (tolerance finer than TF32); valley test inverted (false BIMODAL); "tanh deriv never <0.05" (crosses 1743 on a sibling run); "shuffle y" and "PE OFF via targets=None"; **E15 "offset is used" (threat −0.0017, novelty −0.0008 zero-ablate HELPED — only uncertainty +0.0177 and agency +0.0016 are useful); E15 "420x" (actually 1.79x largest, 1.96x sum — super-additive, not cancelling)**.

## E15 ablation (DONE 2026-10-01) — see research/E15_ABLATION.md
mean-ablate ALL +0.0004 => position variation almost entirely unused.
Bug found: committed `--ablate` passed the ORIGINAL `E_t`, so all deltas would have been 0.0; fixed.
`per_batch_state` still not serialized (means from batch 1). Noise floor 0.0 is deterministic eval, not sampling error — `within_noise` field is meaningless. 1 seed, 1 checkpoint.

## Open
- **E16**: PE plateaus ~33.8 then falls to 18.8 — unexplained.
- No valid B' A/B since the leak fix; 1.226305 ran at `6ae93ce`, the commit BEFORE `969e067`.
- Runs **not bit-reproducible** (5 runs: crossing 1177-1353, peak 0.9848-0.9893).
- Criterion-1 replacement **not implemented**; thresholds uncalibrated.
- Health-flag defect: |bias|/|arg_p50| undefined when |arg|<0.2.
- **A/B ambiguities** (research/AB_SETUP_READONLY.md): checkpoint = final-600 or best? baseline logs truncated (690 lines, no BPB/timing); three names for one B' run (b-causal-s42 / b-stab-s42 / B'-s42); BPB source `--eval-every=50` vs `base_eval.py --device-batch-size=32` — need not agree.

## Next — NOT LAUNCHED
Seed replication of the E15 ablation, then the post-fix B' A/B (s42 first, then 43/44 on instruction).

## Parked until E13 is stable
Vision and 2D world simulation.

## Uncommitted
`logs/**` (excluded). `base_train.py` unchanged.

---

## Related Concepts

Start here if you are new:

- [[RDA_ARCHITECTURE_GUIDE]] — the architecture, taught from first principles
- [[Python for ML]] — the Python needed to read the code
- [[RDA and Persistent State]] — the system as a whole
- [[CURRENT_STATE]] — verified description of every pathway

The retraction list above maps to [[FAILED_EXPERIMENTS]] and [[Label Leakage]].
The open items map to [[RESEARCH_QUESTIONS]] and [[Control Arm]].
