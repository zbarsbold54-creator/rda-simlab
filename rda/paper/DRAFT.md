# Persistent Functional State and Feature-Wise Conditioning in a Small Transformer: Negative Results and Methods

*A toy-scale evaluation of the Regulatory Drive Architecture (RDA)*

> Draft for internal review. Every number is sourced in [claims.md](claims.md). Unsourced items are marked [UNVERIFIED]; missing references are marked [CITE]; open decisions are marked [AUTHOR TO CONFIRM]. Open items are listed in [OPEN_ITEMS.md](OPEN_ITEMS.md).

## Abstract

We report a toy-scale evaluation of a persistent, low-dimensional functional state, the Regulatory Drive Architecture (RDA), added to a small transformer. The state has eight named drive dimensions, a recurrent update with a decay parameter, and per-position FiLM conditioning. The names are design labels; the study does not measure or claim subjective emotion. [CF, Q01]

In a FineWeb-Edu comparison at depth 8 and width 768 over 600 iterations, three training seeds gave a mean final validation BPB of 1.293795 for the baseline (A), 1.276351 for dynamic-state RDA (B'), and 1.275592 for constant-state RDA (C). The mean paired B'-minus-A difference was -0.017444, and the mean C-minus-B' difference was -0.000758. Against a 0.0137 BPB yardstick, the mean A-to-RDA difference exceeds the yardstick in magnitude, and the mean B'-to-C difference does not. [LM]

A single-seed pilot of 4578 iterations (about 150 million tokens, width 512, a different window pattern) did not show the B'-versus-A difference: the final gap was -0.000270, and C was -0.001279 below B'. [PL, CF]

On a synthetic recall task, the multi-timescale decay model (D) and its constant-state control (D_const) stayed near 16-way chance on cross-segment delays in two training seeds. The descriptive D-minus-D_const CROSS contrasts were +0.003418 (seed 43) and 0.000000 (seed 101). A positive control without segment masking recalled cross-segment items at 0.995-1.000, so the benchmark can detect cross-segment recall. [RG, PC]

A back-of-envelope bound says that a single write into a decay-based state contributes at most about 1.9e-3 after 192 steps. [AN]

The results support no demonstrated state-content effect and no cross-segment recall from the decay state. Errors that were corrected, interrupted and resumed runs, an incomplete recall grid, and an unfinished exploratory variant are reported in full.

Keywords: persistent state; feature-wise linear modulation; recurrent state; language modelling; recall; negative result.

## 1. Introduction

Transformers carry most information through attention and through context-dependent activations. RDA adds a persistent functional state and asks whether that state changes what the model predicts or remembers. The working hypothesis is that a recurrent summary of prediction error and event information, fed back through causal per-position conditioning, changes model behaviour. This is a functional hypothesis. It is not a claim that the model feels or experiences anything. [Q01]

This report separates two possible sources of any language-model difference: the availability of the extra FiLM pathway, and the content of a varying state carried by that pathway. A constant-state control keeps the conditioning route and removes position-varying state. A separate recall task asks whether a decay-based state supports retrieval across segments when attention is blocked between them. [CF, Q03]

The report contributes (a) a controlled comparison and a longer single-seed pilot, (b) a recall benchmark with an audit, a calibration record, and a positive control, (c) an account of errors, corrections, and interrupted runs, and (d) a back-of-envelope bound on what one write can contribute to a decay-based state. It is written as a negative-result and methods report. Null and negative results are stated as such.

## 2. Related Work

The manuscript needs verified prior-work coverage before submission. No authors, titles, venues, or dates are given here from memory.

- Nanochat and the base language-model training setup. [CITE: nanochat architecture and training setup]
- Feature-wise linear modulation and related conditioning methods. [CITE: FiLM and feature-wise conditioning]
- The optimizer components used by the training setup. [CITE: Muon and AdamW]
- Recurrent or persistent memory in language models. [CITE: recurrent memory in language models]
- Functional affect- or drive-like representations in machine learning. [CITE: computational affect and drive representations; avoid claims about subjective experience]

## 3. Method

### 3.1 Persistent functional state

The state $E$ has eight dimensions: novelty, uncertainty, threat, integrity, agency, learning progress, arousal, and homeostatic deviation. It is updated recurrently as

$$E_t = \lambda\,E_{t-1} + (1-\lambda)\,\mathrm{drive}_t,$$

where $\lambda$ is the decay. The state conditions the residual stream through FiLM. Conditioning is per position and uses the state available before the current position's update. A later audit tests that causal shift. An earlier label leak in this path is described in Section 7.1. [Q01, Q02, Q07]

### 3.2 Arms

- **A**: the unmodified baseline, with no RDA flags.
- **B'**: RDA with dynamic state.
- **C**: RDA with constant-state FiLM control (`--rda --rda-const-state`).
- **D**: a multi-timescale variant with one decay per drive dimension. The eight decay values are 0.9, 0.95, 0.975, 0.9875, 0.99375, 0.996875, 0.9984375, and 0.99921875. [DS]
- **D_const**: the constant-state control for D.

### 3.3 Recall benchmark: design and validation

Each evaluation row holds two events, one SALIENT and one ORDINARY, and a query for each class. Answers come from 16 values, so chance accuracy is 0.0625. WITHIN delays are 16 and 48 tokens. CROSS delays are 192, 320, 448, 576, 704, and 832 tokens. A block-diagonal causal mask over 128-token segments prevents attention from crossing segments. [Q03, CF]

The corrected generator audit reports 2,048 evaluation rows and 4,096 events. All 512 possible (class, key, value) triples occur. Each of the 16 value IDs occurs 16 times among the 256 events of each class-delay stratum. Training and evaluation row identities are disjoint, with zero overlap. [CB, Q04]

Calibration selected answer weight 20. Weight 5 failed the delay-16 gate of 0.80 (SALIENT 0.062500, ORDINARY 0.062500). Weight 20 passed it (SALIENT 0.996094, ORDINARY 1.000000). The first passing weight was frozen for all arms. [CB]

A drive-manipulation check on the seed-43 D checkpoint passed, with a paired Cohen's d of 5.710474 (95% CI 5.533349 to 5.899485). [MC, Q06]

### 3.4 Pre-set decision rules and power

**Language-model yardstick.** The 0.0137 BPB value is the spread (maximum minus minimum) of the baseline A seeds in an earlier three-seed run. It is used as a decision yardstick for this comparison. The repository records describe it as a proposal, not a calibrated value, and it is not a noise estimate. [PW, Q20]

**Recall.** The primary test is a one-sided exhaustive label-permutation test over the per-seed evaluation gaps, at p <= 0.05. At K=5 its minimum achievable p-value is 1/252 = 0.003968. The secondary tests are a paired sign-flip test (minimum 0.031250 at K=5) and a threshold rule. The planned design has K=5 training seeds and N = 6,144 pooled CROSS rows per class (6 delays x 1,024 rows). [PW, CB, CF, Q25]

At delta 0.02 and a seed SD of 0.01, simulated power was 0.6010 for the sign-flip test, 0.1580 for the threshold rule, and 0.8160 for the permutation test. These figures are conditional on the assumed seed variance; they are not observed power. At delta 0 and seed SD 0.01, the null rejection rates were 0.0310, 0.0005, and 0.0495 for the same three tests. [PW]

The spec records a minimum effect of interest of 0.02. This draft leaves the value for author confirmation: **[AUTHOR TO CONFIRM: minimum effect of interest]**. [PW]

Seed-43 recall rows were viewed descriptively before the power amendment. The record says the amendment changed the definition of N, and did not change the test, the threshold, the minimum effect, or K. [Q19]

## 4. Experiments

### 4.1 600-iteration A/B'/C comparison

The comparison used FineWeb-Edu at depth 8, aspect ratio 96 (width 768), head dimension 128, window pattern L, device batch 2, and 32,768 tokens per step, for 600 iterations with validation every 50 steps. Seeds were 42, 43, and 44. Seed-42 B' and C runs were reused from earlier device-batch-2 logs and not retrained. The other seven runs were trained for this comparison. [CF, LM]

### 4.2 Long pilot, seed 42

The pilot ran A, B', and C for 4,578 iterations with seed 42, device batch 2, 32,768 tokens per step, and validation every 250 steps. That is about 150 million tokens per arm (4,578 x 32,768 = 150,011,904). The pilot did not use the 600-iteration width or window pattern. It used the default aspect ratio and window pattern, and it passed no head-dimension or window flags. Table 1 lists the differences. [PL, CF]

**Table 1.** Configuration of the 600-iteration comparison and the long pilot.

| Setting | 600-iteration comparison | Long pilot |
|---|---:|---:|
| Depth (layers) | 8 | 8 |
| Aspect ratio | 96 | 64 |
| Width (n_embd) | 768 | 512 |
| Head dimension | 128 | default (no flag passed) |
| Window pattern | L | SSSL |
| Device batch size | 2 | 2 |
| Tokens per step | 32768 | 32768 |
| Iterations | 600 | 4578 |
| Validation interval (iterations) | 50 | 250 |

Arms ran sequentially under a resumable driver. Arm C was interrupted after its step-3000 checkpoint and resumed (Section 7.5). [PL]

### 4.3 Decay screen and evaluation-time ablations

The decay screen (branch `exp/multiscale-decay-20261011`, code commit 5bb8933) trained D and D_const on seed 43 for 600 iterations with the 600-iteration configuration. Three evaluation-time ablations were then run on the final checkpoint with `diagnostics/ablate_state.py`. V1 replaces the state with a constant, V2 shuffles state across rows, and V3 reverses time within each row. A V0 entry gives the unablated BPB. [DS, CF]

### 4.4 Recall grid: D and D_const

D and D_const were trained with training seeds 43 and 101, for 1,200 iterations each, with answer weight 20 and segment length 128. Each run was evaluated on 8,192 rows (1,024 per delay and class) with evaluation seed base 3,000,000,000. [RG, CF]

The recall driver planned five training seeds (43, 101, 102, 103, 104). Only seeds 43 and 101 produced results. The reason is not recorded **[UNVERIFIED]**. The seed-43 no-context decision marker in the results JSON records that the no-context stop was lifted. [CF, RG]

**Provenance note.** The driver stopped after the eval-D_const-101 subprocess exited with code 0, before its completion record. The D-101 and D_const-101 entries were then promoted from the temporary artifact `main_d_dconst.json.tmp.promoted-20261014-191304`. The promoted artifact and the current JSON have identical SHA-256 digests. The JSON's top-level `completed` flag is false. Older `.bak` and `.pre-promote` copies exist in the same folder. [RG, Q18]

### 4.5 Positive control and manipulation check

The positive control is arm A with segment masking disabled. It uses training seed 43 and checkpoint 1,200. Segment length is 0 in the training configuration and in evaluation, where a wrapper forces it to 0. It uses the same harness and readout as the recall grid. The folder `recall-A-nomask-s43-w20/` sits at the repository root and is untracked in this working tree. [PC, Q16]

### 4.6 Exploratory gated-write variant

A gated content-write variant of the state was developed on branch `mech2-20261015`. No result from it is reported; Section 9 describes its status. [MX]

## 5. Results

### 5.1 600-iteration comparison

**Table 2.** Final validation BPB by seed, with the 600-iteration comparison's mean and sample SD.

| Arm | Seed 42 | Seed 43 | Seed 44 | Mean (SD) |
|---|---:|---:|---:|---:|
| A | 1.300680 | 1.287114 | 1.293590 | 1.293795 (0.006785) |
| B' | 1.277996 | 1.275198 | 1.275858 | 1.276351 (0.001463) |
| C | 1.277748 | 1.274950 | 1.274079 | 1.275592 (0.001917) |

**Table 3.** Paired differences in final validation BPB.

| Paired difference | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---:|---:|---:|---:|
| B' - A | -0.022684 | -0.011916 | -0.017732 | -0.017444 |
| C - B' | -0.000248 | -0.000248 | -0.001779 | -0.000758 |

The B'-minus-A difference is negative in all three seeds. Its mean, -0.017444, is larger in magnitude than the 0.0137 yardstick. One seed's difference, -0.011916, is smaller than the yardstick, so the mean rather than each seed carries the comparison. The C-minus-B' differences are negative in all seeds (-0.000248 to -0.001779). Their mean, -0.000758, is smaller in magnitude than the yardstick. Median step time for seed 42 was 1273.24 ms for A, 1986.50 ms for B', and 1727.87 ms for C. C was faster than B' in these device-batch-2 runs. [LM]

### 5.2 Long pilot

**Table 4.** Validation BPB at selected steps for the single-seed pilot, with paired gaps.

| Step | A | B' | C | B' - A | C - B' |
|---|---:|---:|---:|---:|---:|
| 1500 | 1.191812 | 1.193531 | 1.191712 | +0.001719 | -0.001819 |
| 3000 | 1.102528 | 1.104800 | 1.102870 | +0.002272 | -0.001930 |
| 4250 | 1.060732 | 1.061305 | 1.059890 | +0.000573 | -0.001415 |
| 4500 | 1.056270 | 1.056153 | 1.054840 | -0.000117 | -0.001313 |
| 4578 (final) | 1.055337 | 1.055067 | 1.053788 | -0.000270 | -0.001279 |

The 600-iteration B'-minus-A advantage (mean -0.017444) is not present at 4,578 iterations. B' is above A at steps 1500, 3000, and 4250, by up to +0.002272 at step 3000. B' is below A only at steps 4500 and 4578, by -0.000117 and -0.000270. C is below B' at every reported step from 1500 onward. The final C-minus-B' gap is -0.001279, and the gaps across the five reported steps range from -0.001279 to -0.001930. [PL]

One seed gives no interval. The pilot configuration differs from the 600-iteration comparison, so these gaps cannot be attributed to FiLM or to state content. The 0.0137 yardstick comes from a three-seed 600-iteration run and is not applied to the pilot. [PL, PW, CF]

The validation curve shows no visible break at the resume point. Between steps 3000 and 3250 the change was -0.009986 for A, -0.010200 for B', and -0.010394 for C. [PL]

### 5.3 Decay screen and ablations

At 600 iterations on seed 43, D's final validation BPB was 1.275386, and D_const's was 1.274947. The seed-43 references were 1.275198 for B' and 1.274950 for C. D did not reach a lower BPB than D_const. [DS]

Replacing D's trained state with a constant (V1) raised its BPB by +0.071142. Cross-row shuffling (V2) raised it by +0.001628, and within-row time reversal (V3) by +0.001070. For D_const all three changes were zero, as expected for a constant state. [DS]

The V1 result shows that D's trained network depends on its state at evaluation time. The same network does not reach a lower BPB than the constant-state arm, D_const. V1 is one checkpoint, and it is not a comparison of two trained arms. [DS, LM]

### 5.4 Recall: per-delay accuracy

Each cell shows SALIENT / ORDINARY accuracy, with 1,024 rows per cell.

**Table 5.** Per-delay accuracy, D and D_const, seeds 43 and 101.

| Run | Delay 16 | Delay 48 | Delay 192 | Delay 320 | Delay 448 | Delay 576 | Delay 704 | Delay 832 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D, seed 43 | 0.996094 / 0.998047 | 0.998047 / 0.999023 | 0.063477 / 0.062500 | 0.063477 / 0.063477 | 0.060547 / 0.061523 | 0.062500 / 0.063477 | 0.061523 / 0.061523 | 0.062500 / 0.061523 |
| D_const, seed 43 | 0.997070 / 1.000000 | 1.000000 / 1.000000 | 0.063477 / 0.061523 | 0.055664 / 0.059570 | 0.067383 / 0.061523 | 0.058594 / 0.066406 | 0.064453 / 0.070312 | 0.057617 / 0.068359 |
| D, seed 101 | 0.997070 / 0.999023 | 1.000000 / 1.000000 | 0.063477 / 0.062500 | 0.062500 / 0.062500 | 0.062500 / 0.062500 | 0.062500 / 0.062500 | 0.062500 / 0.062500 | 0.062500 / 0.062500 |
| D_const, seed 101 | 0.997070 / 0.999023 | 1.000000 / 1.000000 | 0.063477 / 0.062500 | 0.062500 / 0.062500 | 0.062500 / 0.060547 | 0.061523 / 0.063477 | 0.063477 / 0.062500 | 0.062500 / 0.063477 |

Both D and D_const solve the WITHIN delays (16 and 48) for both classes, with accuracy between 0.996094 and 1.000000. Across all four runs, the CROSS cells lie between 0.055664 and 0.067383, against a chance accuracy of 0.0625. [RG]

### 5.5 Recall: pooled CROSS contrast

**Table 6.** Pooled CROSS accuracy (delays 192-832, both classes), gap, and no-context control.

| Run | CROSS SALIENT | CROSS ORDINARY | Gap (S - O) | Both-class 95% CI | No-context pooled | No-context flag |
|---|---:|---:|---:|---:|---:|---:|
| D, seed 43 | 0.062337 | 0.062337 | +0.000000 | [0.058105, 0.066813] | 0.061890 | WITHIN_NULL_CI |
| D_const, seed 43 | 0.061198 | 0.064616 | -0.003418 | [0.058675, 0.067220] | 0.062988 | WITHIN_NULL_CI |
| D, seed 101 | 0.062663 | 0.062500 | +0.000163 | [0.058431, 0.066895] | 0.062134 | WITHIN_NULL_CI |
| D_const, seed 101 | 0.062663 | 0.062500 | +0.000163 | [0.058429, 0.066976] | 0.062012 | WITHIN_NULL_CI |

The descriptive D-minus-D_const gap contrast is +0.003418 for seed 43 and 0.000000 for seed 101. Its two-pair mean is +0.001709. Two pairs give no interval or test, and the pre-set test was defined for the planned design. [RG, Q19]

All four pooled CROSS confidence intervals include chance (0.0625), and all four no-context pooled controls are flagged WITHIN_NULL_CI. [RG]

The two seed-101 runs have identical pooled CROSS values (accuracy 0.062663, gap +0.000163), although their per-delay values differ (Table 5). The records do not explain this, and it is listed as an open item. [RG]

### 5.6 Positive control

**Table 7.** Positive control: arm A without segment masking, seed 43, checkpoint 1,200.

| Delay | SALIENT | ORDINARY | No-context SALIENT | No-context ORDINARY |
|---|---:|---:|---:|---:|
| 16 | 1.000000 | 1.000000 | 0.064453 | 0.063477 |
| 48 | 0.998047 | 1.000000 | 0.061523 | 0.059570 |
| 192 | 0.995117 | 0.996094 | 0.061523 | 0.064453 |
| 320 | 0.998047 | 0.999023 | 0.056641 | 0.067383 |
| 448 | 0.997070 | 0.996094 | 0.063477 | 0.057617 |
| 576 | 0.997070 | 0.998047 | 0.065430 | 0.057617 |
| 704 | 1.000000 | 0.997070 | 0.064453 | 0.061523 |
| 832 | 0.998047 | 0.999023 | 0.062500 | 0.058594 |

Arm A recalls both classes at 0.995117-1.000000 across CROSS delays 192-832, while its no-context control stays at 0.056641-0.067383. [PC]

The benchmark therefore detects cross-segment recall when attention is not blocked. The near-chance D and D_const results are thus not produced by a harness that cannot register cross-segment recall. The control tests the readout and harness on an attention-based path. It does not test a state-based path, and it uses one seed. [PC]

## 6. Analysis: why a decay-based state cannot carry a single write

Under $E_t = \lambda E_{t-1} + (1-\lambda)\,x_t$, a single write $x_0$ contributes $(1-\lambda)\lambda^{\tau}$ to the state $\tau$ steps later. Maximising over $\lambda$ gives $\lambda^* = \tau/(\tau+1)$ and a maximum of

$$\frac{1}{\tau+1}\left(\frac{\tau}{\tau+1}\right)^{\tau} \approx \frac{1}{e\,\tau}.$$

**Table 8.** Single-write contribution $(1-\lambda)\lambda^{\tau}$ for the D decay set at delays 192 and 832, with the continuous maximum and the bound $1/(e\tau)$.

| Decay (lambda) | tau = 192 | tau = 832 |
|---|---:|---:|
| 0.9 | 1.6389e-10 | 8.5068e-40 |
| 0.95 | 2.6418e-06 | 1.4622e-20 |
| 0.975 | 1.9356e-04 | 1.7774e-11 |
| 0.9875 | 1.1170e-03 | 3.5627e-07 |
| 0.99375 | 1.8754e-03 | 3.3920e-05 |
| 0.996875 | 1.7134e-03 | 2.3116e-04 |
| 0.9984375 | 1.1573e-03 | 4.2540e-04 |
| 0.99921875 | 6.7239e-04 | 4.0774e-04 |
| Grid maximum | 1.8754e-03 | 4.2540e-04 |
| Continuous maximum | 1.9111e-03 | 4.4190e-04 |
| Bound 1/(e tau) | 1.9160e-03 | 4.4216e-04 |

For $\tau = 192$, the largest contribution over the D decay set is 1.8754e-03 (at $\lambda$ = 0.99375), against a continuous maximum of 1.9111e-03. For $\tau = 832$, it is 4.2540e-04 (at $\lambda$ = 0.9984375), against a continuous maximum of 4.4190e-04. An unsourced estimate of 2.3e-4 for $\tau = 832$ matches $\lambda$ = 0.996875 (2.3116e-04), not the grid maximum. [AN]

This is a back-of-envelope bound on the information one write can carry in a decay-based state, measured in units of the drive. The state update is a convex combination, so the drive values bound the state [Q22]. It is not an experiment. It does not bound total information from many overlapping writes, and it does not cover a state with an explicit write path. [AN]

## 7. Errors, corrections, and interrupted runs

### 7.1 One-token label leak

An earlier indexing error let the FiLM state at position $p$ depend on the token scored at $p$. Through an off-by-one index, the state at each position could depend on the token that position was scored on. The B'-seed-42 result of 1.226305 BPB, against a baseline of 1.301217 (a gain of 0.0749), measured this leak and is withdrawn. The E13-fixed verdict was withdrawn pending re-measurement, and BPB measurements from that period are not used here. [ER, Q07]

Fix commit 969e067 corrected the indexing. The regression test `diagnostics/test_no_label_leak.py` randomises the FiLM weights before it checks them, and the sweep launcher now runs it as a hard pre-flight gate. [Q08]

The 600-iteration comparison code (5566d90), the pilot driver (1cf28c1), and the decay screen (5bb8933) all descend from the fix. The ancestry of the recall-grid code is not recorded in its results **[UNVERIFIED]**. The later code review records the historical leak as not reproduced in the current tree. [Q07, Q08, Q24]

### 7.2 Saturation of the event projection

An earlier investigation found that the learned event projection could grow without bound. Normalising its output controlled that mechanism, while a separate PE/FiLM amplification path remained. A shorter bound check passed. A 3000-step run without abort was then evaluated against two criteria. The first required max|E| <= 0.98 at every step. The second required that the windowed maximum over the last 1000 steps should not rise. The run failed both. The first crossing was at step 1353, the peak was 0.9848 at step 1675, 151 steps exceeded 0.98, and the final value was 0.9737. [ER, Q11]

The stabilisation should therefore not be described as long-run stable. [ER]

### 7.3 Arm-C pace

A short-run diagnostic measured arm C at about 2.5 times the per-step time of B'. [ER] An earlier pace report attributed the slowdown to VRAM pressure (96% of 8 GB, with repeated allocator flushes). The repository records withdraw that explanation. Peak VRAM was flat and nearly identical (7727 vs 7744 MiB). Shared GPU memory was 14.5 MiB, so no spill to system memory was measured. Idle VRAM before the run was 507 MiB. The cause is recorded as open. [ER, Q09]

An unsourced account describes this slowdown as a VRAM spill at device batch 4. The records do not support a spill. E18 does not state its device batch size. The preceding E17 A/B flags use device batch 4 (line 2173). [ER, Q10]

In the device-batch-2 comparison (Section 5.1), C was faster than B' per step. The 2.5x result therefore does not reproduce in that setting, and its cause is not identified. [LM]

### 7.4 Recall calibration history

The pre-repair calibration stop file says both answer weights failed and training should stop. After the generator was repaired and audited, weight 20 passed the gate and was frozen. The main recall runs followed the repaired calibration. These are two versions of the procedure, not repeated measurements of one unchanged calibration. [CB, Q05]

### 7.5 Interrupted and resumed runs

**Long pilot, arm C.** The last complete checkpoint before the interruption was step 3000 (driver log line 57, at 01:53:55). The next driver event was a stale-lock takeover at 06:01:46, about 4.1 hours later (line 58, old PID 29624). [PL]

The driver then resumed arm C from step 3000 (line 61). Two resume attempts failed. The first failed because the checkpoint's state-dict keys did not match the model (`train.log` line 3495). Commit 0de329a, "Fix RDA checkpoint resume loading," was made between the first and second attempts. The second attempt failed in the optimizer step on a None gradient (`train.log` line 3662). Commit d5ceacd, "Preserve optimizer references on RDA resume," was made before the third attempt. The third attempt completed to step 4578 (`train.log` line 3730, driver line 79). The driver counted two retries for arm C. [PL, Q12, Q15]

The validation curve shows no visible break at step 3000 (Section 5.2). The checkpoint metadata holds an identical data position for A, B', and C at step 3000 (pq_idx 3, rg_idx 7, epoch 1), and the resume path reads that metadata (`scripts/base_train.py` line 450). The driver marked the resumed data position as "approximate" and cleared that note after the next checkpoint. The records do not explain the annotation; it is an open item. [PL, Q13, Q14]

The cause of the stop is **[UNVERIFIED]**. An unsourced account attributes it to a Windows servicing restart. The retained logs record only the stale-lock takeover. [PL]

**Recall driver.** The recall driver stopped after the eval-D_const-101 subprocess exited with code 0. It wrote no completion record, and its log ends at line 3533. The cause is **[UNVERIFIED]**. [RG, Q17]

## 8. Discussion

### 8.1 What the evidence supports

At 600 iterations and three seeds, both RDA arms ended with lower final validation BPB than the baseline in every seed. The mean A-to-RDA difference exceeds the pre-set yardstick. This is consistent with an effect of the shared RDA/FiLM pathway at short training, for example added capacity. It is the only language-model difference in this report that exceeds the yardstick. [LM]

At 600 iterations, dynamic-state B' and constant-state C differ by less than the yardstick. [LM]

### 8.2 What the evidence does not support

- **The 600-iteration advantage is not a durable benefit.** It is absent in the single-seed pilot at about 150 million tokens (final B'-minus-A of -0.000270). [PL]
- **No demonstrated state-content effect.** B' and C differ by -0.000758 at 600 iterations and by -0.001279 at the pilot's final step. The pilot is one seed. [LM, PL]
- **No cross-segment recall from the decay state.** D and D_const sit at chance on CROSS delays in both seeds. The positive control shows the benchmark detects cross-segment recall when attention is available, so these nulls are interpretable at the level of this harness. [RG, PC]
- **Decay-based state carries little per write.** Section 6 bounds the contribution of a single write for these decays. [AN]
- **No claim about subjective experience.** The drive names are design labels. [Q01]

### 8.3 What would be tested next

1. A state with an explicit content write path, so that a write's effect is not limited by the decay bound. Such a variant must first pass the causality, leak, and equivalence tests on CPU.
2. Completing the planned paired training-seed design for recall (seeds 43 and 101-104), estimating training-seed variance, and confirming the minimum effect of interest before interpreting any null.
3. Replicating the pilot over more seeds, and running the 600-iteration and pilot configurations side by side, so that width, window pattern, and training length are not confounded.

## 9. Limitations

- **Scale and data.** One natural-language corpus (FineWeb-Edu) and one synthetic recall task. The model is toy scale (depth 8; widths 768 and 512). Training is short: 600 iterations for the comparison, 4,578 iterations (about 150 million tokens) for the pilot, and 1,200 iterations for recall. [CF]
- **Seeds.** Three seeds at 600 iterations, one seed in the pilot, and two seeds in the recall grid. The planned recall design has five seeds, and the older claim rule had three. Two seeds meet neither. The reason only two were run is **[UNVERIFIED]**. No interval or test is reported for any two-seed contrast. [RG, CF]
- **Configuration.** The pilot differs from the 600-iteration comparison in width (512 vs 768), window pattern (SSSL vs L), and validation interval (250 vs 50). Its head dimension uses the default. The pilot is not a continuation of the comparison, so width and window cannot be separated from training length. [CF]
- **Reused runs.** The seed-42 B' and C results in the 600-iteration comparison were taken from earlier device-batch-2 logs and not retrained. [CF, LM]
- **Yardstick status.** The 0.0137 value comes from a baseline seed spread, and the records describe it as a proposal. It is a yardstick, not a test. [PW, Q20]
- **Interrupted runs.** The pilot's arm C was resumed through two code fixes. The recall driver stopped without a completion record. The D-101 and D_const-101 entries were promoted from a temporary artifact. [PL, RG]
- **Identical pooled values.** The two seed-101 runs share pooled CROSS values, though their per-delay values differ. This is not explained. [RG]
- **Row-value coincidence.** In the recall grid, 516 of 8,192 rows (6.30%) have equal SALIENT and ORDINARY values. This is close to the 1/16 rate expected from independent 16-way draws. Its effect on scoring has not been analysed. [RG, AN]
- **Descriptive look.** The seed-43 recall rows were viewed descriptively before the power amendment. [Q19]
- **Positive control.** One seed, attention-based. It validates the readout and harness, not a state-based path. [PC]
- **Exploratory variant (no result).** A gated content-write variant of the state (branch `mech2-20261015`; commits ca5d85a, ffdc9ce, 0e6ef3a, 3e7466f) was implemented with tests for causality, leakage, and equivalence (`tests/test_gated_label_leak.py`, `tests/test_gated_optimizer.py`, `tests/test_mechanism_gated.py`). An unsourced account reports that those tests passed on CPU. No pass record was located **[UNVERIFIED]**. Its recall run did not complete. The status record (`STATUS_followup.md`) lists MECH_GATED as failed with the error "Smoke directory unexpectedly has checkpoint 20" (line 5). An unsourced account also reports a numerical failure, with non-finite state at about steps 12-19. No record of that was located **[UNVERIFIED]**. The cause is not isolated. No result from this variant is reported, and it is neither evidence for nor against any claim here. [MX]
- **Metric.** BPB may not track the behaviour of interest. [Q23]
- **Planning notes.** `METHODS.md`, `EXPERIMENTS.md`, and `LIMITATIONS.md` still describe some runs as not run. They are not the source of truth for this draft. [OPEN_ITEMS.md]

## 10. Reproducibility

- **600-iteration comparison.** Code commit 5566d90, launcher `run_ab_v3_dbs2_full.ps1`, outputs in `results/ab_v3_dbs2/`. Environment capture is in `logs/ab_v3_dbs2_full_env.txt`. [CF, LM]
- **Long pilot.** Driver `run_lm_pilot.ps1` (commit 1cf28c1; resume fixes 0de329a and d5ceacd). Outputs are in `lm_pilot/{A,Bprime,C}/val_bpb.csv`, `lm_pilot/state.json`, and `lm_pilot/driver.log`. [PL]
- **Decay screen.** Branch `exp/multiscale-decay-20261011`, code commit 5bb8933, launcher `run_ab_v3_dbs2_decay.ps1`, ablations `diagnostics/ablate_state.py`. Outputs are in `results/multiscale_decay/`. [DS]
- **Recall grid.** Driver `run_recall_main_d_dconst.ps1`, evaluation `diagnostics/eval_recall.py`, outputs `results/recall_benchmark/main/main_d_dconst.json` and `.txt`. The provenance note is in Section 4.4. [RG, CF]
- **Positive control.** `recall-A-nomask-s43-w20/` with `eval_nomask_wrapper.py`, `eval_result.txt`, and the driver log. [PC]
- **Calibration and audit.** `results/recall_benchmark/calib/` (`generator_audit.txt`, `answer_weight_selection.txt`, `manip_check-D-s43-w20.txt`, which is UTF-16, and `power_analysis_did_gpu_v6_n6144.txt`). [CB, MC, PW]
- **Leak test.** `diagnostics/test_no_label_leak.py`, run as the pre-flight gate. [Q08]
- **Claims.** [claims.md](claims.md) gives the file and key or line for every number. The parser that produced it is not in the repository. Section 6 values are computed from the formula in the AN rows. [OPEN_ITEMS.md]
- **Hardware and software.** Windows; NVIDIA GeForce RTX 5060 Ti (from the positive-control log); Python from the project virtual environment. [PC]
- No GPU work was performed for this draft. All numbers were parsed from retained files.

## 11. Conclusion

At toy scale, the mean 600-iteration A-to-RDA difference exceeds the pre-set yardstick across three seeds. That difference is absent in a single-seed pilot at about 150 million tokens. The dynamic-state and constant-state arms do not differ beyond the yardstick. The decay-based state shows no cross-segment recall in two seeds, while the benchmark detects that recall when attention is available. The results support neither a demonstrated state-content benefit nor any claim about subjective emotion.
