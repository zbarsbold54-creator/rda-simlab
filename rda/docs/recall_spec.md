Purpose: test whether a persistent internal state retains surprising events better than ordinary ones, beyond the model's attention range.
Row layout (T=1024, BOS first): filler stream from a first-order Markov chain over filler vocab F (|F|=512 reserved token ids). The transition matrix is doubly stochastic (stationary distribution uniform, verified to 1e-3), each token has ~4 likely successors.
Events: each row has exactly 2 events, 1 SALIENT and 1 ORDINARY, order randomized 50/50. An event = lead region (8 tokens) + [K][V]. K from key pool (16 ids), V from value pool (16 ids), distinct keys within a row, both sampled uniformly and independently of class. SALIENT lead = 8 tokens sampled uniformly from F (anomaly burst, breaks the chain). ORDINARY lead = 8 tokens of normal chain filler. Nothing else differs between classes.
Queries: one per event, block [Q][K] where the target at K's position is V (the V token then follows teacher-forced). Both events are queried every row, so queries carry no information about salience. Chance = 1/16 = 0.0625.
Delay = tokens from event [K] position to query [K] position. Eval delay grid: WITHIN-segment {16, 48}; CROSS-segment {192, 320, 448, 576, 704, 832}. Training delays: uniform over the same feasible range. Event position p sampled uniformly over the feasible interval for the given delay, by the SAME rule for both classes. For WITHIN delays the event and query must be in the same 128-token segment; for CROSS delays in different segments. Events/queries must not overlap.
Attention: block-diagonal causal mask with segment length 128 (flag --segment-len 128). Across segments, attention carries no information; only the RDA state can.
Controls and arms: A (no RDA), B', D (multi-timescale decay), plus constant-state controls C and D_const. A with the segment mask is the hard no-memory floor.
Primary outcome: recall accuracy on the answer token. Primary estimand: within-row paired difference (SALIENT minus ORDINARY) on CROSS delays, per arm, and its difference-in-differences against the arm's constant-state control.
Claim rules (fixed in advance):
 (i) a manipulation check must pass: the state is measurably more elevated at salient than at ordinary events for the arm tested;
 (ii) the arm must exceed chance on CROSS delays (bootstrap 95% CI over eval rows above 0.0625);
 (iii) the difference-in-differences vs the constant-state control must have a 95% CI excluding 0 in the predicted direction, pooled over 3 seeds, with the same sign in all 3 seeds;
 (iv) FineWeb BPB is reported for every arm against A trained on the same data mixture.
If an arm is at or below chance on CROSS delays, report a CAPACITY NULL, not a salience null.
Effect-size threshold and eval row counts: set by a power analysis (next prompt) and committed before any arm is evaluated.
Seeds: train rows from seed stream 0 upward, eval rows from seed base 1,000,000,000, never overlapping. Rows are deterministic from (seed, index).

Clarifications after segment-invariance validation
(a) WITHIN delays (16, 48) are attention-reachable; untrained models can recall there via in-context induction (untrained A: 0.203 at delay 16, SALIENT). WITHIN buckets are a positive control after training, and the no-memory floor test applies only to CROSS delays.
(b) Scoring uses argmax restricted to the 16 value ids (chance 0.0625), and full-vocabulary accuracy is reported as a secondary column.
(c) The first floor-check rule was misapplied to WITHIN buckets and was corrected on 2026-10-11.

Pre-registered additions (calibration, floor, manipulation check)
- Floor: claim rule (ii) uses the permutation null from Part 1c, pooled over CROSS buckets, instead of the nominal 1/16. The arm must exceed it: bootstrap 95% CI lower bound above the null's upper bound.
- Calibration: train A on the task for 1200 iterations with answer weight 5; if A's WITHIN delay-16 accuracy is below 0.80 for either class, retrain with answer weight 20; if still below, stop and report. The first setting that passes is frozen for all arms.
- Capacity probe: D and D_const trained with the frozen settings; CROSS recall pooled over CROSS buckets and classes.
- Manipulation check: on the trained D checkpoint, 2048 eval rows, the paired within-row difference (SALIENT minus ORDINARY) of the 8-dim state L2 norm at the event [K] position, standardized (paired Cohen's d). Pass if d >= 0.5 and the 95% bootstrap CI over rows excludes 0. Also report per-drive standardized differences. Training seed: 43 (calibration checkpoint, answer weight 20, results b3b798a). On FAIL (d < 0.5 or 95% bootstrap CI over rows includes 0): the main grid is not launched; the result is reported and the decision is made by the experimenter.
- Interpretation guard: CROSS recall at or below the null for an arm is reported as a CAPACITY NULL; a failed manipulation check makes any salience null uninterpretable.

Generator amendment (2026-10-11)
The previous generator assigned keys and values with seed-dependent affine maps of the row index. In each class, value was therefore a deterministic function of key and index modulo 16; the mapping also changed with the seed. A model could memorize the training stream's mapping by weight, so this task did not establish in-context recall.

The corrected generator samples each row from an RNG seeded by (seed, index). It samples two distinct key symbols uniformly from the key pool and samples two value symbols uniformly and independently from the value pool, without using key, class, or index as a symbol mapping. Event slots assign those sampled symbols to the randomized SALIENT/ORDINARY classes. Eval-grid strata balance the value marginal separately by delay bucket and class through stratified assignment; that balancing must not encode a deterministic mapping from index, key, or class to value. Training and evaluation use identical pools and salts with disjoint (seed, index) streams. Training advances through every row index rather than selecting only even indices.

The existing calibration gate and stop rule are unchanged: train A for 1200 iterations at answer weight 5; if either delay-16 class is below 0.80, retry once at weight 20; if either class remains below 0.80, stop and report without training D or D_const.

No-context replication test (2026-10-11)
The no-context stop rule fired on D_const on the seed-1,000,000,000 eval stream (2048 rows): delay 320 SALIENT accuracy 0.078125 vs permutation-null CI [0.019531, 0.070312]; delay 832 ORDINARY accuracy 0.082031 vs permutation-null CI [0.023438, 0.078125]. The manipulation check and the power analysis were not run as a result.

Replication test (pre-registered before the rerun):
- Fresh eval stream: seed base 2,000,000,000, indices 0 to N-1 with N = 16384 rows, giving 2048 rows per class/delay cell on the same stratified, value-balanced grid. The stream is disjoint from the training stream (seed 0) and from the seed-1,000,000,000 eval stream by seed construction; disjointness is verified before the run.
- Models: D_const (recall-Dconst-s43-w20) and D (recall-D-s43-w20), same checkpoints, step 1200.
- Per-cell rule unchanged: flag a cell if no-context accuracy exceeds its permutation-null 95% CI.
- (a) Pooled test per model: no-context accuracy over all 16 cells with a permutation-null 95% CI, bootstrapped over rows and paired by row.
- (b) Holm correction across all 16 no-context cells (8 delays x 2 classes) per model at alpha 0.05, reported next to the uncorrected flags. Per-cell p-value: one-sided paired bootstrap over the cell's rows, p = fraction of 5000 resamples of (no-context correct minus permutation-null match) with mean <= 0.
- Decision rule: if no cell is significant under Holm AND the pooled test is within its null interval for both models, the original exceedance is judged a multiplicity artifact and the stop is lifted. Otherwise the stop stands and the leak path must be investigated before any further claims.

For FUTURE runs only, the no-context control uses the pooled test plus Holm correction instead of the uncorrected per-cell rule.

Power analysis
--------------
Minimum effect of interest: 0.02, defined as the D vs D_const
difference in the SALIENT-minus-ORDINARY gap pooled over CROSS delays
192-832.

N for the evaluation design is the total paired evaluation rows per
seed across all eight delay buckets (2 WITHIN, 6 CROSS 192-832), with
1,024 rows per delay at N=8192. Each row contains one SALIENT and one
ORDINARY event. The power-analysis N is the pooled CROSS count per
class: N_total x 6/8 = 6,144 for the planned N_total=8,192-row
evaluation. Thus, the planned design is K=5 training seeds per arm
(D and D_const), 8,192 total eval rows per seed (6,144 pooled CROSS
per class).

The primary test is the one-sided exact label-permutation test across
the 2K per-seed eval gaps, rejecting at p <= 0.05; its minimum
achievable p-value is 1/252. The paired sign-flip test by seed index
(minimum p=0.03125 at K=5) and the threshold rule are secondary. A,
B', and C are secondary comparisons run at 3 seeds and are outside
this power analysis.

At delta=0.02 and K=5, empirical power is:

| Power-analysis N (pooled CROSS per class) | sigma_seed | Sign-flip | Threshold | Permutation | Source |
|---:|---:|---:|---:|---:|---|
| 4,096 | 0.005 | 0.8560 | 0.4410 | 0.9770 | v5 |
| 4,096 | 0.01  | 0.5475 | 0.1375 | 0.7830 | v5 |
| 4,096 | 0.02  | 0.2320 | 0.0290 | 0.3825 | v5 |
| 6,144 | 0.005 | 0.9060 | 0.5775 | 0.9970 | v6 |
| 6,144 | 0.01  | 0.6010 | 0.1580 | 0.8160 | v6 |
| 6,144 | 0.02  | 0.2435 | 0.0345 | 0.4105 | v6 |
| 8,192 | 0.005 | 0.9305 | 0.6405 | 0.9975 | v5 |
| 8,192 | 0.01  | 0.5920 | 0.1685 | 0.8380 | v5 |
| 8,192 | 0.02  | 0.2470 | 0.0310 | 0.3995 | v5 |

Power at delta 0.02, sigma_seed 0.01 (0.816) is within about 2 Monte
Carlo standard errors of the 0.80 target.

The earlier v5 N=4096 and N=8192 rows use N as the pooled CROSS
count per class; v6 N=6144 matches the planned 8,192 total eval rows.
These powers are conditional on the assumed sigma_seed, which must be
estimated from the five real training seeds of both D_const and D. If
that estimate exceeds 0.01, report the minimum detectable effect at
80% power for K=5 and power-analysis N=6144 rather than interpreting a
null result as evidence of absence.

The stop/null-reporting rule is to report the minimum detectable effect
at 80% power for the chosen K and N. The CPU-GPU validation (differences
0.016, 0.025, and 0.0195, within Monte Carlo error) applies only to the
earlier paired-row power analysis, not this difference-in-differences
model with training-seed variance. Bootstrap reps were removed because
no reported quantity used them.

Before this amendment, the seed-43 evaluation rows for D and D_const
(pooled CROSS SALIENT/ORDINARY accuracy and gap, N=8192, eval seed base
3,000,000,000) had been viewed descriptively; no seeds 101-104 results
existed, and no hypothesis test (permutation, sign-flip or threshold)
had been run. The amendment corrects the definition of N and adds the
N=6144 power table; it changes no test, threshold, minimum effect or K,
and was prompted by the 6,144 denominator, not by outcome values. Seed
43 also informed the calibration choices (answer weight 20, no-context
stop lift).
