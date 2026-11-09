# Power analysis (draft for recall_spec.md)

## Setup and notation

- **Evaluation N** is the total paired evaluation rows per seed across
  all eight delay buckets (2 WITHIN, 6 CROSS 192-832); at N=8192 there
  are 1,024 rows per delay. Each row has one SALIENT and one ORDINARY
  event, both queried.
- **Power-analysis N** is the pooled CROSS count per class:
  N_total x 6/8. Thus, the planned N_total=8192 rows give N=6144
  pooled CROSS rows per class for the power analysis.
- In the earlier tables, **"seeds" means eval blocks only** (independent
  eval row blocks), not training seeds. The training-seed variance is
  modeled separately below.
- **Primary contrast:** the D vs D_const difference in the
  SALIENT-minus-ORDINARY gap, pooled over CROSS delays 192-832
  (delay buckets 192, 320, 448, 576, 704, 832).
- Base accuracy per class: 0.0638020833333333 (D_const CROSS pooled).
- Results below are from `power_analysis_did_gpu_v5.txt` (N=4096 and
  N=8192) and `power_analysis_did_gpu_v6_n6144.txt` (N=6144): seed
  20261011, 2,000 simulations per cell, with no bootstrap.

## Minimum effect of interest

**0.02**, defined as the D vs D_const difference in the
SALIENT-minus-ORDINARY gap pooled over CROSS delays 192-832.

## Pre-registered design

- K=5 independent training seeds per arm (D and D_const).
- 8,192 total eval rows per training seed (6,144 pooled CROSS per class,
  the power-analysis N).
- Primary test: one-sided exact label-permutation test across the 2K
  per-seed eval gaps; reject at p <= 0.05. The minimum achievable p is
  1/252.
- Secondary tests: paired sign-flip by seed index (K=5, minimum p
  0.03125) and the threshold rule.
- A, B', and C are secondary comparisons at 3 seeds and are outside
  this power analysis.
- At delta=0.02, K=5, planned power-analysis N=6144, v6 permutation
  power is 0.9970 for sigma_seed=0.005, 0.8160 for sigma_seed=0.01,
  and 0.4105 for sigma_seed=0.02.
- Power at delta 0.02, sigma_seed 0.01 (0.816) is within about 2 Monte
  Carlo standard errors of the 0.80 target.
- Before this amendment, the seed-43 evaluation rows for D and D_const
  (pooled CROSS SALIENT/ORDINARY accuracy and gap, N=8192, eval seed
  base 3,000,000,000) had been viewed descriptively; no seeds 101-104
  results existed, and no hypothesis test (permutation, sign-flip or
  threshold) had been run. The amendment corrects the definition of N
  and adds the N=6144 power table; it changes no test, threshold,
  minimum effect or K, and was prompted by the 6,144 denominator, not
  by outcome values. Seed 43 also informed the calibration choices
  (answer weight 20, no-context stop lift).

## Model and tests

For each arm (D, D_const), K independent training seeds. Each seed's
true SALIENT-minus-ORDINARY gap is the arm gap plus
N(0, sigma_seed^2). Eval gaps add paired-row sampling noise from N
paired rows per seed. The estimated contrast is the mean over seeds of
(gap_D - gap_Dconst), paired by seed index.

All tests use a one-sided alternative toward D > D_const and reject at
**p <= 0.05**:

1. **Preferred primary test: exhaustive label permutation** over the
   2K per-seed eval gaps, assigning K gaps to each arm. There are 20
   arrangements for K=3 and 252 for K=5. Minimum achievable p is 0.05
   (K=3) and 1/252 (K=5).
2. **Secondary: exact sign-flip test.** It pairs D and D_const by seed
   index, although that pairing is arbitrary because their training
   seeds are independent. The one-sided test enumerates all 2^K sign
   patterns. Minimum achievable p is 0.125 (K=3) and 0.03125 (K=5).
3. **Secondary: threshold rule.** Reject when the estimated difference
   exceeds the larger of the within-arm seed spreads, where each spread
   is max minus min of the K eval gaps. Its null size is 0.0000-0.0275
   (about 0.001 for K=5), so it is conservative and is not a calibrated
   test.

The 5,000 bootstrap reps were removed because no reported quantity used
them. The threshold rule uses the direct max-minus-min spread; the
sign-flip and permutation tests are exhaustive.

## Stop / null-reporting rule

Report the **minimum detectable effect at 80% power** for the chosen K
and N (the smallest tested true difference-in-differences reaching 80%
power). If no grid point reaches 80%, report that the effect is not
detectable at the chosen K and N and state the largest observed power.

No test reaches 80% power at delta = 0.01. The maximum observed power
at delta = 0.01 is 0.5005 for sign-flip, 0.3035 for the threshold rule,
and 0.7240 for permutation.

All power results are conditional on the assumed sigma_seed. Estimate
sigma_seed from the five real training seeds of both D_const and D
before choosing K and N. If the estimate exceeds 0.01, report the
minimum detectable effect at 80% power for K=5 and power-analysis
N=6144 (8,192 total eval rows); do not interpret failure to detect the
effect as evidence of its absence.

## Delta=0 size

Entries are empirical rejection rates at delta=0: v5 for its grid and
v6 for K=5, N=6144.

| K | N (pooled CROSS/class) | sigma_seed | Sign-flip size | Threshold size | Permutation size |
|---|---:|---:|---:|---:|---:|
| 3 | 1536 | 0.005 | 0.0000 | 0.0255 | 0.0450 |
| 3 | 1536 | 0.01  | 0.0000 | 0.0260 | 0.0495 |
| 3 | 1536 | 0.02  | 0.0000 | 0.0195 | 0.0425 |
| 3 | 4096 | 0.005 | 0.0000 | 0.0275 | 0.0510 |
| 3 | 4096 | 0.01  | 0.0000 | 0.0190 | 0.0475 |
| 3 | 4096 | 0.02  | 0.0000 | 0.0265 | 0.0545 |
| 3 | 8192 | 0.005 | 0.0000 | 0.0255 | 0.0440 |
| 3 | 8192 | 0.01  | 0.0000 | 0.0265 | 0.0560 |
| 3 | 8192 | 0.02  | 0.0000 | 0.0260 | 0.0545 |
| 5 | 1536 | 0.005 | 0.0325 | 0.0020 | 0.0475 |
| 5 | 1536 | 0.01  | 0.0300 | 0.0010 | 0.0420 |
| 5 | 1536 | 0.02  | 0.0340 | 0.0005 | 0.0480 |
| 5 | 4096 | 0.005 | 0.0280 | 0.0015 | 0.0405 |
| 5 | 4096 | 0.01  | 0.0295 | 0.0005 | 0.0430 |
| 5 | 4096 | 0.02  | 0.0290 | 0.0030 | 0.0380 |
| 5 | 6144 | 0.005 | 0.0320 | 0.0015 | 0.0465 |
| 5 | 6144 | 0.01  | 0.0310 | 0.0005 | 0.0495 |
| 5 | 6144 | 0.02  | 0.0320 | 0.0030 | 0.0540 |
| 5 | 8192 | 0.005 | 0.0290 | 0.0010 | 0.0495 |
| 5 | 8192 | 0.01  | 0.0230 | 0.0000 | 0.0380 |
| 5 | 8192 | 0.02  | 0.0340 | 0.0000 | 0.0465 |

With 2,000 simulations per cell, empirical size estimates have standard
error about 0.004-0.005 near size 0.05. K=5 permutation size averages
slightly below nominal because ties are counted conservatively using
the non-strict tail comparison.

## Full power table

The N=4096 and N=8192 rows are copied from the v5 log; the K=5,
N=6144 rows are copied from v6. Here N is the pooled CROSS count per
class. Delta=0 is included as the null-size cell; other deltas are the
simulated powers.

| K | N (pooled CROSS/class) | sigma_seed | delta | Sign-flip | Threshold | Permutation |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 1536 | 0.005 | 0.0 | 0.0000 | 0.0255 | 0.0450 |
| 3 | 1536 | 0.01 | 0.0 | 0.0000 | 0.0260 | 0.0495 |
| 3 | 1536 | 0.02 | 0.0 | 0.0000 | 0.0195 | 0.0425 |
| 3 | 1536 | 0.005 | 0.01 | 0.0000 | 0.1420 | 0.2385 |
| 3 | 1536 | 0.01 | 0.01 | 0.0000 | 0.1110 | 0.1900 |
| 3 | 1536 | 0.02 | 0.01 | 0.0000 | 0.0630 | 0.1150 |
| 3 | 1536 | 0.005 | 0.02 | 0.0000 | 0.4115 | 0.5765 |
| 3 | 1536 | 0.01 | 0.02 | 0.0000 | 0.2705 | 0.4200 |
| 3 | 1536 | 0.02 | 0.02 | 0.0000 | 0.1375 | 0.2350 |
| 3 | 1536 | 0.005 | 0.03 | 0.0000 | 0.7170 | 0.8550 |
| 3 | 1536 | 0.01 | 0.03 | 0.0000 | 0.5050 | 0.6920 |
| 3 | 1536 | 0.02 | 0.03 | 0.0000 | 0.2470 | 0.3745 |
| 3 | 1536 | 0.005 | 0.05 | 0.0000 | 0.9720 | 0.9965 |
| 3 | 1536 | 0.01 | 0.05 | 0.0000 | 0.8905 | 0.9625 |
| 3 | 1536 | 0.02 | 0.05 | 0.0000 | 0.5205 | 0.7060 |
| 3 | 4096 | 0.005 | 0.0 | 0.0000 | 0.0275 | 0.0510 |
| 3 | 4096 | 0.01 | 0.0 | 0.0000 | 0.0190 | 0.0475 |
| 3 | 4096 | 0.02 | 0.0 | 0.0000 | 0.0265 | 0.0545 |
| 3 | 4096 | 0.005 | 0.01 | 0.0000 | 0.2340 | 0.3725 |
| 3 | 4096 | 0.01 | 0.01 | 0.0000 | 0.1170 | 0.2200 |
| 3 | 4096 | 0.02 | 0.01 | 0.0000 | 0.0695 | 0.1350 |
| 3 | 4096 | 0.005 | 0.02 | 0.0000 | 0.6650 | 0.8205 |
| 3 | 4096 | 0.01 | 0.02 | 0.0000 | 0.3535 | 0.5265 |
| 3 | 4096 | 0.02 | 0.02 | 0.0000 | 0.1360 | 0.2410 |
| 3 | 4096 | 0.005 | 0.03 | 0.0000 | 0.9315 | 0.9765 |
| 3 | 4096 | 0.01 | 0.03 | 0.0000 | 0.6480 | 0.7930 |
| 3 | 4096 | 0.02 | 0.03 | 0.0000 | 0.2750 | 0.4215 |
| 3 | 4096 | 0.005 | 0.05 | 0.0000 | 0.9990 | 1.0000 |
| 3 | 4096 | 0.01 | 0.05 | 0.0000 | 0.9485 | 0.9880 |
| 3 | 4096 | 0.02 | 0.05 | 0.0000 | 0.5720 | 0.7315 |
| 3 | 8192 | 0.005 | 0.0 | 0.0000 | 0.0255 | 0.0440 |
| 3 | 8192 | 0.01 | 0.0 | 0.0000 | 0.0265 | 0.0560 |
| 3 | 8192 | 0.02 | 0.0 | 0.0000 | 0.0260 | 0.0545 |
| 3 | 8192 | 0.005 | 0.01 | 0.0000 | 0.3035 | 0.4665 |
| 3 | 8192 | 0.01 | 0.01 | 0.0000 | 0.1305 | 0.2310 |
| 3 | 8192 | 0.02 | 0.01 | 0.0000 | 0.0720 | 0.1315 |
| 3 | 8192 | 0.005 | 0.02 | 0.0000 | 0.7980 | 0.9085 |
| 3 | 8192 | 0.01 | 0.02 | 0.0000 | 0.3980 | 0.5715 |
| 3 | 8192 | 0.02 | 0.02 | 0.0000 | 0.1610 | 0.2755 |
| 3 | 8192 | 0.005 | 0.03 | 0.0000 | 0.9800 | 0.9960 |
| 3 | 8192 | 0.01 | 0.03 | 0.0000 | 0.7000 | 0.8395 |
| 3 | 8192 | 0.02 | 0.03 | 0.0000 | 0.2675 | 0.4100 |
| 3 | 8192 | 0.005 | 0.05 | 0.0000 | 1.0000 | 1.0000 |
| 3 | 8192 | 0.01 | 0.05 | 0.0000 | 0.9770 | 0.9935 |
| 3 | 8192 | 0.02 | 0.05 | 0.0000 | 0.6115 | 0.7880 |
| 5 | 1536 | 0.005 | 0.0 | 0.0325 | 0.0020 | 0.0475 |
| 5 | 1536 | 0.01 | 0.0 | 0.0300 | 0.0010 | 0.0420 |
| 5 | 1536 | 0.02 | 0.0 | 0.0340 | 0.0005 | 0.0480 |
| 5 | 1536 | 0.005 | 0.01 | 0.2285 | 0.0285 | 0.3975 |
| 5 | 1536 | 0.01 | 0.01 | 0.1740 | 0.0120 | 0.2675 |
| 5 | 1536 | 0.02 | 0.01 | 0.0815 | 0.0095 | 0.1565 |
| 5 | 1536 | 0.005 | 0.02 | 0.6165 | 0.2020 | 0.8630 |
| 5 | 1536 | 0.01 | 0.02 | 0.4295 | 0.0815 | 0.6505 |
| 5 | 1536 | 0.02 | 0.02 | 0.2195 | 0.0280 | 0.3635 |
| 5 | 1536 | 0.005 | 0.03 | 0.8845 | 0.5105 | 0.9920 |
| 5 | 1536 | 0.01 | 0.03 | 0.7505 | 0.2805 | 0.9370 |
| 5 | 1536 | 0.02 | 0.03 | 0.3990 | 0.0625 | 0.6045 |
| 5 | 1536 | 0.005 | 0.05 | 0.9975 | 0.9530 | 1.0000 |
| 5 | 1536 | 0.01 | 0.05 | 0.9770 | 0.7765 | 1.0000 |
| 5 | 1536 | 0.02 | 0.05 | 0.7450 | 0.3015 | 0.9335 |
| 5 | 4096 | 0.005 | 0.0 | 0.0280 | 0.0015 | 0.0405 |
| 5 | 4096 | 0.01 | 0.0 | 0.0295 | 0.0005 | 0.0430 |
| 5 | 4096 | 0.02 | 0.0 | 0.0290 | 0.0030 | 0.0380 |
| 5 | 4096 | 0.005 | 0.01 | 0.3835 | 0.0575 | 0.5905 |
| 5 | 4096 | 0.01 | 0.01 | 0.2165 | 0.0215 | 0.3350 |
| 5 | 4096 | 0.02 | 0.01 | 0.1025 | 0.0035 | 0.1640 |
| 5 | 4096 | 0.005 | 0.02 | 0.8560 | 0.4410 | 0.9770 |
| 5 | 4096 | 0.01 | 0.02 | 0.5475 | 0.1375 | 0.7830 |
| 5 | 4096 | 0.02 | 0.02 | 0.2320 | 0.0290 | 0.3825 |
| 5 | 4096 | 0.005 | 0.03 | 0.9860 | 0.8720 | 1.0000 |
| 5 | 4096 | 0.01 | 0.03 | 0.8500 | 0.4430 | 0.9805 |
| 5 | 4096 | 0.02 | 0.03 | 0.4365 | 0.0755 | 0.6535 |
| 5 | 4096 | 0.005 | 0.05 | 1.0000 | 0.9995 | 1.0000 |
| 5 | 4096 | 0.01 | 0.05 | 0.9960 | 0.9270 | 1.0000 |
| 5 | 4096 | 0.02 | 0.05 | 0.8075 | 0.3510 | 0.9605 |
| 5 | 6144 | 0.005 | 0.0 | 0.0320 | 0.0015 | 0.0465 |
| 5 | 6144 | 0.01 | 0.0 | 0.0310 | 0.0005 | 0.0495 |
| 5 | 6144 | 0.02 | 0.0 | 0.0320 | 0.0030 | 0.0540 |
| 5 | 6144 | 0.005 | 0.01 | 0.4315 | 0.0795 | 0.6700 |
| 5 | 6144 | 0.01 | 0.01 | 0.2260 | 0.0260 | 0.3665 |
| 5 | 6144 | 0.02 | 0.01 | 0.0925 | 0.0075 | 0.1730 |
| 5 | 6144 | 0.005 | 0.02 | 0.9060 | 0.5775 | 0.9970 |
| 5 | 6144 | 0.01 | 0.02 | 0.6010 | 0.1580 | 0.8160 |
| 5 | 6144 | 0.02 | 0.02 | 0.2435 | 0.0345 | 0.4105 |
| 5 | 6144 | 0.005 | 0.03 | 0.9940 | 0.9445 | 1.0000 |
| 5 | 6144 | 0.01 | 0.03 | 0.8735 | 0.4735 | 0.9885 |
| 5 | 6144 | 0.02 | 0.03 | 0.4370 | 0.0750 | 0.6605 |
| 5 | 6144 | 0.005 | 0.05 | 1.0000 | 1.0000 | 1.0000 |
| 5 | 6144 | 0.01 | 0.05 | 0.9960 | 0.9510 | 1.0000 |
| 5 | 6144 | 0.02 | 0.05 | 0.8075 | 0.3630 | 0.9630 |
| 5 | 8192 | 0.005 | 0.0 | 0.0290 | 0.0010 | 0.0495 |
| 5 | 8192 | 0.01 | 0.0 | 0.0230 | 0.0000 | 0.0380 |
| 5 | 8192 | 0.02 | 0.0 | 0.0340 | 0.0000 | 0.0465 |
| 5 | 8192 | 0.005 | 0.01 | 0.5005 | 0.1030 | 0.7240 |
| 5 | 8192 | 0.01 | 0.01 | 0.2250 | 0.0240 | 0.3820 |
| 5 | 8192 | 0.02 | 0.01 | 0.1075 | 0.0080 | 0.1680 |
| 5 | 8192 | 0.005 | 0.02 | 0.9305 | 0.6405 | 0.9975 |
| 5 | 8192 | 0.01 | 0.02 | 0.5920 | 0.1685 | 0.8380 |
| 5 | 8192 | 0.02 | 0.02 | 0.2470 | 0.0310 | 0.3995 |
| 5 | 8192 | 0.005 | 0.03 | 0.9990 | 0.9590 | 1.0000 |
| 5 | 8192 | 0.01 | 0.03 | 0.8770 | 0.4865 | 0.9835 |
| 5 | 8192 | 0.02 | 0.03 | 0.4455 | 0.0810 | 0.6730 |
| 5 | 8192 | 0.005 | 0.05 | 1.0000 | 1.0000 | 1.0000 |
| 5 | 8192 | 0.01 | 0.05 | 0.9990 | 0.9615 | 1.0000 |
| 5 | 8192 | 0.02 | 0.05 | 0.8255 | 0.3820 | 0.9655 |

## Smallest tested delta reaching 80% power

`>0.05` marks extrapolation: no tested delta reached 0.80; the maximum
observed power in the grid is shown.

| K | N (pooled CROSS/class) | sigma_seed | Sign-flip | Threshold | Permutation |
|---|---:|---:|---:|---:|---:|
| 3 | 1536 | 0.005 | >0.05 (max 0.0000) | 0.05 | 0.03 |
| 3 | 1536 | 0.01 | >0.05 (max 0.0000) | 0.05 | 0.05 |
| 3 | 1536 | 0.02 | >0.05 (max 0.0000) | >0.05 (max 0.5205) | >0.05 (max 0.7060) |
| 3 | 4096 | 0.005 | >0.05 (max 0.0000) | 0.03 | 0.02 |
| 3 | 4096 | 0.01 | >0.05 (max 0.0000) | 0.05 | 0.05 |
| 3 | 4096 | 0.02 | >0.05 (max 0.0000) | >0.05 (max 0.5720) | >0.05 (max 0.7315) |
| 3 | 8192 | 0.005 | >0.05 (max 0.0000) | 0.03 | 0.02 |
| 3 | 8192 | 0.01 | >0.05 (max 0.0000) | 0.05 | 0.03 |
| 3 | 8192 | 0.02 | >0.05 (max 0.0000) | >0.05 (max 0.6115) | >0.05 (max 0.7880) |
| 5 | 1536 | 0.005 | 0.03 | 0.05 | 0.02 |
| 5 | 1536 | 0.01 | 0.05 | >0.05 (max 0.7765) | 0.03 |
| 5 | 1536 | 0.02 | >0.05 (max 0.7450) | >0.05 (max 0.3015) | 0.05 |
| 5 | 4096 | 0.005 | 0.02 | 0.03 | 0.02 |
| 5 | 4096 | 0.01 | 0.03 | 0.05 | 0.03 |
| 5 | 4096 | 0.02 | 0.05 | >0.05 (max 0.3510) | 0.05 |
| 5 | 6144 | 0.005 | 0.02 | 0.03 | 0.02 |
| 5 | 6144 | 0.01 | 0.03 | 0.05 | 0.02 |
| 5 | 6144 | 0.02 | 0.05 | >0.05 (max 0.3630) | 0.05 |
| 5 | 8192 | 0.005 | 0.02 | 0.03 | 0.02 |
| 5 | 8192 | 0.01 | 0.03 | 0.05 | 0.02 |
| 5 | 8192 | 0.02 | 0.05 | >0.05 (max 0.3820) | 0.05 |

## Validation scope

The CPU-GPU validation (mid-power differences 0.016, 0.025, and
0.0195, within Monte Carlo error) applies to the earlier paired-row
power analysis only. It does not validate this difference-in-differences
generator, which additionally models training-seed variance.
