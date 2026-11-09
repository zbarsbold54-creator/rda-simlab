# DBS=2 A/B Results

This is a controlled comparison of baseline A (no `--rda` flags), arm B' (`--rda`), and arm C (`--rda --rda-const-state`) at device batch size 2, 600 iterations, seeds 42/43/44, using the `fineweb-edu` dataset.

| arm | seed | final val BPB | val@100 | val@300 | val@600 | median ms/step (11-600) | max ms/step | peak MiB | EXIT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 42 | 1.300680 | 1.809270 | 1.481088 | 1.300680 | 1273.24 | 1315.21 | 3707.29 | 0 |
| A | 43 | 1.287114 | 1.803028 | 1.464684 | 1.287114 | 1274.87 | 1370.56 | 3707.29 | 0 |
| A | 44 | 1.293590 | 1.805501 | 1.472777 | 1.293590 | 1266.21 | 1316.36 | 3707.29 | 0 |
| B' | 42 | 1.277996 | 1.802834 | 1.464161 | 1.277996 | 1986.50 | 2120.97 | 4591.88 | 0 |
| B' | 43 | 1.275198 | 1.802521 | 1.462537 | 1.275198 | 1992.29 | 2077.82 | 4591.88 | 0 |
| B' | 44 | 1.275858 | 1.796073 | 1.458186 | 1.275858 | 1996.53 | 2153.61 | 4591.88 | 0 |
| C | 42 | 1.277748 | 1.804009 | 1.463873 | 1.277748 | 1727.87 | 1991.41 | 5172.52 | 0 |
| C | 43 | 1.274950 | 1.801848 | 1.461493 | 1.274950 | 1721.88 | 1803.31 | 5172.52 | 0 |
| C | 44 | 1.274079 | 1.798318 | 1.455649 | 1.274079 | 1732.24 | 1794.84 | 5172.52 | 0 |

Final val BPB mean and sample standard deviation:
- A: mean 1.293794667, SD 0.006785315
- B': mean 1.276350667, SD 0.001462615
- C: mean 1.275592333, SD 0.001916986

Paired final-val-BPB differences:
- B' minus A: seed 42 -0.022684; seed 43 -0.011916; seed 44 -0.017732; mean -0.017444
- C minus B': seed 42 -0.000248; seed 43 -0.000248; seed 44 -0.001779; mean -0.000758333

Commands used:
- Launcher: `powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ab_v3_dbs2_full.ps1`
- Shared flags: `--seed <seed> --depth=8 --aspect-ratio=96 --head-dim=128 --max-seq-len=1024 --window-pattern=L --device-batch-size=2 --total-batch-size=32768 --num-iterations=600 --eval-every=50 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1`
- A: shared flags only; no RDA flags.
- B': shared flags plus `--rda`.
- C: shared flags plus `--rda --rda-const-state`.
- Model tags: `ab-v3-A-s<seed>-dbs2-full`, `ab-v3-Bprime-s<seed>-dbs2-full`, `ab-v3-C-s<seed>-dbs2-full`.

Code under test: `5566d90bc39ad71b06be535ff33a919e1606498f`. Environment capture: [`logs/ab_v3_dbs2_full_env.txt`](./logs/ab_v3_dbs2_full_env.txt).

Anomalies / run notes:
- All nine runs completed with exit 0 and produced the requested validation steps.
- Seed 42 B' and C were already run and are reused from their prior DBS=2 logs; no training rerun was made for those two.
- GPU pre-check was 1310 MiB, below 1500 MiB; no holder stop was triggered.
- The historical A/B launcher used device batch 4 in its shared base flags; this comparison uses device batch 2 as specified for this controlled run.

## Code under test (correction)
The A/B was run from a working tree with uncommitted changes on top of 5566d90. Those changes (the RDA module, --rda / --rda-const-state / --rda-debug-steps flags in base_train.py, loss_eval.py changes) are committed in c19de86e313048bb1a6d9f638f5ab7bd436f9364. To reproduce, check out that commit.
