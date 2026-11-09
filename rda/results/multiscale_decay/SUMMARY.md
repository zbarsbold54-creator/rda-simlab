# Multi-timescale RDA decay screen

Code commit: 5bb8933637685e4cae231539c33038b8eb4e2eaf
Branch: exp/multiscale-decay-20261011

## Training results

| run | final val BPB | val@100 | val@300 | val@600 | median ms/step | peak MiB | EXIT |
|---|---:|---:|---:|---:|---:|---:|---:|
| D | 1.275386 | 1.801160 | 1.462288 | 1.275386 | 2018.435 | 4591.93 | 0 |
| D_const | 1.274947 | 1.801550 | 1.461686 | 1.274947 | 1744.195 | 5172.63 | 0 |
| B' s43 reference | 1.275198 | — | — | — | — | — | reference |
| C s43 reference | 1.274950 | — | — | — | — | — | reference |

## Ablations

| checkpoint | V0 | V1 delta | V2 delta | V3 delta |
|---|---:|---:|---:|---:|
| D | 1.275386 | +0.071142 | +0.001628 | +0.001070 |
| D_const | 1.274946 | +0.000000 | +0.000000 | +0.000000 |

D V0 checkpoint BPB check: recorded 1.275385551, reevaluated 1.275385551, absolute difference 0.
D_const V0 checkpoint BPB check: recorded 1.274946509, reevaluated 1.274946364, absolute difference 1.444e-7.
D_const variants V0-V3 spread: 0.
Largest absolute per-drive scan error against sequential fp32 recurrence: 5.57303429e-6.
Largest decay**(-1024) bound in float64: 7.17249632061e46.

## Exact commands

Default leak regression:
`C:\AI\nanochat\.venv\Scripts\python.exe diagnostics\test_no_label_leak.py`

Scheduled leak regression:
`C:\AI\nanochat\.venv\Scripts\python.exe diagnostics\test_no_label_leak.py --rda-decays 0.9,0.95,0.975,0.9875,0.99375,0.996875,0.9984375,0.99921875`

Screen driver:
`$env:NANOCHAT_DATASET='fineweb-edu'; $env:PYTHONUNBUFFERED='1'; powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ab_v3_dbs2_decay.ps1`

D ablation:
`C:\AI\nanochat\.venv\Scripts\python.exe -m diagnostics.ablate_state --checkpoint-dir C:\Users\HiTech\.cache\nanochat\base_checkpoints\ab-v3-D-s43-dbs2-decay --step 600 --device-batch-size 2 --eval-tokens 524288 --seed 43 --variants V0,V1,V2,V3`

D_const ablation:
`C:\AI\nanochat\.venv\Scripts\python.exe -m diagnostics.ablate_state --checkpoint-dir C:\Users\HiTech\.cache\nanochat\base_checkpoints\ab-v3-D_const-s43-dbs2-decay --step 600 --device-batch-size 2 --eval-tokens 524288 --seed 43 --variants V0,V1,V2,V3`

B' s43 default-path regression:
`C:\AI\nanochat\.venv\Scripts\python.exe -m diagnostics.ablate_state --checkpoint-dir C:\Users\HiTech\.cache\nanochat\base_checkpoints\ab-v3-Bprime-s43-dbs2-full --step 600 --device-batch-size 2 --eval-tokens 524288 --seed 43 --variants V0`

## Anomalies

ANOMALY: Direct invocations of nvidia-smi succeeded; the original same-process driver invocation failed. Relaunch as the requested fresh PowerShell child process succeeded without a sleep.
ANOMALY: Initial PowerShell launch attempt failed during the memory query before training; the later child-process launch passed the preflight, and no run value or threshold changed.
ANOMALY: Existing unrelated worktree modifications were left untouched and excluded from both commits.
