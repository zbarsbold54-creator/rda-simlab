"""Measure eval wall-time scaling vs worker count (round 11, step 1).

Prints, per worker count: wall time, total world steps executed (summed over
all seeds/workers), and aggregate world-steps-per-second.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simlab.eval_lib import REPO, aggregate, run_episode_metrics

POLICY = "heuristic"
DRIVE = "simple"
CONFIG = REPO / "configs" / "balanced.toml"


def _eval_worker(task):
    policy, drive, seed, agents, cap, config_path = task
    return run_episode_metrics(policy, seed, agents, cap, str(config_path), drive=drive)


def run(workers: int, seeds: int, agents: int, cap: int, set_env: bool) -> dict:
    if set_env:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
    tasks = [(POLICY, DRIVE, seed, agents, cap, CONFIG) for seed in range(seeds)]
    t0 = time.monotonic()
    with Pool(processes=workers) as pool:
        results = pool.map(_eval_worker, tasks)
    wall = time.monotonic() - t0
    agg = aggregate(results, wall, cap)
    total_steps = sum(
        max(a["death_step"] for a in r["per_agent"]) for r in results
    )
    return {
        "workers": workers,
        "wall_s": round(wall, 2),
        "total_world_steps": total_steps,
        "steps_per_second": round(total_steps / wall, 1),
        "agg_hash": agg["hash"],
        "cap_pct": round(agg["cap_pct"], 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--cap", type=int, default=10000)
    parser.add_argument("--set-env", action="store_true")
    args = parser.parse_args()
    for workers in args.workers:
        info = run(workers, args.seeds, args.agents, args.cap, args.set_env)
        print(json.dumps(info))


if __name__ == "__main__":
    raise SystemExit(main())
