"""Hazard-count sweep to pick the hazard-based Hard preset.

Runs balanced_5 config with varying hazard.count for the sensor heuristic and
random, 20 seeds, cap 10000, multiprocessing. supply_ratio stays at 2.0.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode_lib import run_episode, summarize

REPO = Path(__file__).resolve().parents[1]
CONFIG = str(REPO / "configs/balanced.toml")
SEEDS = 20
CAP = 10000
AGENTS = 5
HAZARD_COUNTS = [4, 6, 8, 12, 16]
POLICIES = ["heuristic", "random"]


def worker(task: tuple[str, int, int]) -> dict[str, Any]:
    policy, hazard_count, seed = task
    track, food_left = run_episode(
        policy,
        seed,
        agents=AGENTS,
        cap=CAP,
        config_path=CONFIG,
        hazard_count=hazard_count,
    )
    classes = [t["class"] for t in track.values()]
    deaths = [t["death_step"] for t in track.values()]
    return {"seed": seed, "classes": classes, "deaths": deaths, "food_left": food_left}


def main() -> None:
    tasks = []
    for policy in POLICIES:
        for hazard_count in HAZARD_COUNTS:
            for seed in range(SEEDS):
                tasks.append((policy, hazard_count, seed))

    with Pool() as pool:
        results = pool.map(worker, tasks)

    header = (
        f"{'policy':<10}{'haz':>5}{'cap%':>7}{'95% CI':>14}"
        f"{'median':>8}{'starve':>8}{'hazard':>8}{'trunc':>8}{'food_left':>10}"
    )
    print(header)
    print("-" * len(header))

    idx = 0
    for policy in POLICIES:
        for hazard_count in HAZARD_COUNTS:
            rows = results[idx: idx + SEEDS]
            idx += SEEDS
            s = summarize(rows)
            print(
                f"{policy:<10}{hazard_count:5d}"
                f"{s['cap_pct']:6.1f}%  "
                f"[{s['ci'][0]:5.1f},{s['ci'][1]:5.1f}]  "
                f"{s['median']:>8}  "
                f"{s['starve']:>8}{s['hazard']:>8}{s['trunc']:>8}  "
                f"{s['food_left_mean']:8.1f}"
            )


if __name__ == "__main__":
    main()
