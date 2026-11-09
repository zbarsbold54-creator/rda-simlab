"""Solo scenario evaluation (configs/solo.toml, 1 agent, 1 hazard).

For each policy (heuristic with flee, heuristic_noflee, random) run 20 seeds
for 10000 steps and report: cap percent, median survival, death-cause split
(starvation / hazard / truncation), and mean food remaining at the end.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode_lib import run_episode

SEEDS = 20
AGENTS = 1
CAP = 10000
CONFIG = "configs/solo.toml"
POLICIES = ["heuristic", "heuristic_noflee", "random"]


def worker(task: tuple[str, int]) -> tuple[dict[str, dict], float]:
    policy, seed = task
    return run_episode(policy, seed, agents=AGENTS, cap=CAP, config_path=CONFIG)


def main() -> None:
    tasks = [(p, s) for p in POLICIES for s in range(SEEDS)]
    with Pool() as pool:
        results = pool.map(worker, tasks)

    idx = 0
    for p in POLICIES:
        chunk = results[idx: idx + SEEDS]
        idx += SEEDS
        n = 0
        capn = 0
        starve = 0
        hazard = 0
        trunc = 0
        surv = []
        food_lefts = []
        for track, food_left in chunk:
            food_lefts.append(food_left)
            for t in track.values():
                n += 1
                surv.append(t["death_step"])
                if t["class"] == "truncation":
                    capn += 1
                    trunc += 1
                elif t["class"] == "hazard":
                    hazard += 1
                else:
                    starve += 1
        surv.sort()
        cap_pct = 100.0 * capn / n
        median = surv[len(surv) // 2]
        mean_left = sum(food_lefts) / len(food_lefts)
        print(
            f"{p:<18} cap={cap_pct:5.1f}%  median={median}  "
            f"split=({starve},{hazard},{trunc})  food_left(mean)={mean_left:.1f}"
        )


if __name__ == "__main__":
    main()
