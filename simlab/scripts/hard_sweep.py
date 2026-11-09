"""Supply-ratio sweep to pick the Hard preset.

Runs balanced_5 config with varying food.supply_ratio for the sensor
heuristic and heuristic_noflee, 20 seeds, cap 10000, multiprocessing.
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
RATIOS = [0.8, 1.0, 1.2, 1.5]
POLICIES = ["heuristic", "heuristic_noflee"]


def worker(task: tuple[str, float, int]) -> dict[str, Any]:
    policy, ratio, seed = task
    track, food_left = run_episode(
        policy,
        seed,
        agents=AGENTS,
        cap=CAP,
        config_path=CONFIG,
        supply_ratio=ratio,
    )
    classes = [t["class"] for t in track.values()]
    deaths = [t["death_step"] for t in track.values()]
    return {"seed": seed, "classes": classes, "deaths": deaths, "food_left": food_left}


def main() -> None:
    tasks = []
    for policy in POLICIES:
        for ratio in RATIOS:
            for seed in range(SEEDS):
                tasks.append((policy, ratio, seed))

    with Pool() as pool:
        results = pool.map(worker, tasks)

    header = (
        f"{'policy':<18}{'ratio':>6}{'cap%':>7}{'95% CI':>14}"
        f"{'median':>8}{'starve':>8}{'hazard':>8}{'trunc':>8}{'food_left':>10}"
    )
    print(header)
    print("-" * len(header))

    idx = 0
    for policy in POLICIES:
        for ratio in RATIOS:
            rows = results[idx: idx + SEEDS]
            idx += SEEDS
            s = summarize(rows)
            starve_share = 100.0 * s["starve"] / max(1, s["n"] - s["trunc"])
            print(
                f"{policy:<18}{ratio:6.1f}"
                f"{s['cap_pct']:6.1f}%  "
                f"[{s['ci'][0]:5.1f},{s['ci'][1]:5.1f}]  "
                f"{s['median']:>8}  "
                f"{s['starve']:>8}{s['hazard']:>8}{s['trunc']:>8}  "
                f"{s['food_left_mean']:8.1f}   starve%ofDeaths={starve_share:.1f}"
            )


if __name__ == "__main__":
    main()
