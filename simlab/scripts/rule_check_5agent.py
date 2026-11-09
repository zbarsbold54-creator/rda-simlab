"""5-agent Balanced reference check (decision rule).

Runs heuristic, heuristic_noflee and random at 5 agents, 20 seeds,
cap 10000, and evaluates the revised per-agent decision rule.

Rule (5-agent reference only):
  - sensor heuristic cap in [40, 75]% of agents
  - random cap < 10%
  - starved-with->=50%-food-uneaten < 10% of agents
  - heuristic hazard deaths in [5, 50]% of agents
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode_lib import run_episode

SEEDS = 20
AGENTS = 5
CAP = 10000
CONFIG = "configs/balanced.toml"
CAPACITY = 400.0
POLICIES = ["heuristic", "heuristic_noflee", "random"]


def worker(task: tuple[str, int]) -> tuple[dict[str, dict], float]:
    policy, seed = task
    return run_episode(policy, seed, agents=AGENTS, cap=CAP, config_path=CONFIG)


def main() -> None:
    tasks = [(p, s) for p in POLICIES for s in range(SEEDS)]
    with Pool() as pool:
        results = pool.map(worker, tasks)

    stats = {}
    idx = 0
    for p in POLICIES:
        chunk = results[idx: idx + SEEDS]
        idx += SEEDS
        n = 0
        capn = 0
        starve = 0
        hazard = 0
        trunc = 0
        starved_with_food = 0
        surv = []
        for track, food_left in chunk:
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
                    if food_left >= 0.5 * CAPACITY:
                        starved_with_food += 1
        surv.sort()
        stats[p] = {
            "n": n,
            "cap_pct": 100.0 * capn / n,
            "median": surv[len(surv) // 2],
            "starve": starve,
            "hazard": hazard,
            "trunc": trunc,
            "starved_with_food_pct": 100.0 * starved_with_food / n,
        }

    print(f"5-agent Balanced reference, {SEEDS} seeds, cap {CAP}, agents {AGENTS}\n")
    for p in POLICIES:
        s = stats[p]
        print(
            f"{p:<18} cap={s['cap_pct']:5.1f}%  median={s['median']}  "
            f"split=({s['starve']},{s['hazard']},{s['trunc']})  "
            f"starved_with_food={s['starved_with_food_pct']:.1f}%"
        )

    print("\nDecision rule:")
    h = stats["heuristic"]
    r = stats["random"]

    def check(name: str, value: float, lo: float, hi: float | None) -> bool:
        ok = value >= lo and (hi is None or value <= hi)
        rng = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {value:.1f}%  (needed {rng})")
        return ok

    all_ok = True
    all_ok &= check("heuristic cap", h["cap_pct"], 40, 75)
    all_ok &= check("random cap", r["cap_pct"], 0, 10)
    all_ok &= check("starved-with-food (agents)", h["starved_with_food_pct"], 0, 10)
    all_ok &= check("heuristic hazard deaths", h["hazard"] * 100 / h["n"], 5, 50)

    print(f"\nRESULT: {'ALL PASS' if all_ok else 'FAILED'}")


if __name__ == "__main__":
    main()
