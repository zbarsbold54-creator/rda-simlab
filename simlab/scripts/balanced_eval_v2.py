"""Step 6: Balanced evaluation across agent counts and policies.

Uses stdlib multiprocessing (one process per seed). Reports, per cell:
cap percent, median survival, death-cause split, mean food left, and the
count of agents that starved while at least 50% of food was uneaten.

Also verifies multiprocessing == serial on a 2-seed sample and runs a
20-seed final-quality run at 5 agents for heuristic and random.
"""

from __future__ import annotations

import statistics
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode_lib import run_episode

AGENTS = [1, 5, 20]
POLICIES = ["heuristic", "heuristic_noflee", "oracle", "random"]
CAP = 10000
SEEDS_EVAL = 15
SEEDS_FINAL = 20
CONFIG = "configs/balanced.toml"


def worker(task: tuple[str, int, int]) -> tuple[dict[str, Any], float]:
    policy, seed, agents = task
    track, food_left = run_episode(
        policy, seed, agents=agents, cap=CAP, config_path=CONFIG
    )
    return track, food_left


def summarize(tracks: list[dict[str, dict[str, Any]]], food_lefts: list[float]) -> dict[str, Any]:
    surv = []
    causes = {"starvation": 0, "hazard": 0, "truncation": 0}
    cap_count = 0
    starved_with_food = 0
    total = 0
    for track, food_left in zip(tracks, food_lefts):
        capacity = 400.0  # 20 patches x 20 max_amount
        for t in track.values():
            total += 1
            surv.append(t["death_step"])
            causes[t["class"]] += 1
            if t["class"] == "truncation":
                cap_count += 1
            if t["class"] == "starvation" and food_left >= 0.5 * capacity:
                starved_with_food += 1
    surv.sort()
    return {
        "total": total,
        "cap_pct": 100.0 * cap_count / total if total else 0.0,
        "median": surv[len(surv) // 2] if surv else 0,
        "causes": causes,
        "starved_with_food": starved_with_food,
        "mean_food_left": statistics.mean(food_lefts) if food_lefts else 0.0,
    }


def run_grid(policies: list[str], agents_list: list[int], seeds: range) -> dict[tuple[str, int], dict[str, Any]]:
    tasks = [(p, s, a) for a in agents_list for p in policies for s in seeds]
    with Pool() as pool:
        results = pool.map(worker, tasks)
    grid: dict[tuple[str, int], dict[str, Any]] = {}
    idx = 0
    for a in agents_list:
        for p in policies:
            chunk = results[idx: idx + len(seeds)]
            idx += len(seeds)
            tracks = [c[0] for c in chunk]
            food_lefts = [c[1] for c in chunk]
            grid[(p, a)] = summarize(tracks, food_lefts)
    return grid


def verify_mp_equals_serial() -> bool:
    policies = ["heuristic"]
    seeds = range(2)
    tasks = [(p, s, 5) for p in policies for s in seeds]
    with Pool() as pool:
        mp_results = pool.map(worker, tasks)
    serial = [worker(t) for t in tasks]
    for m, s in zip(mp_results, serial):
        # compare death-step/class per agent and food_left
        for aid, mt in m[0].items():
            st = s[0][aid]
            if (mt["death_step"], mt["class"]) != (st["death_step"], st["class"]):
                print("MISMATCH", aid, mt["death_step"], mt["class"], st["death_step"], st["class"])
                return False
        if abs(m[1] - s[1]) > 1e-9:
            return False
    return True


def main() -> None:
    print("step6: verify multiprocessing == serial (2 seeds, heuristic, 5 agents)")
    ok = verify_mp_equals_serial()
    print("  mp == serial:", ok)

    print(f"\nEval grid (15 seeds, cap {CAP}, Balanced):")
    grid = run_grid(POLICIES, AGENTS, range(SEEDS_EVAL))
    header = f"{'policy':<18}{'ag':>3}{'cap%':>7}{'median':>8}{'starve':>7}{'hazard':>7}{'trunc':>7}{'food_left':>9}{'svfood':>7}"
    print(header)
    for a in AGENTS:
        for p in POLICIES:
            g = grid[(p, a)]
            c = g["causes"]
            print(
                f"{p:<18}{a:>3}{g['cap_pct']:>7.1f}{g['median']:>8}"
                f"{c['starvation']:>7}{c['hazard']:>7}{c['truncation']:>7}"
                f"{g['mean_food_left']:>9.1f}{g['starved_with_food']:>7}"
            )

    print("\nFinal 20-seed run at 5 agents (heuristic, random):")
    final = run_grid(["heuristic", "random"], [5], range(SEEDS_FINAL))
    for p in ("heuristic", "random"):
        g = final[(p, 5)]
        c = g["causes"]
        print(
            f"  {p:<12} cap%={g['cap_pct']:.1f} median={g['median']} "
            f"split=({c['starvation']},{c['hazard']},{c['truncation']}) "
            f"food_left={g['mean_food_left']:.1f} starved_with_food={g['starved_with_food']}"
        )


if __name__ == "__main__":
    main()
