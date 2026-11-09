"""Step 2: reconcile sensor vs oracle heuristic.

Runs the historical no-flee sensor heuristic and oracle (heuristic_noflee /
oracle_noflee) on the Balanced preset at 5 agents and reports, per policy:

  - mean thrust, drain per step, food units eaten per agent,
  - energy in (from food) versus energy out (movement drain),
  - median survival.

It then estimates survival purely from the energy ledger:
``est_survival = (initial_energy + mean_food_units) / drain_per_step``
to show how much of the sensor-vs-oracle survival gap is explained by thrust
and drain alone versus by food eaten.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from foraging_diag import run_seed

POLICIES = ("heuristic_noflee", "oracle_noflee")
SEEDS = list(range(6))
AGENTS = 5
INITIAL_ENERGY = 100.0


def summarize(policy: str) -> dict[str, float]:
    drain_total = 0.0
    live_total = 0
    food_total = 0.0
    thrust_total = 0.0
    spent_total = 0.0
    survival: list[int] = []
    classes = {"starvation": 0, "hazard": 0, "truncation": 0}
    for seed in SEEDS:
        track = run_seed(seed, agents=AGENTS, policy=policy)
        for m in track.values():
            survival.append(m["survival"])
            drain_total += m["energy_spent"]
            live_total += m["live_steps"]
            food_total += m["food_units"]
            thrust_total += m["thrust_sum"]
            spent_total += m["energy_spent"]
            classes[m["class"]] += 1
    n = len(SEEDS) * AGENTS
    drain_per_step = drain_total / live_total
    food_per_agent = food_total / n
    return {
        "mean_thrust": thrust_total / live_total,
        "drain_per_step": drain_per_step,
        "food_per_agent": food_per_agent,
        "spent_per_agent": spent_total / n,
        "median_survival": statistics.median(survival),
        "est_survival": (INITIAL_ENERGY + food_per_agent) / drain_per_step,
        "classes": classes,
    }


def main() -> None:
    print(f"Reconcile sensor vs oracle at Balanced, {AGENTS} agents, "
          f"{len(SEEDS)} seeds, cap 10000")
    results = {}
    for policy in POLICIES:
        r = summarize(policy)
        results[policy] = r
        print(f"\n{policy}:")
        print(f"  mean thrust = {r['mean_thrust']:.3f}")
        print(f"  drain per step = {r['drain_per_step']:.4f}")
        print(f"  food units eaten per agent = {r['food_per_agent']:.1f}")
        print(f"  energy in (food) per agent = {r['food_per_agent']:.1f}")
        print(f"  energy spent per agent = {r['spent_per_agent']:.1f}")
        print(f"  median survival = {r['median_survival']}")
        print(f"  estimated survival from ledger = {r['est_survival']:.0f}")
        print(f"  death causes = {r['classes']}")
    s, o = results["heuristic_noflee"], results["oracle_noflee"]
    gap = s["median_survival"] - o["median_survival"]
    print(f"\nMedian survival gap (sensor - oracle) = {gap}")
    thr_gap = (100.0 / o["drain_per_step"]) - (100.0 / s["drain_per_step"])
    print(f"Survival difference explained by drain alone "
          f"(feed food): {thr_gap:.0f} steps")
    food_gap = (s["food_per_agent"] - o["food_per_agent"]) / s["drain_per_step"]
    print(f"Survival difference explained by food eaten alone: {food_gap:.0f} steps")


if __name__ == "__main__":
    main()
