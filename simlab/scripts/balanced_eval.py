"""Step 4: evaluate the Balanced preset (radius 16, regrow 0.02)."""

from __future__ import annotations

import sys
import time
from statistics import mean, median

sys.path.insert(0, "C:\\AI\\SimulationCode")

from episode_lib import run_episode, split

POLICIES = ("heuristic", "oracle_heuristic", "random")
AGENT_COUNTS = (1, 5, 20)
CAP = 10000
BUDGET_SECONDS = 900.0  # ~15 min
DRAIN_PER_AGENT = (0.08 + 0.04 * 6.4) * 0.1  # full-speed pursuit drain
COLS = ("policy", "agents", "median", "cap_pct", "starve/hazard/trunc", "mean_food_left")


def breakeven() -> None:
    patches = 20
    print("breakeven_regrow_rate (assumes full-speed pursuit drain 0.0336/agent/step, 20 patches)")
    for agents in AGENT_COUNTS:
        rate = agents * DRAIN_PER_AGENT / patches
        print(f"  agents={agents:2d} total_drain={agents * DRAIN_PER_AGENT:.4f} breakeven={rate:.5f}")


def calibrate(seed: int) -> dict[tuple[str, int], float]:
    timings: dict[tuple[str, int], float] = {}
    for policy in POLICIES:
        for agents in AGENT_COUNTS:
            start = time.perf_counter()
            run_episode(policy, seed, agents, CAP, config_path="configs/balanced.toml")
            timings[(policy, agents)] = time.perf_counter() - start
    return timings


def main() -> None:
    breakeven()
    start = time.perf_counter()
    # Calibrate with one episode per cell to plan the seed budget.
    timings = calibrate(1)
    per_cell = sum(timings.values()) / len(timings)
    cells = len(POLICIES) * len(AGENT_COUNTS)
    seeds = 20
    if seeds * cells * per_cell > BUDGET_SECONDS:
        seeds = max(1, int(BUDGET_SECONDS // (cells * per_cell)))
        print(f"reduced_seeds=20->{seeds} (estimated {cells * seeds * per_cell:.0f}s vs budget {BUDGET_SECONDS:.0f}s)")
    else:
        print(f"seeds=20 (estimated {cells * seeds * per_cell:.0f}s)")

    print(f"agents=...  seeds={seeds} step_cap={CAP} config=balanced.toml")
    print("policy agents median_survival cap_pct mean_food_left starve/hazard/trunc")
    results: dict[tuple[str, int], tuple[float, float, float, tuple[int, int, int]]] = {}
    for policy in POLICIES:
        for agents in AGENT_COUNTS:
            surv: list[int] = []
            outcomes: list[str] = []
            food_left: list[float] = []
            for seed in range(1, seeds + 1):
                track, fl = run_episode(
                    policy, seed, agents, CAP, config_path="configs/balanced.toml"
                )
                for t in track.values():
                    surv.append(t["death_step"])
                    outcomes.append(t["class"])
                food_left.append(fl)
            s = split(outcomes)
            results[(policy, agents)] = (
                median(surv),
                100.0 * s[2] / len(outcomes),
                mean(food_left),
                s,
            )
            print(
                f"{policy:18s} {agents:2d} {median(surv):14.0f} "
                f"{100.0 * s[2] / len(outcomes):7.1f} {mean(food_left):13.1f} "
                f"{s[0]}/{s[1]}/{s[2]}"
            )
    elapsed = time.perf_counter() - start
    print(f"elapsed_seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
