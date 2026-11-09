"""Report-only world sweep over detection radius and food regrow rate."""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simlab.core.config import load_config
from simlab.core.runner import Runner

SEEDS = range(5)
DETECTION_RADII = (12, 16, 30)
REGROW_RATES = (0.005, 0.01, 0.02, 0.05)
STEP_CAP = 10000
AGENTS = 5


def episode(policy: str, seed: int, radius: int, regrow_rate: float) -> tuple[list[tuple[int, str]], int]:
    base = load_config("configs/harsh_original.toml")
    config = replace(
        base,
        policy=policy,
        hazard=replace(base.hazard, detection_radius=float(radius)),
        food=replace(base.food, regrow_rate=regrow_rate),
    )
    runner = Runner(config, seed=seed, agents=AGENTS)
    runner.setup()
    world = runner._world
    zero_steps: dict[str, int] = {}
    deaths: dict[str, tuple[int, str]] = {}
    for step in range(1, STEP_CAP + 1):
        energy_before = {
            aid: state["energy"]
            for aid, state in world._agents.items()
            if state["is_alive"]
        }
        result = runner.run_step(include_observations=False)
        for aid, previous in energy_before.items():
            state = world._agents[aid]
            if previous > 0.0 and state["energy"] <= 0.0:
                zero_steps.setdefault(aid, step)
            if not state["is_alive"] and aid not in deaths:
                cause = "starvation" if aid in zero_steps else "hazard"
                deaths[aid] = (step, cause)
        if result.terminated:
            break
    outcomes = [
        deaths.get(aid, (STEP_CAP, "truncation")) for aid in world._agents
    ]
    return outcomes, len(world._food_patches)


def split(outcomes: list[tuple[int, str]]) -> tuple[int, int, int]:
    counts = {cause: 0 for cause in ("starvation", "hazard", "truncation")}
    for _, cause in outcomes:
        counts[cause] += 1
    return counts["starvation"], counts["hazard"], counts["truncation"]


def print_breakeven() -> None:
    base = load_config("configs/harsh_original.toml")
    count = min(
        base.food.max_food,
        max(1, int(base.world.width * base.world.height * base.food.spawn_rate / 10)),
    )
    speed = 6.4  # heuristic thrust 0.8 -> speed 8*0.8
    drain = (0.08 + 0.04 * speed) * 0.1
    total_drain = AGENTS * drain
    breakeven = total_drain / count
    print("breakeven_arithmetic (5 moving agents)")
    print(f"  drain_per_step_per_agent={drain:.4f}")
    print(f"  total_drain_per_step={total_drain:.4f}")
    print("  energy_per_food_unit=1.0 (eating transfers food amount to energy 1:1)")
    print(f"  initial_patches={count}")
    print(f"  breakeven_regrow_rate={breakeven:.5f} units/step/patch")
    print(
        "  -> among {0.005,0.01,0.02,0.05}: 0.005 is below supply-equals-drain; "
        "0.01 is near breakeven; 0.02/0.05 are above"
    )


def main() -> None:
    start = time.perf_counter()
    print_breakeven()
    print("agents=5 seeds=5 step_cap=10000 preset=Harsh (original)")
    print(
        "radius regrow_rate heuristic_median heuristic_cap_pct "
        "heuristic_starve/hazard/trunc random_median random_starve/hazard/trunc"
    )
    for radius in DETECTION_RADII:
        for rate in REGROW_RATES:
            heuristic: list[tuple[int, str]] = []
            random: list[tuple[int, str]] = []
            for seed in SEEDS:
                outcomes, _ = episode("heuristic", seed, radius, rate)
                heuristic.extend(outcomes)
                outcomes, _ = episode("random", seed, radius, rate)
                random.extend(outcomes)
            h_split = split(heuristic)
            r_split = split(random)
            cap_percent = 100.0 * h_split[2] / len(heuristic)
            print(
                f"{radius:6d} {rate:10.3f} "
                f"{median(step for step, _ in heuristic):17.0f} "
                f"{cap_percent:16.1f} "
                f"{h_split[0]:4d}/{h_split[1]:4d}/{h_split[2]:4d} "
                f"{median(step for step, _ in random):13.0f} "
                f"{r_split[0]:4d}/{r_split[1]:4d}/{r_split[2]:4d}"
            )
    elapsed = time.perf_counter() - start
    print(f"elapsed_seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
