"""Report-only tuning sweep over existing finite-food world settings."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simlab.core.config import load_config
from simlab.core.runner import Runner
from simlab.interfaces.policy import HeuristicPolicy, RandomPolicy

SEEDS = range(1, 21)
DETECTION_RADII = (8, 12, 16, 30)
FOOD_FACTORS = (1, 3, 10)
STEP_CAP = 20_000


def episode(seed: int, agents: int, radius: int, food_factor: int, kind: str):
    base = load_config("configs/harsh_original.toml")
    config = replace(
        base,
        hazard=replace(base.hazard, detection_radius=float(radius)),
        food=replace(base.food, spawn_rate=base.food.spawn_rate * food_factor),
    )
    runner = Runner(config, seed=seed, agents=agents)
    runner.setup()
    world = runner._world
    policy = HeuristicPolicy() if kind == "heuristic" else RandomPolicy(seed)
    zero_steps: dict[str, int] = {}
    deaths: dict[str, tuple[int, str]] = {}

    for step in range(1, STEP_CAP + 1):
        food_locations = [patch["position"] for patch in world._food_patches]
        actions = {}
        for agent_id, state in world._agents.items():
            if not state["is_alive"]:
                continue
            if isinstance(policy, HeuristicPolicy):
                actions[agent_id] = policy.select_action(
                    state, {"food_locations": food_locations}, [], step - 1
                )
            else:
                actions[agent_id] = policy.act(None)
        energy_before = {
            agent_id: state["energy"]
            for agent_id, state in world._agents.items()
            if state["is_alive"]
        }
        world.step(
            actions,
            include_observations=False,
            include_snapshot=False,
            include_agent_states=False,
        )
        for agent_id, previous_energy in energy_before.items():
            state = world._agents[agent_id]
            if previous_energy > 0.0 and state["energy"] <= 0.0:
                zero_steps.setdefault(agent_id, step)
            if not state["is_alive"] and agent_id not in deaths:
                cause = "starvation" if agent_id in zero_steps else "hazard"
                deaths[agent_id] = (step, cause)
        if not any(state["is_alive"] for state in world._agents.values()):
            break

    outcomes = []
    for agent_id in world._agents:
        survival, cause = deaths.get(agent_id, (STEP_CAP, "truncation"))
        outcomes.append((survival, cause))
    return outcomes, len(world._food_patches)


def split(outcomes: list[tuple[int, str]]) -> tuple[int, int, int]:
    counts = {cause: 0 for cause in ("starvation", "hazard", "truncation")}
    for _, cause in outcomes:
        counts[cause] += 1
    return counts["starvation"], counts["hazard"], counts["truncation"]


def print_energy_budget() -> None:
    config = load_config("configs/harsh_original.toml")
    runner = Runner(config, seed=1, agents=1)
    runner.setup()
    world = runner._world
    initial_food = sum(patch["amount"] for patch in world._food_patches)
    food_consumed = 0.0
    while world.get_step_count() < STEP_CAP:
        food_before = sum(patch["amount"] for patch in world._food_patches)
        world.step(
            {"agent_0": runner._default_action("agent_0")},
            include_observations=False,
            include_snapshot=False,
            include_agent_states=False,
        )
        food_after = sum(patch["amount"] for patch in world._food_patches)
        food_consumed += food_before - food_after
        if not world._agents["agent_0"]["is_alive"]:
            break
    drain = (0.08 + 0.04 * 6.4) * world.dt
    initial_energy = config.agent.initial_energy
    print(
        f"energy_budget seed=1 survival={world.get_step_count()} "
        f"starting_energy={initial_energy:g} heuristic_speed=6.4 "
        f"drain_per_step={drain:.4f} energy_only_steps={initial_energy / drain:.1f} "
        f"starvation_health_steps={100 / 0.5:.0f} "
        f"food_consumed={food_consumed:.4f} food_eaten_ticks="
        f"{world._agents['agent_0']['food_eaten']} initial_food={initial_food:.1f} "
        f"regrowth_per_step=0"
    )


def main() -> None:
    base = load_config("configs/harsh_original.toml")
    print("preset=Harsh (original) agents=5 seeds=20 step_cap=20000")
    print(
        "food_regrowth=not implemented; swept food.spawn_rate only changes initial "
        "patch count and is capped by max_food"
    )
    print(
        "detection food_factor initial_patches heuristic_median heuristic_cap_pct "
        "heuristic_starve/hazard/trunc random_median random_starve/hazard/trunc"
    )
    for radius in DETECTION_RADII:
        for factor in FOOD_FACTORS:
            heuristic: list[tuple[int, str]] = []
            random: list[tuple[int, str]] = []
            patch_count = 0
            for seed in SEEDS:
                outcomes, patch_count = episode(seed, 5, radius, factor, "heuristic")
                heuristic.extend(outcomes)
                outcomes, patch_count = episode(seed, 5, radius, factor, "random")
                random.extend(outcomes)
            h_split = split(heuristic)
            r_split = split(random)
            cap_percent = 100.0 * h_split[2] / len(heuristic)
            print(
                f"{radius:9d} {factor:11d} {patch_count:14d} "
                f"{median(step for step, _ in heuristic):16.0f} "
                f"{cap_percent:17.1f} {h_split[0]:4d}/{h_split[1]:4d}/{h_split[2]:4d} "
                f"{median(step for step, _ in random):13.0f} "
                f"{r_split[0]:4d}/{r_split[1]:4d}/{r_split[2]:4d}"
            )
    for kind in ("heuristic", "random"):
        outcomes, patches = episode(1, 1, 30, 1, kind)
        survival, cause = outcomes[0]
        print(
            f"one_agent seed=1 policy={kind} detection=30 food_factor=1 "
            f"initial_patches={patches} survival={survival} cause={cause}"
        )
    count = min(
        base.food.max_food,
        max(1, int(base.world.width * base.world.height * base.food.spawn_rate / 10)),
    )
    print(
        f"current_food_budget=initial_energy:{base.agent.initial_energy:g} "
        f"food_patches:{count} food_amount_per_patch:20 total_food:{count * 20:g} "
        "regrowth_per_step:0"
    )
    print_energy_budget()


if __name__ == "__main__":
    main()
